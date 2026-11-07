#!/usr/bin/env python3
"""Extract CPS/localBest boundary states from a frozen common-case manifest.

Each case is saved atomically and independently so workers can be resumed or
redistributed without repeating completed forward passes.  The boundary and
layer conventions intentionally match the historical Taxonomy/SEAL CPS code:
hidden_states[20] at double-newline tokens inside ``<think>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECK_WORDS = (
    "verify",
    "make sure",
    "hold on",
    "think again",
    "'s correct",
    "'s incorrect",
    "let me check",
    "seems right",
)
CHECK_PREFIXES = ("wait",)
SWITCH_WORDS = (
    "think differenly",
    "another way",
    "another approach",
    "another method",
    "another solution",
    "another strategy",
    "another technique",
)
SWITCH_PREFIXES = ("alternatively",)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cps_indices(text: str, tokenizer, split_ids: set[int]) -> tuple[list[int], list[int], list[int]]:
    token_ids = tokenizer.encode(text)
    think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    if think_begin_id not in token_ids:
        return [], [], []

    start = token_ids.index(think_begin_id) + 1
    end = token_ids.index(think_end_id, start) if think_end_id in token_ids[start:] else len(token_ids)
    think_ids = token_ids[start:end]
    boundaries = [idx for idx, token_id in enumerate(think_ids) if token_id in split_ids]
    boundaries.append(len(think_ids))

    step_positions: list[int] = []
    check_indices: list[int] = []
    switch_indices: list[int] = []
    for step_idx in range(len(boundaries) - 1):
        step_positions.append(boundaries[step_idx] + start)
        step = tokenizer.decode(think_ids[boundaries[step_idx] + 1 : boundaries[step_idx + 1]])
        step = step.strip(" \n").lower()
        if step.startswith(CHECK_PREFIXES) or any(word in step for word in CHECK_WORDS):
            check_indices.append(step_idx)
        elif step.startswith(SWITCH_PREFIXES) or any(word in step for word in SWITCH_WORDS):
            switch_indices.append(step_idx)
    return step_positions, check_indices, switch_indices


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_completed(path: Path, expected_hash: str, layer: int, hidden_size: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        steps = payload["step"]
        return (
            payload.get("text_sha256") == expected_hash
            and int(payload.get("layer", -1)) == layer
            and steps.ndim == 2
            and int(steps.shape[1]) == hidden_size
            and bool(torch.isfinite(steps).all())
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    args = parser.parse_args()

    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"rank {args.rank} is outside world size {args.world_size}")

    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    records = read_jsonl(args.manifest)
    assigned = [item for item in records if int(item["case_idx"]) % args.world_size == args.rank]
    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    split_ids = {token_id for token, token_id in tokenizer.get_vocab().items() if "ĊĊ" in token}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to("cuda")
    model.eval()
    hidden_size = int(model.config.hidden_size)
    num_hidden = int(model.config.num_hidden_layers) + 1
    if not 0 <= args.layer < num_hidden:
        raise ValueError(f"layer {args.layer} is outside [0, {num_hidden})")

    completed = 0
    extracted = 0
    for item in tqdm(assigned, desc=f"cps-hidden-rank-{args.rank}"):
        case_idx = int(item["case_idx"])
        full_text = item["prompt"] + item["response"]
        text_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        output_path = case_dir / f"case_{case_idx:04d}.pt"
        if load_completed(output_path, text_hash, args.layer, hidden_size):
            completed += 1
            continue

        batch = tokenizer(full_text, return_tensors="pt")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        with torch.inference_mode():
            output = model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        step_positions, check_indices, switch_indices = cps_indices(full_text, tokenizer, split_ids)
        selected = output.hidden_states[args.layer][0]
        if step_positions:
            position_index = torch.tensor(step_positions, dtype=torch.long, device=selected.device)
            step_hidden = selected.index_select(0, position_index).detach().to("cpu")
        else:
            step_hidden = torch.empty((0, hidden_size), dtype=dtype)

        payload = {
            "case_idx": case_idx,
            "source_idx": int(item["source_idx"]),
            "correct": bool(item["correct"]),
            "layer": args.layer,
            "hidden_size": hidden_size,
            "dtype": str(step_hidden.dtype),
            "step": step_hidden,
            "check_index": torch.tensor(check_indices, dtype=torch.long),
            "switch_index": torch.tensor(switch_indices, dtype=torch.long),
            "num_tokens": int(batch["input_ids"].shape[1]),
            "num_steps": int(step_hidden.shape[0]),
            "text_sha256": text_hash,
            "prompt": full_text,
        }
        atomic_torch_save(payload, output_path)
        extracted += 1

        del output, selected, step_hidden, batch, payload
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "assigned": len(assigned),
                "already_complete": completed,
                "extracted": extracted,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
