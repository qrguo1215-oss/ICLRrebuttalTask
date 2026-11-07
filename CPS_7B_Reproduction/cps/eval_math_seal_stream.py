#!/usr/bin/env python3
"""Run SEAL MATH500 intervention with indexed, resumable JSONL output."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PREFIX = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"


def extract_box(text: str) -> str:
    answer = text.split("boxed")[-1]
    if not answer:
        return ""
    if answer[0] != "{":
        return answer.split("$")[0].strip()
    depth = 1
    result = []
    for char in answer[1:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        result.append(char)
    return "".join(result)


def trim_output(output: str) -> str:
    for prefix in ("Answer the following question", "Question:", "Comment:"):
        if prefix in output:
            output = output.split(prefix)[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--steer-coef", type=float, default=-1.0)
    parser.add_argument("--math500-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=10000)
    args = parser.parse_args()

    random.seed(42)
    sys.path.insert(0, str(args.seal_dir.resolve()))
    from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM

    with args.math500_path.open(encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    all_rows = [
        {
            "idx": idx,
            "question": item["problem"],
            "solution": item["solution"],
            "answer": extract_box(item["solution"]),
        }
        for idx, item in enumerate(raw)
    ]
    rows = all_rows[max(0, args.start) : min(len(all_rows), args.end) : args.stride]
    if not rows:
        raise SystemExit("empty evaluation range")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lock_handle = Path(str(args.output) + ".lock").open("a", encoding="utf-8")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    completed: set[int] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(int(json.loads(line)["idx"]))
            except Exception:
                continue
    if all(row["idx"] in completed for row in rows):
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": PREFIX + "Question: " + row["question"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token) :]
        prompts.append(prompt)

    model = Qwen2ForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()
    vector = torch.load(args.vector, weights_only=True).to(model.device)
    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.layer,
        steer_vec=vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
    )

    for row, prompt in zip(rows, prompts):
        if row["idx"] in completed:
            continue
        inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        model.start_new_round()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_tokens,
                use_cache=True,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        response = trim_output(tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True))
        item = {
            "idx": row["idx"],
            "prompt": prompt,
            "problem": row["question"],
            "answer": row["answer"],
            "solution": row["solution"],
            "model_generation": [response],
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        del inputs, generated
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
