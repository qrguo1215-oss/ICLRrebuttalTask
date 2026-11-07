#!/usr/bin/env python3
"""Validate per-case CPS states and build compact legacy-compatible sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--expected-total", type=int, default=1000)
    parser.add_argument("--expected-per-group", type=int, default=500)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_total:
        raise ValueError(f"expected {args.expected_total} manifest rows, found {len(rows)}")
    if len({int(row["case_idx"]) for row in rows}) != len(rows):
        raise ValueError("manifest contains duplicate case_idx values")

    grouped = {True: [], False: []}
    audit = []
    for row in sorted(rows, key=lambda item: int(item["case_idx"])):
        case_idx = int(row["case_idx"])
        path = args.input_dir / "cases" / f"case_{case_idx:04d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        full_text = row["prompt"] + row["response"]
        expected_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        steps = payload["step"]
        checks = payload["check_index"]
        switches = payload["switch_index"]
        if payload.get("text_sha256") != expected_hash:
            raise ValueError(f"text hash mismatch for case {case_idx}")
        if int(payload.get("layer", -1)) != args.layer:
            raise ValueError(f"layer mismatch for case {case_idx}")
        if bool(payload.get("correct")) != bool(row["correct"]):
            raise ValueError(f"correctness mismatch for case {case_idx}")
        if steps.ndim != 2 or tuple(steps.shape)[1:] != (args.hidden_size,):
            raise ValueError(f"bad hidden shape for case {case_idx}: {tuple(steps.shape)}")
        if not torch.isfinite(steps).all():
            raise ValueError(f"non-finite hidden state for case {case_idx}")
        if any(int(index) >= len(steps) for index in checks.tolist() + switches.tolist()):
            raise ValueError(f"out-of-range heuristic index for case {case_idx}")
        grouped[bool(row["correct"])].append(payload)
        audit.append(
            {
                "case_idx": case_idx,
                "source_idx": int(row["source_idx"]),
                "correct": bool(row["correct"]),
                "num_tokens": int(payload["num_tokens"]),
                "num_steps": int(payload["num_steps"]),
                "text_sha256": expected_hash,
            }
        )

    summary = {"layer": args.layer, "hidden_size": args.hidden_size, "groups": {}}
    for correct, label in ((True, "correct"), (False, "incorrect")):
        items = grouped[correct]
        if len(items) != args.expected_per_group:
            raise ValueError(f"expected {args.expected_per_group} {label} cases, found {len(items)}")
        hidden = [{} for _ in range(args.layer + 1)]
        prompts = []
        case_map = []
        for example_id, payload in enumerate(items):
            hidden[args.layer][example_id] = {
                "step": payload["step"],
                "check_index": payload["check_index"],
                "switch_index": payload["switch_index"],
            }
            prompts.append(payload["prompt"])
            case_map.append(
                {
                    "example_id": example_id,
                    "case_idx": int(payload["case_idx"]),
                    "source_idx": int(payload["source_idx"]),
                    "num_steps": int(payload["num_steps"]),
                }
            )
        group_dir = args.output_dir / label
        atomic_torch_save(hidden, group_dir / "hidden.pt")
        atomic_text_write(json.dumps(prompts, ensure_ascii=False) + "\n", group_dir / "prompts.json")
        atomic_text_write(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in case_map),
            group_dir / "case_map.jsonl",
        )
        summary["groups"][label] = {
            "cases": len(items),
            "steps": sum(int(item["num_steps"]) for item in items),
            "hidden_path": str(group_dir / "hidden.pt"),
            "prompts_path": str(group_dir / "prompts.json"),
        }

    atomic_text_write(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audit),
        args.output_dir / "audit.jsonl",
    )
    atomic_text_write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
