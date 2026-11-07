#!/usr/bin/env python3
"""Validate immutable inputs, rank partitioning, reference vectors, and model shape."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig


EXPECTED = {
    "data/common_500_500.jsonl": "c8c9e6aca704c27ddd9da9430bb4b6388801ccd73bc73bc32675fbcd0d6ca200",
    "data/MATH500/test.jsonl": "59790e031289b97b00aa300b8fb6af5c331931b7b91be2c2cb52b257216ea774",
    "reference/cluster_mean_vectors/cluster_mean_vectors.pt": "9596d4b76a128e01a98483174c88289a44e52e5349b191fb21e768f2f5fb06f8",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model")
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()

    checksums = {}
    for relative, expected in EXPECTED.items():
        path = args.root / relative
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}: {actual}")
        checksums[relative] = actual

    manifest = read_jsonl(args.root / "data/common_500_500.jsonl")
    case_ids = [int(row["case_idx"]) for row in manifest]
    if len(manifest) != 1000 or len(set(case_ids)) != 1000:
        raise ValueError("construction manifest must contain 1000 unique cases")
    correctness = Counter(bool(row["correct"]) for row in manifest)
    if correctness != Counter({True: 500, False: 500}):
        raise ValueError(f"unexpected correctness split: {correctness}")
    rank_sizes = [sum(case_id % args.world_size == rank for case_id in case_ids) for rank in range(args.world_size)]
    if sum(rank_sizes) != 1000:
        raise ValueError("rank partition does not cover the full manifest")

    math500 = read_jsonl(args.root / "data/MATH500/test.jsonl")
    if len(math500) != 500:
        raise ValueError(f"expected 500 MATH500 rows, found {len(math500)}")
    reference = torch.load(
        args.root / "reference/cluster_mean_vectors/cluster_mean_vectors.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(reference.shape) != (6, 3584) or not torch.isfinite(reference).all():
        raise ValueError(f"unexpected reference prototype shape: {tuple(reference.shape)}")

    model_summary = None
    if args.model is not None:
        config = AutoConfig.from_pretrained(args.model)
        model_summary = {
            "path": str(args.model),
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
        }
        if model_summary["hidden_size"] != 3584 or model_summary["num_hidden_layers"] != 28:
            raise ValueError(f"unexpected model shape: {model_summary}")

    print(
        json.dumps(
            {
                "status": "ok",
                "checksums": checksums,
                "manifest": {"rows": 1000, "correct": 500, "incorrect": 500},
                "rank_sizes": rank_sizes,
                "math500_rows": len(math500),
                "reference_shape": list(reference.shape),
                "model": model_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
