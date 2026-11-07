#!/usr/bin/env python3
"""Merge and score streamed ReBalance or completed SEAL MATH500 shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rebalance(root: Path) -> list[dict]:
    merged: dict[int, dict] = {}
    for path in sorted(root.glob("shard*.jsonl")):
        for row in read_jsonl(path):
            merged.setdefault(int(row["idx"]), row)
    return [merged[idx] for idx in sorted(merged)]


def load_seal(root: Path) -> list[dict]:
    merged: dict[int, dict] = {}
    for path in sorted(root.rglob("math_eval.jsonl")):
        try:
            start_text, end_text = path.parent.name.split("_", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except ValueError:
            continue
        rows = read_jsonl(path)
        if len(rows) != end - start:
            raise RuntimeError(f"{path}: expected {end - start} rows, found {len(rows)}")
        for offset, row in enumerate(rows):
            idx = start + offset
            row["idx"] = idx
            if idx in merged:
                raise RuntimeError(f"duplicate MATH500 index {idx}: {path}")
            merged[idx] = row
    for path in sorted(root.glob("stream_*.jsonl")):
        for row in read_jsonl(path):
            idx = int(row["idx"])
            if idx in merged:
                raise RuntimeError(f"duplicate MATH500 index {idx}: {path}")
            merged[idx] = row
    return [merged[idx] for idx in sorted(merged)]


def grade(rows: list[dict], seal_dir: Path) -> None:
    sys.path.insert(0, str(seal_dir.resolve()))
    from eval_math_rule.evaluation.parser import extract_answer, strip_string
    from get_math_results import math_equal_with_timeout

    for pos, row in enumerate(rows, start=1):
        gt = strip_string(extract_answer(row["answer"], data_name="omni-math"), skip_unit=False)
        pred = strip_string(
            extract_answer(row["model_generation"][0], data_name="omni-math"),
            skip_unit=False,
        )
        correct = bool(math_equal_with_timeout(pred, gt, timeout=5))
        row.update(
            all_pred=[pred],
            all_eval=[correct],
            mv_pred=pred,
            mv_eval=correct,
            mv_index=0,
        )
        if pos % 50 == 0:
            print(f"graded={pos}/500 correct={sum(bool(x['mv_eval']) for x in rows[:pos])}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("seal", "rebalance"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--regrade", action="store_true")
    args = parser.parse_args()

    rows = load_seal(args.root) if args.kind == "seal" else load_rebalance(args.root)
    indices = [int(row["idx"]) for row in rows]
    if indices != list(range(500)):
        missing = sorted(set(range(500)) - set(indices))
        raise SystemExit(f"expected indices 0..499 exactly; rows={len(rows)} missing={missing[:20]}")

    already_graded = all("mv_eval" in row for row in rows)
    if args.regrade or not already_graded:
        grade(rows, args.seal_dir)

    predictions = []
    for row in rows:
        predictions.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"all_pred", "all_eval", "mv_pred", "mv_eval", "mv_index"}
            }
        )
    write_jsonl(args.root / "predictions.jsonl", predictions)
    write_jsonl(args.root / "math_eval.jsonl", rows)
    correct = sum(bool(row["mv_eval"]) for row in rows)
    metrics = {"acc": correct / len(rows), "correct": correct, "total": len(rows)}
    with (args.root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
