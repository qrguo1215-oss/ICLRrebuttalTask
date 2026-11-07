#!/usr/bin/env python3
"""Create a two-case CPS smoke manifest: one correct and one incorrect."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    picked = []
    for correctness in (True, False):
        row = next((item for item in rows if bool(item["correct"]) is correctness), None)
        if row is None:
            raise ValueError(f"manifest has no case with correct={correctness}")
        picked.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in picked),
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "case_idx": [row["case_idx"] for row in picked]}))


if __name__ == "__main__":
    main()
