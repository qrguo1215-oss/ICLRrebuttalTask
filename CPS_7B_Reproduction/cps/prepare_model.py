#!/usr/bin/env python3
"""Resolve a local model directory or download a Hugging Face model once."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = Path(args.source).expanduser()
    if source_path.is_dir():
        resolved = source_path.resolve()
    else:
        args.local_dir.mkdir(parents=True, exist_ok=True)
        resolved = Path(
            snapshot_download(
                repo_id=args.source,
                local_dir=args.local_dir,
                ignore_patterns=("*.msgpack", "*.h5", "*.ot"),
            )
        ).resolve()

    config = AutoConfig.from_pretrained(resolved, local_files_only=True)
    if int(config.hidden_size) != 3584 or int(config.num_hidden_layers) != 28:
        raise ValueError(
            f"expected the 7B Qwen distill model (hidden=3584, layers=28), got "
            f"hidden={config.hidden_size}, layers={config.num_hidden_layers}"
        )
    print(resolved)


if __name__ == "__main__":
    main()
