#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
MANIFEST_PATH="${MANIFEST_PATH:-$ROOT/data/common_500_500.jsonl}"
MATH500_PATH="${MATH500_PATH:-$ROOT/data/MATH500/test.jsonl}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/cps7b_repro}"
CPS_DIR="$ROOT/cps"
SEAL_DIR="$ROOT/third_party/SEAL"
LEGACY_REFERENCE="${LEGACY_REFERENCE:-$ROOT/reference/cluster_mean_vectors/cluster_mean_vectors.pt}"

GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<<"$GPU_IDS_STRING"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS resolved to an empty list" >&2
  exit 1
fi
WORLD_SIZE="${WORLD_SIZE:-${#GPU_LIST[@]}}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-5}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

require_model() {
  if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
    echo "Set MODEL_PATH to the local DeepSeek-R1-Distill-Qwen-7B directory." >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}
