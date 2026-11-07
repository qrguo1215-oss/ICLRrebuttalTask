#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_SPEC=""
MODEL_SOURCE="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_DIR="$ROOT/models/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="$ROOT/runs/cps7b_repro"
BASE_PYTHON="${PYTHON_BIN:-python3}"
SKIP_INSTALL=0
SKIP_SMOKE=0
SMOKE_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./run.sh --gpus 0,1,2,3 [options]

Required:
  --gpus IDS          Comma- or space-separated visible GPU IDs.

Options:
  --model SOURCE      Local model directory or Hugging Face model ID.
  --model-dir PATH    Relative/local download directory for a Hub model.
  --output PATH       Output directory (default: runs/cps7b_repro).
  --python PATH       Base Python used to create the local environment.
  --skip-install      Use --python directly; do not create/install .venv.
  --skip-smoke        Skip the two-case, one-problem smoke test.
  --smoke-only        Stop after validation and the smoke test.
  -h, --help          Show this message.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --gpus) GPU_SPEC="${2:?missing value for --gpus}"; shift 2 ;;
    --model) MODEL_SOURCE="${2:?missing value for --model}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?missing value for --model-dir}"; shift 2 ;;
    --output) OUTPUT_DIR="${2:?missing value for --output}"; shift 2 ;;
    --python) BASE_PYTHON="${2:?missing value for --python}"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-only) SMOKE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU_SPEC" ]]; then
  echo "--gpus is required" >&2
  usage >&2
  exit 2
fi
cd "$ROOT"
GPU_IDS="${GPU_SPEC//,/ }"
read -r -a gpu_list <<<"$GPU_IDS"
if (( ${#gpu_list[@]} == 0 )); then
  echo "--gpus did not contain any IDs" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU ID: $gpu" >&2
    exit 2
  fi
done

if (( SKIP_INSTALL == 0 )); then
  VENV_DIR="$ROOT/.venv"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  PYTHON_BIN="$VENV_DIR/bin/python"
  "$PYTHON_BIN" -m pip install --disable-pip-version-check -r "$ROOT/requirements.txt"
else
  PYTHON_BIN="$BASE_PYTHON"
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

echo "[setup] resolving model: $MODEL_SOURCE"
MODEL_PATH="$("$PYTHON_BIN" "$ROOT/cps/prepare_model.py" --source "$MODEL_SOURCE" --local-dir "$MODEL_DIR")"
export MODEL_PATH PYTHON_BIN GPU_IDS
export WORLD_SIZE="${#gpu_list[@]}"
export RUN_DIR="$OUTPUT_DIR"

echo "[setup] validating frozen inputs and model"
"$PYTHON_BIN" "$ROOT/cps/validate_setup.py" --root "$ROOT" --model "$MODEL_PATH" --world-size "$WORLD_SIZE"

if (( SKIP_SMOKE == 0 )); then
  echo "[smoke] running two construction cases and one short MATH500 generation"
  SMOKE_RUN_DIR="$ROOT/runs/cps_debug_smoke" SMOKE_GPU="${gpu_list[0]}" "$ROOT/scripts/run_smoke.sh"
fi
if (( SMOKE_ONLY == 1 )); then
  echo "Smoke test completed; full run was not started."
  exit 0
fi

echo "[full] starting CPS 7B reproduction on ${#gpu_list[@]} GPU(s): $GPU_IDS"
"$ROOT/scripts/run_all.sh"
echo "[done] final metrics: $RUN_DIR/eval/math500_localbest/metrics.json"
