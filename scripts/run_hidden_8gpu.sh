#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_model
require_file "$MANIFEST_PATH"
HIDDEN_DIR="$RUN_DIR/hidden_extract"
LOG_DIR="$RUN_DIR/logs/hidden"
mkdir -p "$HIDDEN_DIR" "$LOG_DIR"
cd "$ROOT"

pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  gpu="${GPU_LIST[$((rank % ${#GPU_LIST[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$CPS_DIR/extract_cps_hidden.py" \
    --model "$MODEL_PATH" --manifest "$MANIFEST_PATH" \
    --output-dir "$HIDDEN_DIR" --rank "$rank" --world-size "$WORLD_SIZE" \
    --layer 20 --dtype bfloat16 --attn-implementation sdpa \
    >"$LOG_DIR/rank$(printf '%02d' "$rank").log" 2>&1 &
  pids+=("$!")
  if (( LAUNCH_STAGGER_SECONDS > 0 && rank + 1 < WORLD_SIZE )); then
    sleep "$LAUNCH_STAGGER_SECONDS"
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "one or more hidden extraction workers failed; inspect $LOG_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" "$CPS_DIR/finalize_cps_hidden.py" \
  --manifest "$MANIFEST_PATH" --input-dir "$HIDDEN_DIR" \
  --output-dir "$RUN_DIR/hidden_sources" --layer 20 --hidden-size 3584 \
  >"$RUN_DIR/logs/finalize_hidden.log" 2>&1

echo "CPS 7B hidden extraction and finalization completed"
