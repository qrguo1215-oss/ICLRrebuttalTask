#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_model
require_file "$MATH500_PATH"
VECTOR="$RUN_DIR/localbest_transfer/localbest_transferred.pt"
OUT="$RUN_DIR/eval/math500_localbest"
LOG_DIR="$OUT/logs"
require_file "$VECTOR"
mkdir -p "$OUT" "$LOG_DIR"
cd "$ROOT"

pids=()
for ((rank=0; rank<WORLD_SIZE; rank++)); do
  gpu="${GPU_LIST[$((rank % ${#GPU_LIST[@]}))]}"
  output="$OUT/stream_rr${rank}.jsonl"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$CPS_DIR/eval_math_seal_stream.py" \
    --model "$MODEL_PATH" --seal-dir "$SEAL_DIR" --vector "$VECTOR" \
    --layer 20 --steer-coef 1.0 --math500-path "$MATH500_PATH" \
    --output "$output" --start "$rank" --end 500 --stride "$WORLD_SIZE" \
    --max-tokens 10000 >"$LOG_DIR/rank$(printf '%02d' "$rank").log" 2>&1 &
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
  echo "one or more MATH500 workers failed; inspect $LOG_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" "$CPS_DIR/finalize_math500_eval.py" \
  --kind seal --root "$OUT" --seal-dir "$SEAL_DIR" \
  >"$LOG_DIR/finalize.log" 2>&1
echo "CPS 7B localBest MATH500 evaluation completed"
