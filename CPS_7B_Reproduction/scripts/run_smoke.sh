#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_model
RUN_DIR="${SMOKE_RUN_DIR:-$ROOT/runs/cps_debug_smoke}"
MANIFEST="$RUN_DIR/smoke_1correct_1incorrect.jsonl"
LOG_DIR="$RUN_DIR/logs"
SMOKE_GPU="${SMOKE_GPU:-${GPU_LIST[0]}}"
mkdir -p "$LOG_DIR"
cd "$ROOT"

"$PYTHON_BIN" "$CPS_DIR/make_smoke_manifest.py" --manifest "$MANIFEST_PATH" --output "$MANIFEST"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_BIN" "$CPS_DIR/extract_cps_hidden.py" \
  --model "$MODEL_PATH" --manifest "$MANIFEST" --output-dir "$RUN_DIR/hidden_extract" \
  --rank 0 --world-size 1 --layer 20 --dtype bfloat16 --attn-implementation sdpa \
  >"$LOG_DIR/extract.log" 2>&1
"$PYTHON_BIN" "$CPS_DIR/finalize_cps_hidden.py" \
  --manifest "$MANIFEST" --input-dir "$RUN_DIR/hidden_extract" \
  --output-dir "$RUN_DIR/hidden_sources" --layer 20 --hidden-size 3584 \
  --expected-total 2 --expected-per-group 1 >"$LOG_DIR/finalize.log" 2>&1

SWEEP_DIR="$RUN_DIR/cluster_sweep"
cd "$CPS_DIR"
"$PYTHON_BIN" cluster_fixed_k_multi_seed.py \
  --hidden_paths "$RUN_DIR/hidden_sources/correct/hidden.pt" "$RUN_DIR/hidden_sources/incorrect/hidden.pt" \
  --source_names correct incorrect \
  --prompts_paths "$RUN_DIR/hidden_sources/correct/prompts.json" "$RUN_DIR/hidden_sources/incorrect/prompts.json" \
  --output_root "$SWEEP_DIR" --layer 20 --k 6 --pca_dims 20 --seeds 0 \
  --max_steps_per_example 30 --n_init 2 --max_iter 100 >"$LOG_DIR/cluster.log" 2>&1
"$PYTHON_BIN" select_fixed_k6_seed_run.py \
  --run_dirs "$SWEEP_DIR/pca20/seed0" --output_dir "$SWEEP_DIR/summary" --k 6 \
  >"$LOG_DIR/select.log" 2>&1

BEST_RUN="$SWEEP_DIR/pca20/seed0"
"$PYTHON_BIN" generate_cluster_steering_vectors_multi.py \
  --hidden_paths "$RUN_DIR/hidden_sources/correct/hidden.pt" "$RUN_DIR/hidden_sources/incorrect/hidden.pt" \
  --source_names correct incorrect --assignments_path "$BEST_RUN/assignments_full.jsonl" \
  --output_dir "$BEST_RUN/cluster_mean_vectors" --layer 20 --source_cluster_dir "$BEST_RUN" \
  >"$LOG_DIR/vectors.log" 2>&1
"$PYTHON_BIN" "$CPS_DIR/match_legacy_clusters.py" \
  --legacy-vectors "$LEGACY_REFERENCE" \
  --new-vectors "$BEST_RUN/cluster_mean_vectors/cluster_mean_vectors.pt" \
  --output-dir "$RUN_DIR/localbest_transfer" >"$LOG_DIR/match.log" 2>&1

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_BIN" "$CPS_DIR/eval_math_seal_stream.py" \
  --model "$MODEL_PATH" --seal-dir "$SEAL_DIR" \
  --vector "$RUN_DIR/localbest_transfer/localbest_transferred.pt" \
  --layer 20 --steer-coef 1.0 --math500-path "$MATH500_PATH" \
  --output "$RUN_DIR/eval/stream_smoke.jsonl" \
  --start 0 --end 1 --stride 1 --max-tokens 64 >"$LOG_DIR/eval_one.log" 2>&1

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "hidden_sources/summary.json").read_text())
match = json.loads((root / "localbest_transfer/match_manifest.json").read_text())
rows = [json.loads(line) for line in (root / "eval/stream_smoke.jsonl").read_text().splitlines() if line]
assert summary["groups"]["correct"]["cases"] == 1
assert summary["groups"]["incorrect"]["cases"] == 1
assert len(match["assignments"]) == 6
assert len(rows) == 1 and rows[0]["idx"] == 0
print(json.dumps({"status": "ok", "hidden": summary["groups"], "localbest": match["transferred_localbest"], "eval_rows": len(rows)}, indent=2))
PY
