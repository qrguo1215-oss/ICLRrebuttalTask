#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SOURCE_DIR="$RUN_DIR/hidden_sources"
SWEEP_DIR="$RUN_DIR/cluster_sweep_layer20_k6_multiseed"
SUMMARY_DIR="$SWEEP_DIR/summary"
LOG_DIR="$RUN_DIR/logs"
CORRECT_HIDDEN="$SOURCE_DIR/correct/hidden.pt"
INCORRECT_HIDDEN="$SOURCE_DIR/incorrect/hidden.pt"
CORRECT_PROMPTS="$SOURCE_DIR/correct/prompts.json"
INCORRECT_PROMPTS="$SOURCE_DIR/incorrect/prompts.json"

mkdir -p "$LOG_DIR"
for path in "$CORRECT_HIDDEN" "$INCORRECT_HIDDEN" "$CORRECT_PROMPTS" "$INCORRECT_PROMPTS" "$LEGACY_REFERENCE"; do
  require_file "$path"
done

cd "$CPS_DIR"
if [[ ! -f "$SUMMARY_DIR/best_run.json" ]]; then
  "$PYTHON_BIN" cluster_fixed_k_multi_seed.py \
    --hidden_paths "$CORRECT_HIDDEN" "$INCORRECT_HIDDEN" \
    --source_names correct incorrect \
    --prompts_paths "$CORRECT_PROMPTS" "$INCORRECT_PROMPTS" \
    --output_root "$SWEEP_DIR" --layer 20 --k 6 \
    --pca_dims 0 20 50 100 200 400 --seeds 0 1 2 \
    --max_steps_per_example 30 --n_init 50 --max_iter 1000 \
    >"$LOG_DIR/cluster_fixed_k6_multiseed.log" 2>&1

  mapfile -t run_dirs < <(find "$SWEEP_DIR" -mindepth 2 -maxdepth 2 -type d -name 'seed*' | sort)
  if (( ${#run_dirs[@]} != 18 )); then
    echo "expected 18 completed clustering runs, found ${#run_dirs[@]}" >&2
    exit 1
  fi
  "$PYTHON_BIN" select_fixed_k6_seed_run.py \
    --run_dirs "${run_dirs[@]}" --output_dir "$SUMMARY_DIR" --k 6 \
    >"$LOG_DIR/select_fixed_k6_seed_run.log" 2>&1
fi

best_run_dir="$($PYTHON_BIN -c "import json; print(json.load(open(r'$SUMMARY_DIR/best_run.json'))['best_run_dir'])")"
vector_dir="$best_run_dir/cluster_mean_vectors"
signed_dir="$best_run_dir/signed_cluster_vectors"

if [[ ! -f "$vector_dir/cluster_mean_vectors.pt" ]]; then
  "$PYTHON_BIN" generate_cluster_steering_vectors_multi.py \
    --hidden_paths "$CORRECT_HIDDEN" "$INCORRECT_HIDDEN" \
    --source_names correct incorrect \
    --assignments_path "$best_run_dir/assignments_full.jsonl" \
    --output_dir "$vector_dir" --layer 20 --source_cluster_dir "$best_run_dir" \
    >"$LOG_DIR/generate_cluster_vectors.log" 2>&1
fi

if [[ ! -f "$signed_dir/manifest.json" ]]; then
  "$PYTHON_BIN" make_signed_cluster_vectors.py \
    --vector_dir "$vector_dir" --output_dir "$signed_dir" --num_clusters 6 \
    >"$LOG_DIR/make_signed_cluster_vectors.log" 2>&1
fi

localbest_dir="$RUN_DIR/localbest_transfer"
if [[ ! -f "$localbest_dir/match_manifest.json" ]]; then
  "$PYTHON_BIN" "$CPS_DIR/match_legacy_clusters.py" \
    --legacy-vectors "$LEGACY_REFERENCE" \
    --new-vectors "$vector_dir/cluster_mean_vectors.pt" \
    --output-dir "$localbest_dir" \
    --legacy-localbest-cluster 2 --legacy-localbest-coef -0.6 \
    >"$LOG_DIR/match_legacy_clusters.log" 2>&1
fi

printf 'CPS clustering and vector generation completed\nbest_run_dir=%s\nsigned_vector_dir=%s\nlocalbest_vector=%s\n' \
  "$best_run_dir" "$signed_dir" "$localbest_dir/localbest_transferred.pt"
