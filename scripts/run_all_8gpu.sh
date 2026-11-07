#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_model
mkdir -p "$RUN_DIR/logs"
cd "$ROOT"
printf '[%s] host=%s slurm_job_id=%s gpu_ids=%s run_dir=%s\n' \
  "$(date -Is)" "$(hostname)" "${SLURM_JOB_ID:-none}" "$GPU_IDS_STRING" "$RUN_DIR" \
  | tee -a "$RUN_DIR/logs/pipeline.log"

printf '[%s] hidden extraction start\n' "$(date -Is)" | tee -a "$RUN_DIR/logs/pipeline.log"
"$SCRIPT_DIR/run_hidden_8gpu.sh"
printf '[%s] clustering/vector generation start\n' "$(date -Is)" | tee -a "$RUN_DIR/logs/pipeline.log"
"$SCRIPT_DIR/run_cluster_and_vectors.sh" | tee -a "$RUN_DIR/logs/pipeline.log"
printf '[%s] MATH500 evaluation start\n' "$(date -Is)" | tee -a "$RUN_DIR/logs/pipeline.log"
"$SCRIPT_DIR/run_math500_8gpu.sh"
printf '[%s] pipeline complete\n' "$(date -Is)" | tee -a "$RUN_DIR/logs/pipeline.log"
