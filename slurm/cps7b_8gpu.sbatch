#!/usr/bin/env bash
#SBATCH --job-name=cps7b-repro
#SBATCH --partition=i64m1tga800u8ka
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:a800:8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_PATH:?export MODEL_PATH before sbatch, or pass it with --export}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export WORLD_SIZE="${WORLD_SIZE:-8}"
cd "$ROOT"
exec scripts/run_all_8gpu.sh
