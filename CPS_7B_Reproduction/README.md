# CPS/localBest 7B reproduction

This directory is a self-contained reproduction of CPS/localBest on
`DeepSeek-R1-Distill-Qwen-7B`. A collaborator only needs CUDA GPUs, Python 3.10,
network access to Hugging Face, and one command.

## Run

Choose the GPU IDs available on the machine:

```bash
./run.sh --gpus 0,1,2,3,4,5,6,7
```

The number of workers is inferred from the list. One, two, four, eight, or
another GPU count can be used without editing any file. The pipeline is fully
resumable, so the same command can be rerun after interruption.

The command automatically:

1. creates `.venv` with access to existing system CUDA/PyTorch packages;
2. installs missing dependencies from `requirements.txt`;
3. downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` into `models/`;
4. verifies data checksums, the 500/500 split, rank coverage, vector shape, and
   model architecture;
5. runs a two-case/one-problem smoke test;
6. extracts CPS states, clusters them, transfers the frozen localBest direction,
   evaluates all 500 MATH500 problems, and grades the result.

The file to return after completion is:

```text
runs/cps7b_repro/eval/math500_localbest/metrics.json
```

Please also return these audit files:

```text
runs/cps7b_repro/localbest_transfer/match_manifest.json
runs/cps7b_repro/cluster_sweep_layer20_k6_multiseed/summary/best_run.json
runs/cps7b_repro/logs/pipeline.log
```

## Useful options

Use an existing local model instead of downloading:

```bash
./run.sh --gpus 0,1,2,3 --model /path/to/DeepSeek-R1-Distill-Qwen-7B
```

Use an already prepared Python environment and skip package installation:

```bash
./run.sh --gpus 0,1,2,3 --python /path/to/python --skip-install
```

Only validate and smoke-test the machine:

```bash
./run.sh --gpus 0 --smoke-only
```

Skip the smoke test after it has already passed:

```bash
./run.sh --gpus 0,1,2,3,4,5,6,7 --skip-smoke
```

Run `./run.sh --help` for all options.

## Frozen experimental protocol

- Construction manifest: the included model-specific 500 correct and 500
  incorrect trajectories.
- CPS representation: `hidden_states[20]` at double-newline tokens inside
  `<think>`.
- Clustering: fixed `k=6`; PCA dimensions `0,20,50,100,200,400`; seeds
  `0,1,2`; intrinsic selection across 18 runs.
- Prototype: raw mean of all layer-20 steps assigned to each cluster.
- localBest: Hungarian matching between centered new and historical 7B
  prototypes, followed by transfer of the frozen historical `-0.6*C2` choice.
  The new MATH500 test set is not searched for another coefficient.
- Evaluation: the included SEAL-compatible Qwen2 backend, layer 20,
  double-newline steering, greedy decoding, 10,000-token maximum, and the
  included 500-problem MATH500 set.

## Internal layout

```text
cps/                 extraction, clustering, matching, inference, grading
data/                immutable construction manifest and MATH500 data
reference/           historical 7B prototypes and selection evidence
scripts/             resumable internal stages called by run.sh
third_party/SEAL/    minimal pinned inference and math-grading backend
```

Input hashes are recorded in `checksums.sha256`; third-party provenance is in
`THIRD_PARTY.md`. Model files, environments, caches, and results are excluded
from Git.
