# CPS/localBest reproduction on DeepSeek-R1-Distill-Qwen-7B

This repository packages the CPS/localBest 7B reproduction as a standalone,
resumable pipeline. It uses the same model-specific 500-correct/500-incorrect
construction cases and the same MATH500 generation/evaluation protocol used by
the accompanying SEAL/ReBalance runs.

No model weights or full experiment outputs are stored in this repository.

## Frozen protocol

- Model: `DeepSeek-R1-Distill-Qwen-7B` (28 layers, hidden size 3584).
- Construction data: `data/common_500_500.jsonl`, containing exactly 500
  correct and 500 incorrect trajectories for this model.
- CPS states: `hidden_states[20]` at double-newline tokens inside `<think>`.
- Clustering: PCA dimensions `0,20,50,100,200,400`; KMeans `k=6`; seeds
  `0,1,2`; intrinsic selection by silhouette, stability ARI, balance, inertia.
- Vectors: raw means of all steps assigned to each selected cluster.
- localBest transfer: KMeans labels are permutation-invariant, so the new six
  prototypes are matched to the historical 7B prototypes by Hungarian matching
  on centered cosine similarity. The frozen historical choice `-0.6*C2` is then
  transferred to the matched new prototype. MATH500 is not searched again.
- Evaluation: SEAL Qwen2 backend, layer 20, double-newline steering, greedy
  decoding, maximum 10,000 new tokens, 500 MATH500 problems.

## Repository layout

```text
cps/                 construction, clustering, matching, inference, scoring
data/                frozen 500/500 manifest and MATH500 evaluation data
reference/           historical 7B prototypes and selection evidence
scripts/             one-GPU smoke and resumable multi-GPU pipeline
slurm/               eight-A800 sbatch template
third_party/SEAL/    minimal SEAL inference and math-grading backend
```

## Environment

Python 3.10 and CUDA-capable PyTorch are recommended. The tested environment
used PyTorch 2.3.1+cu121, Transformers 4.46.3, NumPy 2.2.6, SciPy 1.14.0, and
scikit-learn 1.7.2.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch must be installed from a cluster-specific wheel index, install it
first and then install the remaining requirements.

## Validate inputs

```bash
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
export PYTHON_BIN=/path/to/python
$PYTHON_BIN cps/validate_setup.py --model "$MODEL_PATH"
```

The validator checks all frozen SHA-256 hashes, the 500/500 split, eight-rank
coverage, MATH500 length, reference-vector shape, and model architecture.

## One-GPU smoke test

Run this before allocating a full node:

```bash
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B \
PYTHON_BIN=/path/to/python \
SMOKE_GPU=0 \
scripts/run_smoke_1gpu.sh
```

The smoke test uses one correct case, one incorrect case, one small KMeans run,
and one MATH500 example with a 64-token limit. Its cluster mapping is only a
plumbing check and is not an experimental result.

## Full eight-GPU run

```bash
export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
export PYTHON_BIN=/path/to/python
export GPU_IDS="0 1 2 3 4 5 6 7"
export WORLD_SIZE=8
scripts/run_all_8gpu.sh
```

On the HKUST(GZ) cluster, the included template can be submitted with:

```bash
sbatch --export=ALL,MODEL_PATH="$MODEL_PATH",PYTHON_BIN="$PYTHON_BIN" \
  slurm/cps7b_8gpu.sbatch
```

Override the partition or other resources using normal `sbatch` arguments when
running on a different cluster.

## Outputs and resumption

The default output root is `runs/cps7b_repro`; override it with `RUN_DIR`.

- `hidden_extract/cases/case_XXXX.pt`: case-atomic CPS states.
- `hidden_sources/`: validated correct/incorrect legacy-compatible sources.
- `cluster_sweep_layer20_k6_multiseed/`: 18 intrinsic clustering runs.
- `localbest_transfer/match_manifest.json`: auditable cluster assignment.
- `localbest_transfer/localbest_transferred.pt`: frozen final vector.
- `eval/math500_localbest/metrics.json`: final accuracy.

Extraction validates and skips completed case files. Evaluation uses one JSONL
shard per rank and skips completed problem indices. Re-running the same command
therefore resumes safely after preemption or node failure.

## Integrity

Run `sha256sum -c checksums.sha256` from the repository root. Third-party source
provenance is documented in `THIRD_PARTY.md`.
