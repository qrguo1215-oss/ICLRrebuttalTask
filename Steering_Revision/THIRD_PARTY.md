# Sources and notices

- `vendor/seal_model.py` retains the SEAL Qwen2 wrapper used in the original experiments, derived from Hugging Face/Qwen2. Its original Apache-2.0 copyright header is retained.
- `vendor/rebalance_model.py` retains the original ReBalance confidence-based dynamic Qwen2 wrapper. The upstream MIT notice is included in `vendor/ReBalance_LICENSE`; the Qwen/Hugging Face Apache-2.0 header remains in the source.
- `vendor/evaluation/` contains the original SEAL mathematical parser/grader and its local utilities. Original source attributions in the files are retained.
- `vendor/latex2sympy2.py`, `vendor/gen/` and `vendor/PS.g4` contain the existing latex2sympy parser dependencies. Its license is in `vendor/LICENSE.txt`.
- `assets/data/` packages the frozen 500-problem MATH500 and selected GSM8K sets, preserving original item indices. Dataset ownership and upstream terms remain with the respective benchmark authors; no new license over benchmark content is asserted here.
- `assets/vectors/` contains the five previously verified SEAL/ReBalance/CPS intervention tensors. No 7B CPS tensor is supplied or synthesized.
- Model weights are NOT included. `assets/models.json` identifies DeepSeek's public distill model downloads and exact expected files; model licenses are those of the upstream models.

This task does not modify or depend on the repository's separate `CPS_7B_Reproduction` task.
