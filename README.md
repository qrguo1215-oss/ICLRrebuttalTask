# ICLR rebuttal experiments

Each experiment is isolated in its own directory. The CPS/localBest 7B
reproduction is in [`CPS_7B_Reproduction/`](CPS_7B_Reproduction/).

## Current Table 2 Correction Only

For the fixed-200-example baseline/replay correction, use the standalone
[`Table2_Revision/`](Table2_Revision/), **not** the broad Steering_Revision run.
It bundles 2,400 reusable records and schedules only 2,000 missing generations,
including the recovered CPS 7B vector. No sampling calibration or GSM reruns.

```bash
cd Table2_Revision
bash run.sh plan
bash run.sh run --gpus 0,1,2,3 --workers-per-gpu 1
```

Return `Table2_Revision/runs/table2_results.tar.gz`. Read its README for the
frozen protocol, historical-source caveats, automatic safety gates and resume.

The paper result corrections (Tables 2/6/7/8 and GSM Figure 6) are a separate
task in [`Steering_Revision/`](Steering_Revision/). For that task, run:

```bash
cd Steering_Revision
./run.sh --gpus 0,1,2,3
```

It downloads the frozen models, checks all inputs, runs preflight tests,
resumes the corrected experiments and creates one results archive to return.
GPU numbers and concurrency are configurable; no cluster-specific setup is needed.

## Causal Rebuttal Task

The reviewer-facing causal controls are in [`Causal_Revision/`](Causal_Revision/),
separate from the table/data corrections. Clone the **whole** repository; this
task reuses the audited native adapters and frozen assets from `Steering_Revision`.

```bash
cd Causal_Revision
bash run.sh plan
bash run.sh smoke --gpus 0,1
bash run.sh run --gpus 0,1 --workers-per-gpu 1
```

The default plan includes depth controls, text/cache replay, fixed random
directions, independently seeded item-level occlusion, and blind counterpart
materials. The README explains smaller priority modules and exact reuse of
compatible correction results. Full runs are expensive: inspect `plan` first.
Models download automatically; send back the generated results archive.
Strategy judgments require returned external labels and are never fabricated.

## CPS 7B Construction Task

To run it, clone this repository and execute:

```bash
cd CPS_7B_Reproduction
./run.sh --gpus 0,1,2,3,4,5,6,7
```

The command installs missing Python dependencies in a local environment,
downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, validates all frozen
inputs, runs a smoke test, resumes the full construction pipeline, and writes
the final MATH500 result under `runs/cps7b_repro/`.
