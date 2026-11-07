# ICLR rebuttal experiments

Each experiment is isolated in its own directory. The CPS/localBest 7B
reproduction is in [`CPS_7B_Reproduction/`](CPS_7B_Reproduction/).

The paper result corrections (Tables 2/6/7/8 and GSM Figure 6) are a separate
task in [`Steering_Revision/`](Steering_Revision/). For that task, run:

```bash
cd Steering_Revision
./run.sh --gpus 0,1,2,3
```

It downloads the frozen models, checks all inputs, runs preflight tests,
resumes the corrected experiments and creates one results archive to return.
GPU numbers and concurrency are configurable; no cluster-specific setup is needed.

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
