#!/usr/bin/env python3
"""Collaborator entry point. No inference unless run/smoke is explicitly chosen."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from spec import (ROOT, VERSION, configurations, data, splits, jobs, make_job, job_key,
                  path_for, read, write, digest, sha, signature, verify_assets, checked_result)


def reject_model_overrides(path, expected):
    # Transformers prefers monolithic weights and optional tokenizer sidecars.
    suspicious = []
    for f in Path(path).iterdir():
        if not f.is_file() or f.name in expected:
            continue
        if (f.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf") or
            f.name.startswith(("adapter", "tokenizer", "added_tokens", "special_tokens", "vocab", "merges")) or
            f.name.endswith(".index.json")):
            suspicious.append(f.name)
    if suspicious:
        raise ValueError(f"unexpected model/tokenizer overrides in {path}: {sorted(suspicious)}; use a clean model directory")


def parent_death_guard():
    # Linux: terminate a worker even when the controller is killed with SIGKILL.
    import ctypes
    expected = int(os.environ["REVISION_PARENT_PID"])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install worker parent-death guard")
    if os.getppid() != expected:
        raise RuntimeError("controller disappeared before worker startup")


def shutdown(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}; stopping workers")


def configure_runtime():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    import numpy as np
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU with native BF16 support is required")
    return dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, capability=list(torch.cuda.get_device_capability()),
                gpu_name=torch.cuda.get_device_name(), attention="sdpa", tf32=False,
                model_determinism="strict", sampling_cumsum="warn-only")


def ensure_models(model_root, configs):
    from huggingface_hub import snapshot_download
    manifest = read(ROOT / "assets/models.json")
    result = {}
    for size in sorted({configurations()[k]["size"] for k in configs}):
        spec = manifest[size]
        path = Path(model_root) / spec["directory"]
        missing = any(not (path / name).is_file() for name in spec["files"])
        if missing:
            snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=str(path),
                              allow_patterns=list(spec["files"]))
        reject_model_overrides(path, spec["files"])
        print(f"Verifying model content: {spec['directory']}", flush=True)
        for name, info in spec["files"].items():
            if sha(path/name) != info["sha256"]:
                raise ValueError(f"model differs from frozen experiment weights/config: {spec['directory']}/{name}")
        result[size] = str(path.resolve())
    return result


def worker(args):
    parent_death_guard()
    import torch
    import numpy as np
    from native import load_model
    from engine import Engine
    plan = read(args.plan)
    if plan["code_signature"] != signature():
        raise ValueError("code/assets changed after planning")
    runtime = configure_runtime()
    runmeta = read(Path(plan["output"]) / "RUN.json")
    if runtime != runmeta["runtime"]:
        raise ValueError("GPU architecture or inference environment differs across workers")
    cfg = configurations()[plan["config_id"]]
    model, tokenizer, vector = load_model(cfg, plan["model_path"])
    engine = Engine(model, tokenizer, vector, cfg)
    items = data(cfg["dataset"])
    output, run_id = Path(plan["output"]), plan["run_id"]
    smoke = bool(plan.get("smoke"))
    if plan["kind"] == "smoke":
        item = items[splits(cfg["dataset"])["calibration"][0]]
        audit = engine.audit(item)
        generated = {}
        kernel = read(ROOT/f"assets/calibration/{cfg['id']}.json")["selected"]
        for c in ("G", "G_off", "self_token", "self_thought", "V", "token_replay", "thought_replay", "S", "SV", "E_all", "EV_all"):
            row = engine.generate(item, c, max_tokens=256, direct=kernel)
            row.update(smoke=True, run_id=run_id, config_id=cfg["id"])
            write(output / "smoke" / cfg["id"] / f"{c}.json.gz", row)
            generated[c] = row["token_ids"]
        for c in ("G_off", "self_token", "self_thought"):
            audit[c+"_identity"] = generated[c] == generated["G"]
        audit["shadow_invariance"] = generated["V"] == engine.generate(item, "V", max_tokens=256, record_shadow=False)["token_ids"]
        audit["seed_repeat"] = generated["S"] == engine.generate(item, "S", max_tokens=256, direct=kernel)["token_ids"]
        if cfg["method"] == "rebalance":
            model._prev_step_mean = torch.tensor([.123], device=model.device)
            model._last_maxprob = torch.tensor([.987], device=model.device)
            audit["cross_problem_reset"] = generated["V"] == engine.generate(item, "V", max_tokens=256)["token_ids"]
        if any(v is False for v in audit.values()):
            raise AssertionError(audit)
        write(output / "smoke" / cfg["id"] / "PASS.json", dict(run_id=run_id, audit=audit))
        print(json.dumps(dict(smoke=cfg["id"], passed=True)), flush=True)
        return
    if plan["kind"] == "scan":
        from calibration import scan, candidate_grid
        candidates = candidate_grid(read(ROOT/f"assets/calibration/{cfg['id']}.json"))
        for job in plan["jobs"]:
            i = job["index"]
            dest = output / "profiles" / cfg["id"] / f"{i:04d}.npz"
            baseline_path = path_for(output, cfg["id"], i, "G")
            base = checked_result(baseline_path, run_id, allow_smoke=smoke)
            meta = dict(run_id=run_id, index=i, config_id=cfg["id"], candidates=candidates,
                        baseline_token_hash=digest(base["token_ids"]), smoke=smoke)
            if dest.exists():
                saved = np.load(dest, allow_pickle=False)
                if json.loads(str(saved["metadata"])) != json.loads(json.dumps(meta)):
                    raise ValueError("stale calibration cache")
                continue
            result = scan(engine, items[i], base, candidates)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(f".tmp.{os.getpid()}.npz")
            np.savez_compressed(tmp, **result, metadata=json.dumps(meta))
            os.replace(tmp, dest)
            print(json.dumps(dict(scan=cfg["id"], index=i, events=result["events"])), flush=True)
        return
    calibration = read(output / "calibration" / f"{cfg['id']}.json") if plan.get("needs_calibration") else None
    if calibration and (calibration["run_id"] != run_id or calibration["smoke"] != smoke):
        raise ValueError("stale or smoke-only calibration")
    for job in plan["jobs"]:
        dest = path_for(output, **{k: job[k] for k in ("config_id", "index", "condition", "seed")})
        if dest.exists():
            checked_result(dest, run_id, job, allow_smoke=smoke)
            continue
        condition = job["condition"]
        stage = condition.split("_")[-1]
        threshold = calibration["thresholds"].get(stage) if calibration else None
        selected = calibration["direct"]["selected"] if calibration else None
        started = time.monotonic()
        row = engine.generate(items[job["index"]], condition, seed=job["seed"],
                              max_tokens=256 if smoke else 10000, threshold=threshold, direct=selected)
        row.update(job, run_id=run_id, smoke=smoke, runtime=runtime, dtype=cfg["dtype"],
                   calibration_sha256=digest(calibration) if calibration else None,
                   wall_seconds=time.monotonic()-started)
        write(dest, row)
        print(json.dumps(dict(config=cfg["id"], index=job["index"], condition=condition,
                              seed=job["seed"], tokens=row["token_count"])), flush=True)


def run_stage(output, runmeta, models, cfgid, kind, tasks, gpus, workers_per_gpu, smoke=False):
    unique = {job_key(j): j for j in tasks}
    pending = []
    for j in unique.values():
        if kind == "generate":
            dest = path_for(output, j["config_id"], j["index"], j["condition"], j["seed"])
            if dest.exists():
                checked_result(dest, runmeta["run_id"], j, allow_smoke=smoke)
                continue
        pending.append(j)
    if kind == "generate" and not pending:
        return
    if kind == "smoke" and (output/"smoke"/cfgid/"PASS.json").exists():
        if read(output/"smoke"/cfgid/"PASS.json")["run_id"] != runmeta["run_id"]:
            raise ValueError("stale smoke checkpoint")
        return
    cfg = configurations()[cfgid]
    devices = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    slots = min(len(devices), len(pending)) if pending else 1
    bins, costs = [[] for _ in range(slots)], [0] * slots
    # The already measured G length balances expensive tails without reading correctness.
    estimates = {}
    questions = data(cfg["dataset"])
    for j in pending:
        if j["index"] in estimates:
            continue
        base = path_for(output, cfgid, j["index"], "G")
        estimates[j["index"]] = read(base)["token_count"] if base.exists() else max(100, len(questions[j["index"]]["problem"]))
    for job in sorted(pending, key=lambda j: -estimates[j["index"]]):
        slot = min(range(slots), key=costs.__getitem__)
        bins[slot].append(job)
        costs[slot] += estimates[job["index"]]
    processes = []
    print(f"Stage {cfgid}/{kind}: {len(pending)} pending items, {slots} workers", flush=True)
    try:
        for n in range(slots):
            plan = dict(code_signature=runmeta["code_signature"], run_id=runmeta["run_id"], output=str(output),
                        config_id=cfgid, model_path=models[cfg["size"]], kind=kind, jobs=bins[n], smoke=smoke,
                        needs_calibration=any(j["condition"] not in ("G", "V", "token_replay", "thought_replay") for j in bins[n]))
            label = f"{cfgid}_{kind}_{n}"
            plan_path, log_path = output / "plans" / f"{label}.json", output / "logs" / f"{label}.log"
            write(plan_path, plan)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[n], PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2",
                       REVISION_PARENT_PID=str(os.getpid()))
            process = subprocess.Popen([sys.executable, str(ROOT/"task.py"), "_worker", "--plan", str(plan_path)],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, handle, log_path))
        next_monitor = 0.
        while any(p.poll() is None for p, _, _ in processes):
            for p, _, log in processes:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"worker failed; inspect {log}; completed items are retained")
            if time.monotonic() >= next_monitor:
                next_monitor = time.monotonic() + 30
                try:
                    stats = subprocess.run(["nvidia-smi", "--id="+",".join(gpus),
                                            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    if stats.returncode == 0:
                        line = json.dumps(dict(time=time.time(), stage=f"{cfgid}/{kind}", gpu=stats.stdout.strip().splitlines()))
                        with (output/"logs/gpu.jsonl").open("a") as f:
                            f.write(line+"\n")
                        print(line, flush=True)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            time.sleep(1)
        if any(p.returncode != 0 for p, _, _ in processes):
            raise RuntimeError("worker failed; see logs")
    finally:
        for p, handle, _ in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            handle.close()


def freeze_calibration(output, meta, cfgid, indices, smoke=False):
    import numpy as np
    from calibration import finish, candidate_grid
    cfg = configurations()[cfgid]
    if not smoke and set(indices) != set(splits(cfg["dataset"])["calibration"]):
        raise ValueError("calibration100 membership mismatch")
    reference = read(ROOT / f"assets/calibration/{cfgid}.json")
    candidates = candidate_grid(reference)
    profiles, margins, sources = [], [], {}
    for i in indices:
        path = output/"profiles"/cfgid/f"{i:04d}.npz"
        values = np.load(path, allow_pickle=False)
        md = json.loads(str(values["metadata"]))
        base = checked_result(path_for(output, cfgid, i, "G"), meta["run_id"], allow_smoke=smoke)
        if md != json.loads(json.dumps(dict(run_id=meta["run_id"], index=i, config_id=cfgid, candidates=candidates,
                                           baseline_token_hash=digest(base["token_ids"]), smoke=smoke))):
            raise ValueError("profile provenance mismatch")
        profiles.append({k: values[k] for k in ("grid", "mass", "native_mass", "events", "native_flip", "sample_flip")})
        margins.extend(e["margin"] for e in base["events"])
        sources[str(i)] = dict(profile_sha256=sha(path), baseline_tokens=md["baseline_token_hash"])
    fitted, cdf = finish(profiles, candidates)
    thresholds = {f"q{int(q*100)}": float(np.quantile(margins, q)) for q in (.1, .25, .5)}
    record = dict(run_id=meta["run_id"], config_id=cfgid, smoke=smoke, indices=sorted(indices), sources=sources,
                  direct=fitted, thresholds=thresholds, threshold_weighting="event-pooled baseline margin",
                  historical_selected=reference["selected"], selection_uses_correctness=False)
    write(output/"calibration"/f"{cfgid}.json", record)
    with (output/"calibration"/f"{cfgid}_cdf.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["delta", "F_native", "F_sampling"])
        writer.writerows(zip(*cdf))


def plan_jobs(config_ids, scopes):
    result = {}
    for key in config_ids:
        for stage in ("baseline", "native", "replay", "direct", "whole"):
            if stage == "replay" and "replay" not in scopes:
                continue
            if stage == "direct" and "direct" not in scopes:
                continue
            if stage == "whole" and not ("whole" in scopes or "gsm-whole" in scopes and key.startswith("gsm")):
                continue
            for j in jobs(key, stage):
                result[job_key(j)] = j
    return list(result.values())


def orchestrate(args):
    verify_assets()
    configs = args.configs.split(",") if args.configs else list(configurations())
    if not set(configs) <= set(configurations()):
        raise ValueError("unknown config; 7B CPS is intentionally absent")
    scopes = args.scopes.split(",")
    if not set(scopes) <= {"replay", "direct", "gsm-whole", "whole"}:
        raise ValueError("invalid scopes")
    work = plan_jobs(configs, scopes)
    if args.command == "plan":
        from collections import Counter
        print(json.dumps(dict(generations=len(work), frozen_forward_cases=100*len(configs),
                              by_condition=dict(Counter(j["condition"] for j in work)), scopes=scopes), indent=2))
        return
    if not args.gpus or args.workers_per_gpu < 1:
        raise ValueError("specify --gpus and positive --workers-per-gpu")
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)) or any(not g for g in gpus):
        raise ValueError("duplicate/empty GPU identifiers")
    output = Path(args.output or ("runs/smoke" if args.command == "smoke" else "runs/revision")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    models = ensure_models(args.model_root, configs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], PYTHONNOUSERSITE="1")
    runtime = json.loads(subprocess.check_output([sys.executable, str(ROOT/"task.py"), "_runtime"], env=env))
    meta = dict(version=VERSION, code_signature=signature(), runtime=runtime,
                model_manifest_sha256=sha(ROOT/"assets/models.json"), scopes=scopes, configs=configs,
                max_new_tokens=10000, seeds=list(range(8)), batch_size=1, smoke=args.command == "smoke")
    meta["run_id"] = digest(meta)
    if (output/"RUN.json").exists() and read(output/"RUN.json") != meta:
        raise ValueError("output belongs to another protocol/runtime; use a new output directory")
    write(output/"RUN.json", meta)
    write(output/"EXPECTED.json", work)
    # Smoke tests only see calibration examples, never held-out replay/test cases.
    for key in configs:
        run_stage(output, meta, models, key, "smoke", [], gpus, 1, True)
    if args.command == "smoke":
        # Exercise calibration and end-to-end sampling with ONE calibration case/config.
        for key in configs:
            i = splits(configurations()[key]["dataset"])["calibration"][0]
            g = make_job(key, i, "G")
            run_stage(output, meta, models, g["config_id"], "generate", [g], gpus, 1, True)
            run_stage(output, meta, models, key, "scan", [make_job(key, i, "scan")], gpus, 1, True)
            freeze_calibration(output, meta, key, [i], True)
            run_stage(output, meta, models, key, "generate", [make_job(key, i, c) for c in ("S", "SV", "E_q25", "EV_q25")], gpus, 1, True)
        print("Smoke complete; no full experiment started.")
        return
    for key in sorted({j["config_id"] for j in work if j["condition"] == "G"}):
        run_stage(output, meta, models, key, "generate", [j for j in work if j["config_id"] == key and j["condition"] == "G"], gpus, args.workers_per_gpu)
    # All methods are rescanned on the exact same 100 prefixes and original grid.
    # This repairs state leaks AND numeric calibration/runtime kernel mismatch.
    for key in configs:
        ids = splits(configurations()[key]["dataset"])["calibration"]
        run_stage(output, meta, models, key, "scan", [make_job(key, i, "scan") for i in ids], gpus, args.workers_per_gpu)
        freeze_calibration(output, meta, key, ids)
    # Finish the high-priority replay/direct repairs before whole-thought work.
    for stage_conditions in ({"V", "token_replay", "thought_replay"}, {"S", "SV"},
                             {f"{c}_{s}" for c in ("E", "EV") for s in ("q10", "q25", "q50", "all")}):
        for key in configs:
            subset = [j for j in work if j["config_id"] == key and j["condition"] in stage_conditions]
            if subset:
                run_stage(output, meta, models, key, "generate", subset, gpus, args.workers_per_gpu)
    subprocess.run([sys.executable, str(ROOT/"report.py"), "--output", str(output), "--cpu-workers", str(args.cpu_workers)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke", "plan"):
        a = sub.add_parser(name)
        a.add_argument("--gpus")
        a.add_argument("--workers-per-gpu", type=int, default=1)
        a.add_argument("--cpu-workers", type=int, default=4)
        a.add_argument("--model-root", default=str(ROOT/"models"))
        a.add_argument("--output")
        a.add_argument("--configs")
        a.add_argument("--scopes", default="replay,direct,gsm-whole")
        a.set_defaults(func=orchestrate)
    a = sub.add_parser("_worker")
    a.add_argument("--plan", required=True)
    a.set_defaults(func=worker)
    sub.add_parser("_runtime").set_defaults(func=lambda _: print(json.dumps(configure_runtime())))
    sub.add_parser("verify").set_defaults(func=lambda _: print(json.dumps(dict(assets=len(verify_assets()), signature=signature()))))
    args = p.parse_args()
    if args.command in ("run", "smoke"):
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGHUP, shutdown)
    args.func(args)


if __name__ == "__main__":
    main()
