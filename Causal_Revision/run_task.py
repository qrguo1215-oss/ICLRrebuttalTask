#!/usr/bin/env python3
"""Portable resumable orchestration. No site-specific scheduler or credentials."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bridge import ROOT, SHARED
from protocol import (PRIMARY, DEFAULT_CONFIGS, MODULES, code_signature, plan, job, key,
                      result_path, checked, settings)
from spec import configurations, data, splits, read, write, digest, sha, signature, verify_assets


def assert_packages():
    from importlib.metadata import version
    for file in (SHARED/"requirements.txt", ROOT/"requirements.txt"):
        for line in file.read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                name, expected = line.strip().split("==")
                actual = version(name).split("+")[0]
                if actual != expected:
                    raise RuntimeError(f"Pinned runtime required: {name}=={expected}, found {actual}. Run without --skip-install.")


def worker(args):
    import numpy as np
    from task import configure_runtime, parent_death_guard
    from native import load_model
    from causal_engine import CausalEngine
    parent_death_guard()
    assert_packages()
    task = read(args.worker_plan)
    output = Path(task["output"])
    meta = read(output/"RUN.json")
    if meta["code_signature"] != code_signature():
        raise ValueError("Code changed after plan freeze")
    runtime = configure_runtime()
    if runtime != meta["runtime"]:
        raise ValueError("Worker GPU/runtime differs from the frozen run; use a separate output directory")
    cfg = configurations()[task["config"]]
    model, tokenizer, vector = load_model(cfg, task["model_path"])
    engine = CausalEngine(model, tokenizer, vector, cfg)
    items = data(cfg["dataset"])
    if task["kind"] == "audit":
        item = items[splits(cfg["dataset"])["calibration"][0]]
        audit = engine.audit(item)
        rows = {c: engine.run(item, c, max_tokens=256) for c in ("G", "G_off", "self_cache")}
        for c in ("G_off", "self_cache"):
            if rows[c]["token_ids"] != rows["G"]["token_ids"]:
                raise AssertionError(f"Null-control token mismatch: {c}")
        audit.update(null_token_identity=True, g_token_hash=digest(rows["G"]["token_ids"]))
        first = engine.run(item, "boundary_official", seed=0, max_tokens=96)
        engine.run(item, "V", max_tokens=96)
        repeat = engine.run(item, "boundary_official", seed=0, max_tokens=96)
        if first["token_ids"] != repeat["token_ids"]:
            raise AssertionError("Same-seed sampling differs after an intervening native run")
        audit.update(sampling_seed_repeat=True, sampling_token_hash=digest(first["token_ids"]))
        write(output/"audits"/f"{cfg['id']}_{task['device']}.json", dict(run_id=meta["run_id"], **audit))
        return
    if task["kind"] == "scan":
        from calibration import candidate_grid
        from frozen_scan import scan
        candidates = candidate_grid(read(SHARED/f"assets/calibration/{cfg['id']}.json"))
        for j in task["jobs"]:
            baseline = checked(result_path(output, job(cfg["id"], j["index"], "G")), meta)
            dest = output/"profiles"/cfg["id"]/f"{j['index']:04d}.npz"
            identity = dict(run_id=meta["run_id"], baseline_tokens=digest(baseline["token_ids"]), candidates=candidates)
            if dest.exists():
                with np.load(dest, allow_pickle=False) as old:
                    if json.loads(str(old["metadata"])) != json.loads(json.dumps(identity)):
                        raise ValueError("Stale calibration profile")
                continue
            row = scan(engine, items[j["index"]], baseline, candidates,
                       random_audit="random" in meta["modules"] and cfg["id"] == PRIMARY, smoke=meta["smoke"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.stem+f".tmp.{os.getpid()}.npz")
            np.savez_compressed(tmp, **row, metadata=json.dumps(identity))
            os.replace(tmp, dest)
            print(json.dumps(dict(scan=cfg["id"], index=j["index"], events=row["events"])), flush=True)
        return
    calpath = output/"calibration"/f"{cfg['id']}.json"
    calibration = read(calpath) if calpath.exists() else None
    if calibration and calibration["run_id"] != meta["run_id"]:
        raise ValueError("Stale calibration")
    for j in task["jobs"]:
        dest = result_path(output, j)
        if dest.exists():
            checked(dest, meta, j)
            continue
        baseline = checked(result_path(output, job(cfg["id"], j["index"], "G")), meta) if j["condition"].startswith("micro_") else None
        row = engine.run(items[j["index"]], j["condition"], j["seed"],
                         max_tokens=256 if meta["smoke"] else 10000,
                         calibration=calibration, baseline=baseline, smoke=meta["smoke"])
        row.update(j, run_id=meta["run_id"], runtime=runtime, smoke=meta["smoke"], dtype=cfg["dtype"],
                   calibration_sha256=digest(calibration) if calibration else None)
        write(dest, row)
        print(json.dumps(dict(job=key(j), status=row.get("status", "ok"), tokens=row.get("token_count"))), flush=True)


def stage(output, meta, models, cfgid, kind, tasks, gpus, workers, audit_device=None):
    tasks = [j for j in tasks if not (kind == "generate" and result_path(output, j).exists() and checked(result_path(output, j), meta, j))]
    if not tasks and kind != "audit":
        return
    devices = [audit_device] if audit_device is not None else [g for g in gpus for _ in range(workers)]
    slots = 1 if kind == "audit" else min(len(devices), len(tasks))
    bins, costs = [[] for _ in range(slots)], [0]*slots
    lengths = {}
    for j in tasks:
        if j["index"] not in lengths:
            path = result_path(output, job(cfgid, j["index"], "G"))
            lengths[j["index"]] = read(path)["token_count"] if path.exists() else len(data(configurations()[cfgid]["dataset"])[j["index"]]["problem"])
    for j in sorted(tasks, key=lambda j: -lengths[j["index"]]):
        n = min(range(slots), key=costs.__getitem__)
        bins[n].append(j)
        costs[n] += lengths[j["index"]]
    active = []
    label = f"{cfgid}_{kind}_{time.time_ns()}"
    print(f"{label}: {len(tasks)} pending, {slots} workers", flush=True)
    try:
        for n in range(slots):
            task = dict(output=str(output), config=cfgid, kind=kind, jobs=bins[n], device=devices[n],
                        model_path=models[configurations()[cfgid]["size"]])
            path = output/"plans"/f"{label}_{n}.json"
            write(path, task)
            logpath = output/"logs"/f"{label}_{n}.log"
            logpath.parent.mkdir(parents=True, exist_ok=True)
            log = logpath.open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(devices[n]), REVISION_PARENT_PID=str(os.getpid()),
                       OMP_NUM_THREADS="2", PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false")
            p = subprocess.Popen([sys.executable, str(ROOT/"run_task.py"), "_worker", "--worker-plan", str(path)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((p, log, logpath))
        next_monitor = 0
        while any(p.poll() is None for p, _, _ in active):
            for p, _, path in active:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"Worker failed: {path}\n{path.read_text()[-3500:]}")
            if time.monotonic() >= next_monitor:
                next_monitor = time.monotonic()+30
                stats = subprocess.run(["nvidia-smi", "--id="+",".join(gpus),
                                        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                row = dict(time=time.time(), stage=label, devices=stats.stdout.strip().splitlines(), monitor_error=stats.stderr.strip())
                with (output/"logs/gpu.jsonl").open("a") as f:
                    f.write(json.dumps(row)+"\n")
                print(json.dumps(row), flush=True)
            time.sleep(1)
        for p, _, path in active:
            if p.returncode != 0:
                raise RuntimeError(f"Worker failed: {path}")
    finally:
        for p, _, _ in active:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, log, _ in active:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            log.close()


def freeze_calibration(output, meta, cfgid):
    import numpy as np
    from calibration import candidate_grid
    from frozen_scan import freeze
    indices = splits(configurations()[cfgid]["dataset"])["calibration"][:1 if meta["smoke"] else 100]
    paths = [output/"profiles"/cfgid/f"{i:04d}.npz" for i in indices]
    profiles = []
    for path in paths:
        with np.load(path, allow_pickle=False) as p:
            if json.loads(str(p["metadata"]))["run_id"] != meta["run_id"]:
                raise ValueError("Mixed calibration runs")
            profiles.append({k: p[k].copy() for k in p.files if k != "metadata"})
    candidates = candidate_grid(read(SHARED/f"assets/calibration/{cfgid}.json"))
    row, cdf = freeze(profiles, candidates)
    row.update(run_id=meta["run_id"], smoke=meta["smoke"], config_id=cfgid, indices=indices,
               input_hashes={p.name: sha(p) for p in paths})
    dest = output/"calibration"/f"{cfgid}.json"
    if dest.exists() and read(dest) != row:
        raise ValueError("Frozen calibration changed; use a new run directory")
    write(dest, row)
    np.savetxt(output/"calibration"/f"{cfgid}_cdf.csv", np.stack(cdf, axis=1), delimiter=",", header="deficit,native_cdf,matched_cdf", comments="")


def import_revision(source, output, meta, jobs):
    """Only directly compatible deterministic results, never old sampling kernels."""
    from spec import path_for, checked_result
    prior = read(source/"RUN.json")
    if meta["smoke"] or prior.get("smoke") or prior.get("code_signature") != signature() or prior.get("runtime") != meta["runtime"]:
        raise ValueError("Correction output is not the same frozen generation implementation/runtime")
    imported = 0
    for j in jobs:
        if j["condition"] not in ("G", "V", "token_replay", "thought_replay"):
            continue
        dest = result_path(output, j)
        if dest.exists():
            checked(dest, meta, j)
            continue
        path = path_for(source, j["config_id"], j["index"], j["condition"], j["seed"])
        if not path.exists():
            continue
        row = checked_result(path, prior["run_id"])
        if row["index"] != j["index"] or row["condition"] != j["condition"] or row["seed"] != j["seed"] or row["runtime"] != meta["runtime"]:
            raise ValueError(f"Incompatible import: {path}")
        row.update(j, run_id=meta["run_id"], smoke=False,
                   imported_from=dict(run_id=prior["run_id"], relative_path=str(path.relative_to(source)), sha256=sha(path)))
        write(dest, row)
        imported += 1
    print(f"Imported {imported} exact deterministic results; original files unchanged.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("plan", "verify", "smoke", "run", "report", "_worker", "_runtime"))
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--modules", nargs="+", choices=MODULES, default=list(MODULES))
    ap.add_argument("--gpus", default="0", help="Comma-separated CUDA device IDs, e.g. 0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--model-root", type=Path, default=ROOT/"models")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--revision-results", type=Path)
    ap.add_argument("--worker-plan", type=Path)
    args = ap.parse_args()
    if args.command == "_worker":
        return worker(args)
    if args.command == "_runtime":
        from task import configure_runtime
        assert_packages()
        print(json.dumps(configure_runtime()))
        return
    if args.command == "report":
        if args.output is None:
            ap.error("report requires --output")
        subprocess.run([sys.executable, str(ROOT/"analysis.py"), "--output", str(args.output.resolve()),
                        "--cpu-workers", str(args.cpu_workers)], check=True)
        return
    verify_assets()
    if args.command == "verify":
        print(json.dumps(dict(assets="PASS", generation_signature=code_signature())))
        return
    smoke = args.command == "smoke"
    jobs = plan(args.configs, args.modules, smoke)
    print(json.dumps(dict(configs=args.configs, modules=args.modules, generations=len(jobs),
                          conditions=dict(Counter(j["condition"] for j in jobs)), protocol=settings()), indent=2), flush=True)
    if args.command == "plan":
        return
    if args.workers_per_gpu < 1 or args.cpu_workers < 1:
        ap.error("worker counts must be positive")
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        ap.error("--gpus requires distinct integer device IDs")
    output = (args.output or ROOT/"runs"/("smoke" if smoke else "causal")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from task import ensure_models, shutdown
    signal.signal(signal.SIGTERM, shutdown)
    models = ensure_models(args.model_root, args.configs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0], PYTHONNOUSERSITE="1")
    runtime = json.loads(subprocess.check_output([sys.executable, str(ROOT/"run_task.py"), "_runtime"], env=env, text=True))
    spec = dict(code_signature=code_signature(), shared_generation_signature=signature(), smoke=smoke,
                configs=args.configs, modules=args.modules, protocol=settings(), jobs_sha256=digest(jobs), runtime=runtime)
    meta = dict(spec, run_id=digest(spec))
    if (output/"RUN.json").exists() and read(output/"RUN.json") != meta:
        raise ValueError("Output belongs to another protocol/code/runtime; choose a NEW --output")
    write(output/"RUN.json", meta)
    write(output/"JOBS.json", jobs)
    # On every selected GPU, verify the same no-op sequence before any expensive run.
    for cfgid in args.configs:
        audits, sampling = [], []
        for gpu in gpus:
            p = output/"audits"/f"{cfgid}_{gpu}.json"
            if not p.exists():
                stage(output, meta, models, cfgid, "audit", [], gpus, 1, audit_device=gpu)
            a = read(p)
            if a["run_id"] != meta["run_id"]:
                raise ValueError("Stale preflight audit")
            audits.append(a["g_token_hash"])
            sampling.append(a["sampling_token_hash"])
        if len(set(audits)) != 1 or len(set(sampling)) != 1:
            raise ValueError("No-op or same-seed sampling tokens differ across GPUs")
    if args.revision_results:
        import_revision(args.revision_results.resolve(), output, meta, jobs)
    for cfgid in args.configs:
        selected = [j for j in jobs if j["config_id"] == cfgid]
        stage(output, meta, models, cfgid, "generate", [j for j in selected if j["condition"] in ("G", "V")], gpus, args.workers_per_gpu)
        needs = "depth" in args.modules or cfgid == PRIMARY and bool(set(args.modules) & {"random", "occlusion"})
        if needs:
            indices = splits(configurations()[cfgid]["dataset"])["calibration"][:1 if smoke else 100]
            stage(output, meta, models, cfgid, "scan", [job(cfgid, i, "scan") for i in indices], gpus, args.workers_per_gpu)
            freeze_calibration(output, meta, cfgid)
    # Stages are ordered by reviewer relevance, not by whichever yields better results.
    groups = (("depth_", "micro_", "boundary_official"),
              ("token_replay", "thought_replay", "cache_only", "self_"),
              ("reverse", "random_"), ("whole", "discover_"), ("official_full",))
    for prefixes in groups:
        for cfgid in args.configs:
            selected = [j for j in jobs if j["config_id"] == cfgid and j["condition"].startswith(prefixes)]
            stage(output, meta, models, cfgid, "generate", selected, gpus, args.workers_per_gpu)
    for j in jobs:
        checked(result_path(output, j), meta, j)
    if smoke:
        # All sham variants must exactly reproduce the same raw baseline tokens.
        for cfgid in args.configs:
            i = splits(configurations()[cfgid]["dataset"])["calibration"][0]
            baseline = checked(result_path(output, job(cfgid, i, "G")), meta)["token_ids"]
            for c in ("self_token", "self_thought", "self_cache"):
                p = result_path(output, job(cfgid, i, c))
                if p.exists() and checked(p, meta)["token_ids"] != baseline:
                    raise AssertionError(f"Sham generation differs: {cfgid}/{c}")
        write(output/"SMOKE_PASS.json", dict(run_id=meta["run_id"], jobs=len(jobs), results_are_not_scientific=True))
        print("SMOKE PASS. No test-set/full-length experiments were launched.", flush=True)
    else:
        subprocess.run([sys.executable, str(ROOT/"analysis.py"), "--output", str(output), "--cpu-workers", str(args.cpu_workers)], check=True)


if __name__ == "__main__":
    main()
