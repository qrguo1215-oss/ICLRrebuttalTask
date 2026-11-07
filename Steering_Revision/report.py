"""CPU-only grading, complete paired reports and collaborator return bundle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
from spec import ROOT, configurations, data, splits, read, write, path_for, checked_result, digest, sha, signature, generation_seed

sys.path.insert(0, str(ROOT / "vendor"))


def evaluator_identity():
    files = {str(f.relative_to(ROOT)):sha(f) for f in list((ROOT/"vendor").rglob("*.py"))+[Path(__file__)]}
    packages = {name:version(name) for name in ("sympy", "antlr4-python3-runtime", "regex", "word2number", "numpy")}
    return dict(files=files, packages=packages, python=sys.version)


def compare_child(connection, text, response, gold):
    try:
        from evaluation.parser import extract_answer, strip_string
        from evaluation.grader import math_equal
        old = strip_string(extract_answer(text, data_name="omni-math"), skip_unit=False)
        full = strip_string(extract_answer(response, data_name="omni-math"), skip_unit=False)
        target = strip_string(str(gold).strip(), skip_unit=False)
        connection.send(dict(status="ok", prediction=old, strict_prediction=full,
                             correct=bool(math_equal(old, target)),
                             strict_correct=bool(math_equal(full, target, include_percentage=False))))
    except Exception as exc:
        connection.send(dict(status="error", error=type(exc).__name__ + ": " + str(exc)))
    finally:
        connection.close()


def evaluate(row, gold, timeout=8):
    # Only used in the separate CPU command, never a CUDA-initialized process.
    ctx = mp.get_context("fork")
    rx, tx = ctx.Pipe(duplex=False)
    p = ctx.Process(target=compare_child, args=(tx, row["text"], row["response_text"], gold))
    p.start()
    tx.close()
    try:
        if rx.poll(timeout):
            try:
                result = rx.recv()
            except EOFError:
                result = dict(status="error", error="evaluator exited without result")
        else:
            result = dict(status="timeout")
    finally:
        rx.close()
        if p.is_alive():
            p.terminate()
        p.join()
    return result


def grade_job(payload):
    source, destination, run_id, gold, evaluator_id = payload
    source_hash = sha(source)
    if Path(destination).exists():
        cached = read(destination)
        if cached["source_sha256"] == source_hash and cached["evaluator_id"] == evaluator_id and cached["status"] == "ok":
            return cached
    row = checked_result(source, run_id)
    result = evaluate(row, gold)
    result.update(index=row["index"], condition=row["condition"], seed=row["seed"], config_id=row["config_id"],
                  run_id=run_id, evaluator_id=evaluator_id, source_sha256=source_hash,
                  token_count=row["token_count"], hit_max=row["hit_max"], closed=row["closed"])
    write(destination, result)
    return result


def paired_interval(values):
    rng = np.random.default_rng(20261108)
    values = np.asarray(values, dtype=float)
    boot = np.array([values[rng.integers(len(values), size=len(values))].mean() for _ in range(10000)]) * 100
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(estimate_pp=float(values.mean()*100), ci_low_pp=float(lo), ci_high_pp=float(hi),
                negative_point=bool(values.mean() < 0), ci_strictly_negative=bool(hi < 0))


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_reports(output, meta, graded):
    tables = {k: [] for k in ("table2", "table6", "table7", "table8", "figure6")}
    details = {}
    for cfgid in meta["configs"]:
        cfg = configurations()[cfgid]
        split = splits(cfg["dataset"])
        conditions = ["G", "V"]
        if "direct" in meta["scopes"]:
            conditions += ["S", "SV"]
        whole = "whole" in meta["scopes"] or "gsm-whole" in meta["scopes"] and cfg["dataset"] == "gsm"
        if whole:
            conditions += [f"{c}_{s}" for c in ("E", "EV") for s in ("q10", "q25", "q50", "all")]

        def condition_rows(condition, indices):
            return [r for i in indices for r in graded[str(path_for(output, cfgid, i, condition, 0).parent)].get(i, [])]

        def stats(condition, indices):
            rows = condition_rows(condition, indices)
            seeds = sorted({r["seed"] for r in rows})
            if len(rows) != len(indices)*len(seeds) or not seeds:
                raise ValueError("incomplete condition coverage")
            seed_acc = [np.mean([r["correct"] for r in rows if r["seed"] == s])*100 for s in seeds]
            return dict(config=cfgid, dtype=cfg["dtype"], condition=condition, n=len(indices), seeds=",".join(map(str, seeds)),
                        accuracy=float(np.mean(seed_acc)), seed_sd=float(np.std(seed_acc, ddof=1)) if len(seeds)>1 else None,
                        strict_accuracy=float(np.mean([r["strict_correct"] for r in rows])*100),
                        mean_tokens=float(np.mean([r["token_count"] for r in rows])),
                        hit_max=float(np.mean([r["hit_max"] for r in rows])),
                        closure=float(np.mean([r["closed"] for r in rows])))

        def vector(condition, indices):
            rows = condition_rows(condition, indices)
            return np.asarray([np.mean([r["correct"] for r in rows if r["index"] == i]) for i in indices])

        for c in conditions:
            if c in ("G", "V", "S", "SV"):
                tables["table6"].append(stats(c, split["test"]))
        g, v = vector("G", split["test"]), vector("V", split["test"])
        native = v-g
        tables["table7"].append(dict(config=cfgid, effect="V-G", **paired_interval(native)))
        if "direct" in meta["scopes"]:
            s, sv = vector("S", split["test"]), vector("SV", split["test"])
            for label, values in (("SV-S", sv-s), ("interaction", sv-s-native)):
                tables["table7"].append(dict(config=cfgid, effect=label, **paired_interval(values)))
        if "replay" in meta["scopes"] and cfg["dataset"] == "math":
            for c in ("G", "V", "token_replay", "thought_replay"):
                tables["table2"].append(stats(c, split["replay"]))
        if whole:
            for stage in ("q10", "q25", "q50", "all"):
                e, ev = vector("E_"+stage, split["test"]), vector("EV_"+stage, split["test"])
                tables["figure6"].append(dict(config=cfgid, stage=stage, n=400,
                    native_gain=float(native.mean()*100), E=float(e.mean()*100), EV=float(ev.mean()*100),
                    **paired_interval(ev-e)))
        cal = read(output / "calibration" / f"{cfgid}.json")
        tables["table8"].append(dict(config=cfgid, n=len(cal["indices"]), candidates=len(cal["direct"]["all_candidates"]),
                                    **cal["direct"]["selected"]))
        details[cfgid] = dict(baseline_source="shared SEAL BF16 baseline" if cfg["method"] == "cps" else "own wrapper/precision",
                              calibration_sha256=digest(cal))
    dest = output / "summary"
    dest.mkdir(exist_ok=True)
    for key, values in tables.items():
        write_csv(dest/f"{key}.csv", values)
    write(dest/"summary.json", dict(run_id=meta["run_id"], tables=tables, provenance=details,
                                  primary_grading="legacy parser/percentage convention", strict_sensitivity="untrimmed text; no implicit x100 tolerance",
                                  bootstrap="10000 paired problem resamples after within-item seed mean"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--cpu-workers", type=int, default=4)
    args = p.parse_args()
    output = Path(args.output).resolve()
    meta, expected = read(output/"RUN.json"), read(output/"EXPECTED.json")
    if meta["smoke"] or signature() != meta["code_signature"]:
        raise ValueError("cannot report smoke or changed code as formal results")
    from task import plan_jobs
    if expected != plan_jobs(meta["configs"], meta["scopes"]):
        raise ValueError("expected task list changed; cannot silently report missing conditions")
    evaluator = evaluator_identity()
    evaluator_id = digest(evaluator)
    write(output/"EVALUATOR.json", dict(evaluator_id=evaluator_id, **evaluator))
    tasks = []
    indexed = {ds: data(ds) for ds in ("math", "gsm")}
    for job in expected:
        source = path_for(output, **job)
        row = checked_result(source, meta["run_id"], job)
        if row["runtime"] != meta["runtime"]:
            raise ValueError("mixed numerical runtimes")
        cfg = configurations()[job["config_id"]]
        if row["dtype"] != cfg["dtype"]:
            raise ValueError("dtype mismatch")
        if job["condition"].startswith(("S", "E")):
            if row["rng_seed"] != generation_seed(cfg, job["index"], job["seed"], job["condition"] in ("S", "SV")):
                raise ValueError("sampling RNG seed mismatch")
            cal = read(output/"calibration"/f"{cfg['id']}.json")
            if row["calibration_sha256"] != digest(cal):
                raise ValueError("mixed calibration versions")
            if job["condition"] in ("S", "SV"):
                if any(row["kernel"][k] != cal["direct"]["selected"][k] for k in ("temperature", "top_p")):
                    raise ValueError("sampling kernel mismatch")
            elif row["threshold"] != cal["thresholds"].get(job["condition"].split("_")[-1]):
                raise ValueError("whole-thought threshold mismatch")
            elif row["kernel"] != {"temperature": .6, "top_p": .95}:
                raise ValueError("whole-thought sampling kernel mismatch")
        native = job["condition"] in ("V", "SV") or job["condition"].startswith("EV_")
        expected_cache = "native" if native else "clean_replay" if job["condition"].endswith("replay") else "unsteered"
        if row["cache_policy"] != expected_cache:
            raise ValueError("cache policy mismatch")
        if job["index"] not in splits(cfg["dataset"])["test"]:
            continue
        destination = output/"grading"/source.relative_to(output/"results")
        tasks.append((str(source), str(destination), meta["run_id"], indexed[cfg["dataset"]][job["index"]]["answer"], evaluator_id))
    graded, errors, disagreements = {}, [], []
    with ProcessPoolExecutor(args.cpu_workers, mp_context=mp.get_context("fork")) as pool:
        for payload, row in zip(tasks, pool.map(grade_job, tasks, chunksize=1)):
            if row["status"] != "ok":
                errors.append(dict(source=payload[0], **row))
                continue
            group = graded.setdefault(str(Path(payload[0]).parent), {})
            group.setdefault(row["index"], []).append(row)
            if row["correct"] != row["strict_correct"] or row["prediction"] != row["strict_prediction"]:
                disagreements.append(row)
    write(output/"grading_errors.json", errors)
    write(output/"grading_sensitivity.json", disagreements)
    if errors:
        raise RuntimeError(f"{len(errors)} evaluator errors/timeouts; retained GPU results; rerun report.py to retry CPU grading")
    make_reports(output, meta, graded)
    files = [output/"RUN.json", output/"EXPECTED.json", output/"EVALUATOR.json", output/"grading_errors.json", output/"grading_sensitivity.json"]
    for name in ("results", "profiles", "calibration", "grading", "summary", "smoke", "logs"):
        files.extend(p for p in (output/name).rglob("*") if p.is_file() and ".tmp." not in p.name)
    write(output/"MANIFEST.json", {str(f.relative_to(output)):sha(f) for f in files})
    files.append(output/"MANIFEST.json")
    archive = output.parent / f"{output.name}_results.tar.gz"
    tmp = archive.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(output)), recursive=False)
    os.replace(tmp, archive)
    print(f"Complete. Return this file: {archive}")
    if disagreements:
        print(f"Grading sensitivity: {len(disagreements)} records; see grading_sensitivity.json before publication.")


if __name__ == "__main__":
    main()
