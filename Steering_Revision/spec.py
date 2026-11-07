"""Portable frozen experiment specification; safe to import without torch."""
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "steering-revision-portable-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def generation_seed(config, index, seed, boundary_sampling=False):
    if not boundary_sampling and config["method"] == "cps" and config["dataset"] == "math":
        raw = f"20260917:{index}:{seed}:whole-thought-tokens".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31-1)
    salt = "matched-local" if boundary_sampling else "whole-thought-tokens"
    raw = f"20260920|{index}|{seed}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63-1)


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def data(dataset):
    rows = [json.loads(s) for s in (ROOT / f"assets/data/{dataset}.jsonl").read_text().splitlines() if s.strip()]
    result = {int(r["index"]): r for r in rows}
    if len(result) != 500 or len(rows) != 500:
        raise ValueError("expected exactly 500 unique data records")
    return result


def splits(dataset):
    return read(ROOT / f"assets/splits/{dataset}.json")


def configurations():
    configs = {}
    for size in ("15", "7"):
        for method in ("seal", "rebalance", "cps"):
            if size == "7" and method == "cps":
                continue
            for ds in ("math", "gsm"):
                key = f"{ds}{size}_{method}"
                configs[key] = dict(id=key, size=size, method=method, dataset=ds,
                                    layer=19 if size == "15" and method == "rebalance" else 20,
                                    coefficient=1. if method == "cps" else -1.,
                                    dtype="float16" if method == "rebalance" else "bfloat16",
                                    vector=f"assets/vectors/{size}_{method}.pt")
    return configs


def source_config(config_id, condition):
    # Same unsteered wrapper, precision, prompt and seed; verify zero identity in smoke.
    return config_id.replace("_cps", "_seal") if condition == "G" else config_id


def path_for(output, config_id, index, condition, seed=0):
    config_id = source_config(config_id, condition)
    return Path(output) / "results" / config_id / condition / f"{index:04d}_{seed}.json.gz"


def make_job(config_id, index, condition, seed=0):
    return dict(config_id=source_config(config_id, condition), index=index, condition=condition, seed=seed)


def job_key(job):
    return job["config_id"], int(job["index"]), job["condition"], int(job["seed"])


def jobs(config_id, scope):
    cfg = configurations()[config_id]
    split = splits(cfg["dataset"])
    test = split["test"]
    if scope == "baseline":
        return [make_job(config_id, i, "G") for i in sorted(test + split["calibration"])]
    if scope == "native":
        return [make_job(config_id, i, "V") for i in test]
    if scope == "replay":
        return [make_job(config_id, i, c) for i in split.get("replay", []) for c in ("token_replay", "thought_replay")]
    if scope == "direct":
        return [make_job(config_id, i, c, s) for i in test for c in ("S", "SV") for s in range(8)]
    if scope == "whole":
        result = []
        for stage in ("q10", "q25", "q50", "all"):
            seeds = range(4) if config_id == "math15_cps" and stage == "all" else range(2) if config_id == "math15_cps" else [0]
            result.extend(make_job(config_id, i, f"{c}_{stage}", s) for i in test for c in ("E", "EV") for s in seeds)
        return result
    raise ValueError(scope)


def verify_assets():
    manifest = read(ROOT / "assets/checksums.json")
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"asset checksum mismatch: {name}")
    for ds in ("math", "gsm"):
        s = splits(ds)
        if len(s["calibration"]) != 100 or len(s["test"]) != 400:
            raise ValueError("100/400 split required")
        if len(set(s["calibration"] + s["test"])) != 500:
            raise ValueError("duplicate/overlapping split")
        if set(data(ds)) != set(s["calibration"] + s["test"]):
            raise ValueError("dataset and split mismatch")
        if ds == "math" and (len(set(s["replay"])) != 200 or not set(s["replay"]) <= set(s["test"])):
            raise ValueError("invalid fixed deep200")
    for key in configurations():
        c = read(ROOT / f"assets/calibration/{key}.json")
        if c["cases"] != 100 or c["selection_uses_correctness"] or "no mixture gate" not in c["kernel"]:
            raise ValueError("invalid historical direct calibration")
        if any("gate_probability" in x for x in c["all_candidates"]):
            raise ValueError("mixture kernel is forbidden")
    return manifest


def signature():
    # CPU evaluator/report fixes must not invalidate expensive raw generations.
    paths = sorted([ROOT / name for name in ("native.py", "engine.py", "spec.py", "task.py", "calibration.py")]+
                   [ROOT / "vendor" / name for name in ("seal_model.py", "rebalance_model.py")]+
                   [p for p in (ROOT / "assets").rglob("*.json") if p.name != "checksums.json"] + list((ROOT / "assets").rglob("*.jsonl")) +
                   list((ROOT / "assets").rglob("*.pt")) + [ROOT / "requirements.txt"])
    return digest({"version": VERSION, "files": {str(p.relative_to(ROOT)): sha(p) for p in paths}})


def checked_result(path, run_id, job=None, allow_smoke=False):
    row = read(path)
    if row["run_id"] != run_id or (row.get("smoke") and not allow_smoke):
        raise ValueError(f"incompatible result: {path}")
    if job and job_key(row) != job_key(job):
        raise ValueError(f"result key mismatch: {path}")
    if not row.get("smoke") and row["max_new_tokens"] != 10000:
        raise ValueError("truncated debug record in formal results")
    return row
