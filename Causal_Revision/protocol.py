"""Frozen causal estimands and task coverage; no CUDA imports."""
import hashlib
from pathlib import Path

from bridge import ROOT, SHARED
from spec import configurations, data, digest, read, sha, signature, splits, write

VERSION = "causal-rebuttal-v1"
PRIMARY = "math15_cps"
DEFAULT_CONFIGS = [PRIMARY, "math15_seal", "math15_rebalance"]
MODULES = ("depth", "replay", "random", "occlusion", "counterparts")
SEEDS = list(range(8))
DISCOVERY_SEEDS = list(range(100, 108))
Q50_SEEDS = list(range(200, 216))
RANDOM_SCALES = (.25, .5, 1., 2., 4.)
RANDOM_DIRECTIONS = tuple(range(8))
MICRO_ORDINAL = 8


def code_signature():
    paths = [p for p in ROOT.glob("*.py") if p.name not in ("analysis.py", "counterparts.py", "causal_stats.py")]
    paths += [ROOT / "requirements.txt", ROOT / "run.sh"]
    return digest(dict(version=VERSION, shared_generation=signature(),
                       files={p.name: sha(p) for p in sorted(paths)}))


def seeded(*parts):
    raw = ":".join(map(str, (20261108,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)


def job(config, index, condition, seed=0):
    return dict(config_id=config, index=int(index), condition=condition, seed=int(seed))


def key(j):
    return j["config_id"], int(j["index"]), j["condition"], int(j["seed"])


def result_path(output, j):
    return output / "results" / j["config_id"] / j["condition"] / f"{j['index']:04d}_{j['seed']}.json.gz"


def plan(configs, modules, smoke=False):
    if not configs or len(configs) != len(set(configs)) or not set(configs) <= set(configurations()):
        raise ValueError("Unknown/duplicate config. No 7B CPS vector is registered.")
    if not modules or not set(modules) <= set(MODULES):
        raise ValueError("Unknown/empty module selection")
    if PRIMARY not in configs and set(modules) & {"occlusion", "counterparts"}:
        raise ValueError("Occlusion/counterparts are preregistered only for math15_cps; choose core modules for other configs")
    jobs = {}
    for cfgid in configs:
        cfg = configurations()[cfgid]
        s = splits(cfg["dataset"])
        calibration = s["calibration"][:1] if smoke else s["calibration"]
        test = calibration if smoke else s["test"]
        deep = calibration if smoke else s.get("replay", s["test"][:200])
        seeds = [0] if smoke else SEEDS

        def add(indices, conditions, ss=(0,)):
            for i in indices:
                for c in conditions:
                    for seed in ss:
                        j = job(cfgid, i, c, seed)
                        jobs[key(j)] = j

        add(sorted(set(calibration + test)), ["G"])
        add(test, ["V"])
        if "depth" in modules:
            add(test, ["depth_matched", "depth_shallow", "depth_deep", "boundary_official"], seeds)
            add(deep, ["micro_matched", "micro_shallow", "micro_deep"], seeds)
            add(deep, ["micro_greedy"])
        if "replay" in modules:
            add(deep, ["token_replay", "thought_replay", "cache_only", "self_token", "self_thought", "self_cache"])
        if "random" in modules:
            add(deep, ["reverse"])
            directions = [0] if smoke else RANDOM_DIRECTIONS
            add(deep, [f"random_norm_{d}" for d in directions])
            # Only the primary config receives the additional behavioural-dose audit.
            if cfgid == PRIMARY:
                add(deep, [f"random_calibrated_{d}" for d in directions])
        if "occlusion" in modules and cfgid == PRIMARY:
            for site in ("boundary", "adjacent"):
                for stage in ("q25", "q50", "all"):
                    ss = [0] if smoke else Q50_SEEDS if stage == "q50" else SEEDS
                    add(test, [f"whole_{site}_{stage}", f"wholeV_{site}_{stage}"], ss)
                    add(test, [f"discover_{site}_{stage}"], [100] if smoke else DISCOVERY_SEEDS)
        if "counterparts" in modules and cfgid == PRIMARY:
            # Neutral token-level sampling; not thought-gated, and never pass@k accuracy.
            add(test, ["official_full"], seeds)
    return sorted(jobs.values(), key=key)


def settings():
    return dict(version=VERSION, full_max_new_tokens=10000, smoke_max_new_tokens=256,
                batch_size=1, evaluation_seeds=SEEDS, q50_evaluation_seeds=Q50_SEEDS,
                discovery_seeds=DISCOVERY_SEEDS, micro_boundary_ordinal=MICRO_ORDINAL,
                random_directions=list(RANDOM_DIRECTIONS), random_scale_grid=list(RANDOM_SCALES),
                depth_temperature_factors=dict(shallow=.5, matched=1., deep=2.),
                depth_gate="exact reference nucleus departure probability at each current state",
                micro="one forced non-greedy action at the predeclared boundary; clean greedy tail",
                q50_caveat="new same-batch replication, not proof about historical batch cause",
                adjacent="sample after processing first non-split token immediately following a split",
                unknowns="no exact live count matching, no equivalence inferred from a non-significant difference",
                shape_control="not included: no claim that exact full-CDF matching is necessary",
                runtime_bound="sampling cumsum warn-only, other operators strict",
                random_calibration="all directions retained; scale minimizes problem-equal deficit W1 on calibration100",
                headroom="cross-fitted constant rescue/harm null, not an exclusion of all ceilings",
                counterparts="blind matched-pool human/LLM judging requires returned labels; never auto-fabricated")


def checked(path, meta, expected=None):
    path = Path(path)
    row = read(path)
    if row.get("run_id") != meta["run_id"] or row.get("smoke") != meta["smoke"]:
        raise ValueError(f"Stale result: {path}")
    if expected is not None and key(row) != key(expected):
        raise ValueError(f"Result key mismatch: {path}")
    if row.get("runtime") != meta["runtime"]:
        raise ValueError(f"Mixed inference environments: {path}")
    if row.get("max_new_tokens") != (256 if meta["smoke"] else 10000):
        raise ValueError(f"Debug token budget in formal output: {path}")
    if row.get("dtype") != configurations()[row["config_id"]]["dtype"]:
        raise ValueError(f"Wrong numerical precision: {path}")
    needs_calibration = row["condition"].startswith(("depth_", "micro_", "whole", "discover_", "random_calibrated_"))
    if needs_calibration:
        calpath = path.parents[3] / "calibration" / f"{row['config_id']}.json"
        if not calpath.exists() or row.get("calibration_sha256") != digest(read(calpath)):
            raise ValueError(f"Missing or mismatched frozen calibration: {path}")
    return row

