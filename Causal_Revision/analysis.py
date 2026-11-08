"""Separate CPU-only grading and audit; raw generations remain immutable."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import os
import tarfile

import numpy as np

from bridge import ROOT
from protocol import PRIMARY, plan, key, job, result_path, checked, code_signature, seeded
from spec import read, write, sha, digest, data, splits, configurations
from report import evaluate, evaluator_identity, gold_answer, write_csv
from causal_stats import interval, mcnemar, headroom_null, wilson


def grade(payload):
    source, dest, meta, j, gold, evaluator = payload
    identity = dict(source_sha256=sha(source), evaluator_id=evaluator)
    if Path(dest).exists():
        old = read(dest)
        if all(old.get(k) == v for k, v in identity.items()) and old["status"] == "ok":
            return old
    row = checked(source, meta, j)
    result = evaluate(row, gold)
    result.update(j, **identity, run_id=meta["run_id"])
    write(dest, result)
    return result


def validate_micro(rows):
    groups = defaultdict(list)
    for r in rows.values():
        if r["condition"].startswith("micro_"):
            groups[r["config_id"], r["index"]].append(r)
    for group in groups.values():
        available = [r.get("status", "ok") == "ok" for r in group]
        if any(available) != all(available):
            raise ValueError("Micro eligibility differs across arms/seeds")
        if not any(available):
            continue
        if len({r["frozen_prefix_sha256"] for r in group}) != 1:
            raise ValueError("Micro conditions did not start from the same prefix")
        for r in group:
            interventions = [e for e in r["events"] if e.get("forced_departure")]
            expected = 0 if r["condition"] == "micro_greedy" else 1
            if len(interventions) != 1 or sum(e["departure"] for e in interventions) != expected:
                raise ValueError("Micro experiment must have exactly one forced alternative (zero for baseline)")


def reports(output, meta, raw, graded, repeats=10000):
    summaries, contrasts, seeds, depths, occlusion, random_rows = [], [], [], [], [], []
    dest = output/"summary"
    dest.mkdir(exist_ok=True)
    groups = defaultdict(dict)
    opportunity = {}
    for k, r in graded.items():
        groups[k[0], k[2]].setdefault(k[1], {})[k[3]] = r
    for (cfgid, c), items in groups.items():
        ids = sorted(items)
        selected_seeds = sorted(items[ids[0]])
        if any(sorted(items[i]) != selected_seeds for i in ids):
            raise ValueError("Unequal seed coverage within condition")
        acc = np.array([[items[i][s]["correct"] for s in selected_seeds] for i in ids], float)
        g = np.array([graded[(cfgid, i, "G", 0)]["correct"] for i in ids], float)
        v = np.array([graded[(cfgid, i, "V", 0)]["correct"] for i in ids], float)
        records = [raw[(cfgid, i, c, s)] for i in ids for s in selected_seeds]
        summary = dict(config=cfgid, condition=c, n=len(ids), seeds=",".join(map(str, selected_seeds)),
                       accuracy=float(acc.mean()), seed_sd=float(acc.mean(axis=0).std(ddof=1)) if len(selected_seeds)>1 else None,
                       strict_accuracy=float(np.mean([items[i][s]["strict_correct"] for i in ids for s in selected_seeds])),
                       rescue=float(acc[g == 0].mean()) if (g == 0).any() else None,
                       harm=float((1-acc[g == 1]).mean()) if (g == 1).any() else None,
                       baseline_accuracy=float(g.mean()), native_accuracy=float(v.mean()),
                       mean_tokens=float(np.mean([r["token_count"] for r in records])),
                       hit_max=float(np.mean([r["hit_max"] for r in records])),
                       eos_rate=float(np.mean([r["eos_reached"] for r in records])),
                       boxed_rate=float(np.mean([r["closed"] for r in records])),
                       sampled_tokens=float(np.mean([r.get("sampled_tokens", 0) for r in records])),
                       sampled_thoughts=float(np.mean([r.get("sampled_thoughts", 0) for r in records])),
                       mean_think_splits=float(np.mean([sum(e.get("boundary", True) for e in r["events"]) for r in records])),
                       repeated_8gram_fraction=float(np.mean([surface_repetition(r["token_ids"]) for r in records])),
                       forward_calls=float(np.mean([r.get("forward_calls", np.nan) for r in records])) if all("forward_calls" in r for r in records) else None,
                       wall_seconds=float(np.mean([r["wall_seconds"] for r in records])),
                       timing_caveat="instrumented, concurrent workers; not a hardware-matched throughput benchmark",
                       gain_ci=interval(acc.mean(axis=1)-g, repeats))
        summaries.append(summary)
        for sn, s in enumerate(selected_seeds):
            seeds.append(dict(config=cfgid, condition=c, seed=s, n=len(ids), accuracy=float(acc[:, sn].mean()), **mcnemar(g, acc[:, sn])))
        if c.startswith(("random_", "reverse")):
            random_rows.append(dict(config=cfgid, condition=c, accuracy=summary["accuracy"], native_accuracy=summary["native_accuracy"],
                                    native_minus_control=interval(v-acc.mean(axis=1), repeats)))
        if c.startswith(("depth_", "micro_", "boundary_official")):
            eventrows = [e for r in records for e in r["events"] if e.get("gate_probability") is not None]
            departed = [e["deficit"] for e in eventrows if e["departure"]]
            depths.append(dict(config=cfgid, condition=c, n=len(ids),
                               events=len(eventrows), departures=sum(e["departure"] for e in eventrows),
                               mean_departures_per_generation=sum(e["departure"] for e in eventrows)/len(records),
                               zero_gap_departures=sum(e["departure"] and e["deficit"] == 0 for e in eventrows),
                               mean_conditional_deficit=float(np.mean(departed)) if departed else None,
                               live_count_matching_claim=False,
                               forced_single_departure=c.startswith("micro_") and c != "micro_greedy"))
    for cfgid in meta["configs"]:
        for left, right in (("depth_matched", "depth_shallow"), ("depth_matched", "depth_deep"),
                            ("micro_matched", "micro_shallow"), ("micro_matched", "micro_deep"),
                            ("micro_matched", "micro_greedy"), ("V", "token_replay"), ("V", "thought_replay"),
                            ("cache_only", "G")):
            a, b = groups.get((cfgid, left)), groups.get((cfgid, right))
            if not a or not b:
                continue
            ids = sorted(set(a) & set(b))
            differences = [np.mean([r["correct"] for r in a[i].values()])-np.mean([r["correct"] for r in b[i].values()]) for i in ids]
            contrasts.append(dict(config=cfgid, left=left, right=right, estimand="left minus right", **interval(differences, repeats)))
    # q discovery and effect estimation use disjoint seed ranges; no selection on S failures.
    for site in ("boundary", "adjacent"):
        for stage in ("q25", "q50", "all"):
            names = [f"{c}_{site}_{stage}" for c in ("whole", "wholeV", "discover")]
            if not all((PRIMARY, n) in groups for n in names):
                continue
            a, b, discovery = [groups[PRIMARY, n] for n in names]
            ids = sorted(set(a) & set(b) & set(discovery))
            ss = sorted(a[ids[0]])
            if set(ss) & set(discovery[ids[0]]) or any(sorted(a[i]) != sorted(b[i]) for i in ids):
                raise ValueError("Discovery/evaluation seeds overlap or effect seeds are unpaired")
            q = np.array([np.mean([r["correct"] for r in discovery[i].values()]) for i in ids])
            x = np.array([[a[i][s]["correct"] for s in ss] for i in ids], float)
            y = np.array([[b[i][s]["correct"] for s in ss] for i in ids], float)
            g = np.array([graded[PRIMARY, i, "G", 0]["correct"] for i in ids], float)
            v = np.array([graded[PRIMARY, i, "V", 0]["correct"] for i in ids], float)
            delta = (y-x).mean(axis=1)
            itemrows = [dict(index=i, q_discovery=float(q[n]),
                             q_wilson_low=wilson(sum(r["correct"] for r in discovery[i].values()), len(discovery[i]))[0],
                             q_wilson_high=wilson(sum(r["correct"] for r in discovery[i].values()), len(discovery[i]))[1],
                             clean=float(x[n].mean()), steered=float(y[n].mean()), delta=float(delta[n]),
                             greedy_correct=bool(g[n]), native_correct=bool(v[n])) for n, i in enumerate(ids)]
            write_csv(dest/f"items_{site}_{stage}.csv", itemrows)
            opportunity[site, stage] = {i: dict(q=float(q[n]), delta=float(delta[n]), greedy=bool(g[n])) for n, i in enumerate(ids)}
            for label, keep in (("all", np.ones(len(ids), bool)), ("greedy_wrong", g == 0), ("greedy_correct", g == 1)):
                if not keep.any():
                    continue
                null = headroom_null(q, x, y, [seeded("fold", i)%5 for i in ids], g, v, mask=keep, repeats=repeats)
                occlusion.append(dict(site=site, stage=stage, stratum=label, n=int(keep.sum()),
                                      marginal=interval(delta[keep], repeats),
                                      interaction=interval(delta[keep]-(v-g)[keep], repeats), headroom_null=null,
                                      evidence_scope="occlusion and a restricted headroom null; neither proves shared mechanism"))
            for label, low, high in (("q0_25", 0, .25), ("q25_50", .25, .5), ("q50_75", .5, .75), ("q75_100", .75, 1.0001)):
                keep = (q >= low) & (q < high) & (g == 0)
                if keep.any():
                    occlusion.append(dict(site=site, stage=stage, stratum=label, n=int(keep.sum()), marginal=interval(delta[keep], repeats)))
    common_support = []
    for stage in ("q25", "q50", "all"):
        a, b = opportunity.get(("boundary", stage)), opportunity.get(("adjacent", stage))
        if not a or not b:
            continue
        for bin_no in range(4):
            # Same questions, same predeclared discovery-q bin for BOTH policies.
            ids = [i for i in a if i in b and not a[i]["greedy"] and
                   min(3, int(a[i]["q"]*4)) == bin_no == min(3, int(b[i]["q"]*4))]
            if ids:
                common_support.append(dict(stage=stage, q_bin=bin_no, n=len(ids), indices=ids,
                                           boundary_minus_adjacent=interval([a[i]["delta"]-b[i]["delta"] for i in ids], repeats),
                                           mean_q_difference=float(np.mean([a[i]["q"]-b[i]["q"] for i in ids])),
                                           limitation="descriptive common-support comparison; noisy q, not exact equal-headroom intervention"))
    direction_summary = []
    for cfgid in meta["configs"]:
        for group in ("random_norm_", "random_calibrated_"):
            values = [r for r in random_rows if r["config"] == cfgid and r["condition"].startswith(group)]
            if values:
                native = values[0]["native_accuracy"]
                accuracies = np.array([r["accuracy"] for r in values])
                direction_summary.append(dict(config=cfgid, group=group, directions=len(values),
                                              mean_accuracy=float(accuracies.mean()),
                                              direction_sd=float(accuracies.std(ddof=1)) if len(values)>1 else None,
                                              native_midrank_percentile=float(100*((accuracies < native).mean()+.5*(accuracies == native).mean()))))
    write(dest/"summary.json", dict(run_id=meta["run_id"], conditions=summaries, contrasts=contrasts,
                                   random_directions=random_rows, occlusion=occlusion,
                                   random_distribution=direction_summary, common_headroom_support=common_support,
                                   interpretation="No sign/equivalence/shared-mechanism conclusion is selected automatically",
                                   bootstrap="10000 paired problem resamples; within-problem fixed evaluation-seed mean",
                                   efficiency="tokens/EOS/cap reported; boxed is formatting, not correctness or true closure",
                                   strategy_judging="Requires blind external labels; see counterparts/README.txt"))
    for name, rows in (("conditions", summaries), ("contrasts", contrasts), ("per_seed", seeds), ("depth_audit", depths)):
        flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
        write_csv(dest/f"{name}.csv", flat)
    plots(dest, summaries, occlusion)


def surface_repetition(tokens):
    grams = [tuple(tokens[n:n+8]) for n in range(max(0, len(tokens)-7))]
    return 1-len(set(grams))/len(grams) if grams else 0.


def plots(dest, summaries, occlusion):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cfgid in sorted({r["config"] for r in summaries}):
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in summaries:
            if r["config"] == cfgid and r["condition"] in ("G", "V", "depth_matched", "depth_shallow", "depth_deep", "boundary_official") and r["harm"] is not None and r["rescue"] is not None:
                ax.scatter(100*r["harm"], 100*r["rescue"], label=r["condition"])
        ax.set(xlabel="Harm on initially correct items (%)", ylabel="Rescue on initially wrong items (%)", title=cfgid)
        if ax.collections:
            ax.legend()
        fig.tight_layout()
        fig.savefig(dest/f"{cfgid}_rescue_harm.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for site in ("boundary", "adjacent"):
        rows = [r for r in occlusion if r["site"] == site and r["stratum"] == "all"]
        if rows:
            x = list(range(len(rows)))
            y = [100*r["marginal"]["mean"] for r in rows]
            err = [[max(0, yy-100*r["marginal"]["ci_low"]) for yy, r in zip(y, rows)],
                   [max(0, 100*r["marginal"]["ci_high"]-yy) for yy, r in zip(y, rows)]]
            ax.errorbar(x, y, yerr=err, marker="o", label=site)
            ax.set_xticks(x, [r["stage"] for r in rows])
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xlabel="Frozen exploration gate", ylabel="Additional native gain (pp)")
    if ax.lines and occlusion:
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest/"occlusion.png", dpi=160)
    plt.close(fig)


def bundle(output):
    folders = ("results", "grading", "profiles", "calibration", "summary", "audits", "logs", "counterparts")
    files = [output/n for n in ("RUN.json", "JOBS.json", "EVALUATOR.json", "grading_errors.json", "coverage.json") if (output/n).exists()]
    files += [p for name in folders for p in (output/name).rglob("*") if p.is_file() and ".tmp." not in p.name]
    write(output/"MANIFEST.json", {str(p.relative_to(output)): sha(p) for p in files})
    archive = output.parent/f"{output.name}_results.tar.gz"
    tmp = archive.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for p in files+[output/"MANIFEST.json"]:
            tar.add(p, arcname=str(p.relative_to(output)), recursive=False)
    os.replace(tmp, archive)
    print(f"Return archive: {archive}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cpu-workers", type=int, default=4)
    ap.add_argument("--smoke-audit", action="store_true", help="Test CPU pipeline; never publish as experiment results")
    args = ap.parse_args()
    output = args.output.resolve()
    meta = read(output/"RUN.json")
    if meta["smoke"] != args.smoke_audit or meta["code_signature"] != code_signature():
        raise ValueError("Smoke/formal output or code mismatch")
    jobs = plan(meta["configs"], meta["modules"], meta["smoke"])
    if read(output/"JOBS.json") != jobs:
        raise ValueError("Task list does not match frozen protocol")
    evaluator = evaluator_identity()
    evaluator["causal_analysis"] = {n: sha(ROOT/n) for n in ("analysis.py", "causal_stats.py")}
    evaluator_id = digest(evaluator)
    write(output/"EVALUATOR.json", dict(evaluator_id=evaluator_id, **evaluator))
    raw, tasks, unavailable = {}, [], []
    for j in jobs:
        source = result_path(output, j)
        row = checked(source, meta, j)
        cfg = configurations()[j["config_id"]]
        if row.get("status") == "unavailable":
            if not j["condition"].startswith("micro_"):
                raise ValueError("Unexpected missing non-micro condition")
            unavailable.append(dict(j, reason=row["reason"]))
            raw[key(j)] = row
            continue
        if not meta["smoke"] and j["index"] not in splits(cfg["dataset"])["test"]:
            continue
        raw[key(j)] = row
        dest = output/"grading"/source.relative_to(output/"results")
        tasks.append((str(source), str(dest), meta, j, gold_answer(data(cfg["dataset"])[j["index"]]), evaluator_id))
    validate_micro(raw)
    for k, r in raw.items():
        if r["condition"] in ("self_cache", "self_token", "self_thought") and r["token_ids"] != raw[k[0], k[1], "G", 0]["token_ids"]:
            raise ValueError("Null replay changed tokens; causal results invalid")
    write(output/"coverage.json", dict(expected=len(jobs), unavailable_micro=unavailable, smoke=meta["smoke"]))
    graded, errors = {}, []
    with ProcessPoolExecutor(args.cpu_workers, mp_context=mp.get_context("fork")) as pool:
        for r in pool.map(grade, tasks, chunksize=1):
            if r["status"] == "ok":
                graded[key(r)] = r
            else:
                errors.append(r)
    write(output/"grading_errors.json", errors)
    if errors:
        raise RuntimeError("Grading failed; raw inference retained. Fix evaluator and rerun report, not generation.")
    write(output/"summary/grading_sensitivity.json", [r for r in graded.values()
          if r["correct"] != r["strict_correct"] or r["prediction"] != r["strict_prediction"]])
    reports(output, meta, raw, graded, repeats=100 if meta["smoke"] else 10000)
    if "counterparts" in meta["modules"]:
        from counterparts import prepare
        prepare(output, meta, raw, graded)
    bundle(output)


if __name__ == "__main__":
    main()
