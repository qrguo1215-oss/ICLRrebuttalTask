"""Blinded, equal-pool strategy comparison. Never manufacture judge labels."""
import argparse
from pathlib import Path
import random

from bridge import ROOT
from protocol import PRIMARY, seeded
from spec import data, read, write, digest
from causal_stats import interval


RUBRIC = """Compare the mathematical strategy of the QUERY and each CANDIDATE.
Ignore lexical overlap, stylistic phrasing, length and whether their final answer agrees.
Same strategy requires compatible main mathematical reduction, intermediate dependencies,
and essential operations/case distinctions. Two correct answers alone are insufficient.
Return one of true, false, or null (genuinely uncertain) per candidate, and a brief rationale.
Do not infer which system generated any text. Do not use the private allocation key.
"""


def prepare(output, meta, raw, graded):
    packets, allocation, excluded = [], [], []
    indices = sorted({k[1] for k in graded if k[0] == PRIMARY and k[2] == "official_full"})
    for i in indices:
        v, g = graded[PRIMARY, i, "V", 0], graded[PRIMARY, i, "G", 0]
        if not v["correct"] or g["correct"]:
            continue
        if not v["strict_correct"] or g["strict_correct"]:
            excluded.append(dict(index=i, reason="rescue_label_depends_on_grading_convention"))
            continue
        correct = sorted(k[3] for k in graded if k[:3] == (PRIMARY, i, "official_full") and
                         graded[k]["correct"] and graded[k]["strict_correct"])
        # Fixed disjoint proposal pools, never optimize pair similarity.
        candidates = [s for s in correct if s < 4]
        queries = [s for s in correct if s >= 4]
        if len(candidates) < 2 or not queries:
            excluded.append(dict(index=i, reason="fewer_than_two_poolA_correct_or_no_poolB_correct"))
            continue
        rng = random.Random(seeded("blind-pool", i))
        target_seeds = rng.sample(candidates, 2)
        query_seed = rng.choice(queries)
        targets = [raw[PRIMARY, i, "official_full", s]["response_text"] for s in target_seeds]
        for arm, query in (("steering_vs_sample", raw[PRIMARY, i, "V", 0]["response_text"]),
                           ("sample_vs_sample", raw[PRIMARY, i, "official_full", query_seed]["response_text"])):
            identity = digest(dict(run=meta["run_id"], index=i, arm=arm))[:20]
            packets.append(dict(id=identity, problem=data("math")[i]["problem"], query=query, candidates=targets))
            allocation.append(dict(id=identity, index=i, arm=arm, target_seeds=target_seeds,
                                   query_seed=query_seed if arm == "sample_vs_sample" else None))
    random.Random(20261108).shuffle(packets)
    dest = output/"counterparts"
    write(dest/"blind_packet.json", dict(rubric=RUBRIC, pool_size=2, entries=packets))
    write(dest/"private_allocation_DO_NOT_SEND_TO_JUDGE.json", allocation)
    write(dest/"labels_template.json", [dict(id=p["id"], matches=[None, None], rationale="") for p in packets])
    write(dest/"coverage.json", dict(packet_pairs=len(allocation)//2, excluded=excluded,
                                    scope="grading-concordant greedy-wrong/native-correct; sufficient concordant correct samples in disjoint pools",
                                    status="awaiting_external_labels" if packets else "insufficient_correct_samples"))
    (dest/"README.txt").write_text(
        "Send ONLY blind_packet.json to independent human/LLM judges.\n"
        "Return labels_template.json with true/false/null matches and nonempty rationales.\n"
        "Record judge model/version/prompt and retain independent adjudication for uncertain cases.\n"
        "Never send private_allocation_DO_NOT_SEND_TO_JUDGE.json.\n"
        "Then run: python counterparts.py --output RUN_DIRECTORY --judgments LABELS.json --judge-description DESCRIPTION\n"
        "No strategy conclusion can be reported before actual judgments return.\n", encoding="utf-8")


def score(output, judgments, judge_description):
    packet = read(output/"counterparts/blind_packet.json")
    allocation = read(output/"counterparts/private_allocation_DO_NOT_SEND_TO_JUDGE.json")
    labels = read(judgments)
    by_id = {r["id"]: r for r in labels}
    if len(by_id) != len(labels) or set(by_id) != {p["id"] for p in packet["entries"]}:
        raise ValueError("Missing/duplicate/unknown blind judgment IDs")
    by_item, unknown = {}, []
    for a in allocation:
        label = by_id[a["id"]]
        if len(label["matches"]) != 2 or any(type(x) is not bool and x is not None for x in label["matches"]) or not label.get("rationale", "").strip():
            raise ValueError("Require two boolean/null labels and nonempty rationale")
        verdict = True if True in label["matches"] else None if None in label["matches"] else False
        by_item.setdefault(a["index"], {})[a["arm"]] = verdict
    differences = []
    for i, arms in by_item.items():
        if None in arms.values():
            unknown.append(i)
        else:
            differences.append(int(arms["steering_vs_sample"])-int(arms["sample_vs_sample"]))
    result = dict(judge_description=judge_description, unresolved_indices=unknown, paired_items=by_item,
                  estimate=interval(differences) if differences else None,
                  interpretation="difference vs equal-size correct-correct baseline; not proof of new or identical knowledge")
    write(output/"counterparts/returned_labels.json", labels)
    write(output/"counterparts/scored.json", result)
    from analysis import bundle
    bundle(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--judge-description", required=True)
    args = parser.parse_args()
    score(args.output.resolve(), args.judgments, args.judge_description)
