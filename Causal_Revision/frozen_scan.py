"""Label-free calibration on the immutable 100-question calibration split."""
import numpy as np
import torch

import bridge
from calibration import compress_event, merge_profiles, finish
from engine import owned_cache
from native import make_prompt
from kernels import direction
from protocol import RANDOM_DIRECTIONS, RANDOM_SCALES


@torch.inference_mode()
def scan(engine, item, baseline, candidates, random_audit=False, smoke=False):
    engine.reset()
    encoded = engine.tokenizer(make_prompt(engine.tokenizer, item["problem"]), return_tensors="pt").to(engine.model.device)
    mask = encoded["attention_mask"]
    out = engine.forward(encoded["input_ids"], mask)
    cache = owned_cache(out.past_key_values)
    if int(out.logits[0, -1].argmax()) != baseline["token_ids"][0]:
        raise ValueError("Frozen-prefix prefill differs from recorded baseline")
    original = engine.vector
    numbers = [0] if smoke else list(RANDOM_DIRECTIONS)
    random_vectors = {n: direction(original, engine.config["id"], n) for n in numbers} if random_audit else {}
    profiles, positions, margins, adjacent_margins, native_gaps, random_gaps = [], [], [], [], [], []
    previous_boundary = False
    think = bool((encoded["input_ids"] == engine.start).any().item())
    for pos, token in enumerate(baseline["token_ids"][:-1]):
        if token == engine.eos:
            break
        if token == engine.start:
            think = True
        if token == engine.end:
            think = False
        mask = engine.extend(mask)
        boundary = think and token in engine.splits
        if boundary:
            native = engine.shadow(engine.query(token), mask, cache, native=True).logits[0, -1].float()
            random_tokens = []
            try:
                for n in random_vectors:
                    for scale in RANDOM_SCALES:
                        engine.vector = random_vectors[n] * scale
                        donor = engine.shadow(engine.query(token), mask, cache, native=True)
                        random_tokens.append(int(donor.logits[0, -1].argmax()))
                        del donor
            finally:
                engine.vector = original
        out = engine.forward(engine.query(token), mask, cache)
        cache = owned_cache(out.past_key_values)
        base = out.logits[0, -1].float()
        if int(base.argmax()) != baseline["token_ids"][pos+1]:
            raise ValueError(f"Frozen-prefix replay is not identical at token {pos+1}")
        if boundary:
            profiles.append(compress_event(base, native, candidates))
            positions.append(pos)
            margins.append(float((base.topk(2).values[0]-base.topk(2).values[1]).item()))
            native_gaps.append(float((base.max()-base[native.argmax()]).item()))
            random_gaps.append([float((base.max()-base[t]).item()) for t in random_tokens])
        if think and previous_boundary and not boundary:
            adjacent_margins.append(float((base.topk(2).values[0]-base.topk(2).values[1]).item()))
        previous_boundary = boundary
        del out
    if not profiles:
        raise ValueError("No eligible calibration split; do not silently drop or replace the problem")
    grid, mass, native_mass = merge_profiles(profiles, native_points=True)
    return dict(grid=grid, mass=mass, native_mass=native_mass, events=len(profiles), positions=positions,
                margins=margins, adjacent_margins=adjacent_margins, native_gaps=native_gaps, random_gaps=random_gaps,
                random_numbers=list(random_vectors), random_scale_grid=list(RANDOM_SCALES),
                native_flip=float(np.mean([r["native_flip"] for r in profiles])),
                sample_flip=np.mean([r["sample_flip"] for r in profiles], axis=0))


def freeze(profiles, candidates):
    direct, cdf = finish(profiles, candidates)
    # Equal-problem quantiles: resampling each problem to equal event weight is
    # unnecessary; evaluate the weighted empirical CDF directly.
    def quantiles(field):
        nonempty = [p[field] for p in profiles if len(p[field])]
        if len(nonempty) != len(profiles):
            raise ValueError(f"Missing {field}; do not silently change calibration population")
        margins = np.concatenate(nonempty)
        weights = np.concatenate([np.full(len(p), 1./len(p)/len(profiles)) for p in nonempty])
        order = np.argsort(margins, kind="stable")
        return {label: float(margins[order[np.searchsorted(np.cumsum(weights[order]), q)]])
                for label, q in (("q25", .25), ("q50", .5))}
    thresholds = quantiles("margins")
    adjacent = quantiles("adjacent_margins")
    random = {}
    for ix, n in enumerate(profiles[0]["random_numbers"]):
        scores = []
        for si, scale in enumerate(RANDOM_SCALES):
            # W1 within each problem, then equal-problem averaging. No labels.
            vals = [np.abs(np.sort(p["native_gaps"])-np.sort(p["random_gaps"][:, ix*len(RANDOM_SCALES)+si])).mean()
                    for p in profiles]
            scores.append(dict(scale=scale, w1=float(np.mean(vals))))
        selected = min(scores, key=lambda x: (x["w1"], x["scale"]))
        random[str(int(n))] = dict(selected_scale=selected["scale"], all_scales=scores)
    return dict(direct=direct, thresholds=thresholds, thresholds_adjacent=adjacent, random_scales=random,
                quantile_weighting="equal problem, equal boundary within problem", selection_uses_correctness=False), cdf
