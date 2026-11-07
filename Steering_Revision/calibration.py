"""Exact empirical-deficit W1 using the SAME token-level kernel as generation."""
import numpy as np
import torch

from native import nucleus_from_sorted, make_prompt
from engine import owned_cache


def candidate_grid(reference):
    return sorted({(float(c["temperature"]), float(c["top_p"])) for c in reference["all_candidates"]})


def compress_event(base_logits, native_logits, candidates):
    base_logits = base_logits.float()
    sorted_logits, sorted_ids = torch.sort(base_logits, descending=True)
    deficits = (sorted_logits[0] - sorted_logits).cpu().numpy()
    grid, inverse = np.unique(deficits, return_inverse=True)
    greedy = int(base_logits.argmax().item())
    native = int(native_logits.argmax().item())
    masses, changes = [], []
    # Do not replace this with grouped FP64 softmax: cutoff rounding is part of the kernel.
    for temperature, top_p in candidates:
        probs = nucleus_from_sorted(sorted_logits, temperature, top_p)
        masses.append(np.bincount(inverse, weights=probs.cpu().numpy().astype(np.float64), minlength=len(grid)))
        changes.append(1 - float(probs[sorted_ids == greedy].sum().item()))
    return {"grid": grid, "mass": np.stack(masses), "native_gap": float((base_logits.max() - base_logits[native]).item()),
            "native_flip": int(native != greedy), "sample_flip": np.asarray(changes)}


def merge_profiles(profiles, native_points=False):
    grid = np.unique(np.concatenate([p["grid"] for p in profiles] +
                                    [np.asarray([p["native_gap"] for p in profiles], dtype=np.float32)] if native_points
                                   else [p["grid"] for p in profiles])).astype(np.float64)
    mass = np.zeros((profiles[0]["mass"].shape[0], len(grid)), dtype=np.float64)
    native = np.zeros(len(grid), dtype=np.float64)
    for row in profiles:
        positions = np.searchsorted(grid, row["grid"])
        mass[:, positions] += row["mass"] / len(profiles)
        if native_points:
            native[np.searchsorted(grid, row["native_gap"])] += 1 / len(profiles)
        else:
            native[positions] += row["native_mass"] / len(profiles)
    if not np.allclose(mass.sum(axis=1), 1, atol=2e-6) or not np.isclose(native.sum(), 1):
        raise ValueError("non-normalized empirical distributions")
    return grid, mass, native


@torch.inference_mode()
def scan(engine, item, baseline, candidates):
    engine.reset()
    encoded = engine.tokenizer(make_prompt(engine.tokenizer, item["problem"]), return_tensors="pt").to(engine.model.device)
    mask = encoded["attention_mask"]
    out = engine.forward(encoded["input_ids"], mask)
    cache = owned_cache(out.past_key_values)
    think = bool((encoded["input_ids"] == engine.start).any().item())
    events, positions = [], []
    for pos, token in enumerate(baseline["token_ids"]):
        if token == engine.eos or pos == len(baseline["token_ids"]) - 1:
            break
        if token == engine.start:
            think = True
        elif token == engine.end:
            think = False
        mask = engine.extend(mask)
        query = engine.query(token)
        boundary = think and token in engine.splits
        donor = engine.shadow(query, mask, cache, native=True) if boundary else None
        out = engine.forward(query, mask, cache)
        cache = owned_cache(out.past_key_values)
        if boundary:
            events.append(compress_event(out.logits[0, -1], donor.logits[0, -1], candidates))
            positions.append(pos)
        del donor, out
    if not events:
        raise ValueError(f"no boundary for calibration case {item['index']}; do not silently drop it")
    grid, mass, native = merge_profiles(events, native_points=True)
    return dict(grid=grid, mass=mass, native_mass=native, events=len(events), positions=positions,
                native_flip=float(np.mean([e["native_flip"] for e in events])),
                sample_flip=np.mean([e["sample_flip"] for e in events], axis=0))


def finish(profiles, candidates):
    grid, mass, native = merge_profiles(profiles)
    fv = np.cumsum(native)
    fs = np.cumsum(mass, axis=1)
    distances = (np.abs(fs[:, :-1] - fv[None, :-1]) * np.diff(grid)).sum(axis=1)
    rows = []
    for n, (t, p) in enumerate(candidates):
        rows.append(dict(temperature=t, top_p=p, wasserstein=float(distances[n]),
                         positive_deficit_rate=float(1 - mass[n, 0]),
                         token_change_rate=float(np.mean([r["sample_flip"][n] for r in profiles]))))
    selected = min(rows, key=lambda r: (r["wasserstein"], r["temperature"], r["top_p"]))
    position = candidates.index((selected["temperature"], selected["top_p"]))
    return dict(selected=selected, all_candidates=rows,
                native_positive_deficit_rate=float(1-native[0]),
                native_token_change_rate=float(np.mean([r["native_flip"] for r in profiles])),
                cases=len(profiles), events=int(sum(r["events"] for r in profiles)),
                selection_uses_correctness=False, kernel="direct_top_p",
                weighting="equal problem, equal boundary within problem",
                objective="exact discrete W1 integral; native token-level FP32 nucleus probabilities"), (grid, fv, fs[position])
