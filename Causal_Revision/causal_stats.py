"""CPU statistics with problem-level units, explicit nulls and no outcome tuning."""
import math
import numpy as np


def wilson(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    p, z = successes/n, 1.959963984540054
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0., center-radius), min(1., center+radius)


def interval(values, repeats=10000, seed=20261108):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Nonempty finite problem-level values required")
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for start in range(0, repeats, 200):
        n = min(200, repeats-start)
        boot[start:start+n] = x[rng.integers(len(x), size=(n, len(x)))].mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi), n=len(x),
                crosses_zero=bool(lo <= 0 <= hi), equivalence_test=False)


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    rescue, harm = int((~a & b).sum()), int((a & ~b).sum())
    n, k = rescue+harm, min(rescue, harm)
    # log-sum avoids overflow for larger future datasets.
    p = min(1., 2*sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)-n*math.log(2)) for j in range(k+1))) if n else 1.
    return dict(rescue_count=rescue, harm_count=harm, exact_two_sided_p=p)


def slope(q, y):
    q, y = np.asarray(q), np.asarray(y)
    return float(np.dot(q-q.mean(), y-y.mean())/np.dot(q-q.mean(), q-q.mean())) if np.var(q) > 0 else 0.


def headroom_null(q, clean, steered, folds, reference_g, reference_v, mask=None, repeats=10000):
    """Cross-fitted r(1-q)-h*q; q comes from disjoint discovery seeds.

    Fit rescue/harm on other problems' G/V outcomes, NOT the S/SV effect being
    explained. Transport those rates to discovery-q under a restricted null.
    Bootstrap refits rates; a subgroup retains the full training population.
    """
    q = np.asarray(q, float)
    clean, steered = np.asarray(clean, float), np.asarray(steered, float)
    folds = np.asarray(folds)
    reference_g, reference_v = np.asarray(reference_g, float), np.asarray(reference_v, float)
    mask = np.ones(len(q), bool) if mask is None else np.asarray(mask, bool)
    if clean.shape != steered.shape or clean.ndim != 2 or len(q) != len(clean):
        raise ValueError("Expected paired item x seed arrays")
    if reference_g.shape != q.shape or reference_v.shape != q.shape or mask.shape != q.shape or not mask.any():
        raise ValueError("Require full-population G/V reference and a nonempty evaluation mask")
    if len(set(folds)) < 2:
        return dict(status="insufficient_folds", n=int(mask.sum()))

    def compute(ids, shift=0):
        x, y = clean[ids], np.roll(steered[ids], shift, axis=1)
        g, v, selected = reference_g[ids], reference_v[ids], mask[ids]
        qs, fs = q[ids], folds[ids]
        predicted = np.zeros(len(ids))
        rates = []
        for fold in sorted(set(fs)):
            train, test = fs != fold, fs == fold
            if not train.any():
                return None
            # Jeffreys smoothing handles zero failures/successes without division by zero.
            r = ((1-g[train])*v[train]).sum()+.5
            r /= (1-g[train]).sum()+1
            h = (g[train]*(1-v[train])).sum()+.5
            h /= g[train].sum()+1
            predicted[test] = r*(1-qs[test])-h*qs[test]
            rates.append(dict(fold=int(fold), rescue=float(r), harm=float(h),
                              training_G_wrong=int((1-g[train]).sum()), training_G_correct=int(g[train].sum())))
        observed = (y-x).mean(axis=1)
        residual = observed-predicted
        return float(residual[selected].mean()), slope(qs[selected], residual[selected]), observed[selected], predicted[selected], rates

    ids = np.arange(len(q))
    point = compute(ids)
    rng = np.random.default_rng(20261108)
    strata = [ids[(folds == f) & (mask == selected)] for f in sorted(set(folds)) for selected in (False, True)
              if ((folds == f) & (mask == selected)).any()]
    boot = []
    for _ in range(repeats):
        resampled = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        boot.append(compute(resampled)[:2])
    ci = np.quantile(np.asarray(boot), [.025, .975], axis=0)
    coupling = [dict(seed_shift=k, residual_mean=compute(ids, k)[0], residual_slope=compute(ids, k)[1],
                     paired_rescue_joint=float(((1-clean[mask])*np.roll(steered[mask], k, axis=1)).mean()),
                     paired_harm_joint=float((clean[mask]*(1-np.roll(steered[mask], k, axis=1))).mean()))
                for k in range(min(4, clean.shape[1]))]
    return dict(status="ok", n=int(mask.sum()), fit_population_n=len(q), rate_source="held-out-fold native G/V only",
                residual_mean=point[0], residual_mean_ci=ci[:, 0].tolist(),
                residual_slope=point[1], residual_slope_ci=ci[:, 1].tolist(),
                slope_identifiable=bool(np.var(q[mask]) > 0),
                observed_delta=point[2].tolist(), predicted_delta=point[3].tolist(), rates=point[4],
                coupling_sensitivity=coupling, discovery_sampling_noise="not corrected; only eight independent seeds",
                inference_scope="constant rescue/harm null only; not all possible ceiling or distribution-shift explanations")
