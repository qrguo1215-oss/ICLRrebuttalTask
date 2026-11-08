"""Departure gating is independent of the conditional depth distribution."""
import torch

from bridge import ROOT
from native import nucleus
from protocol import seeded


def depth_distribution(logits, kernel, factor=1.):
    z, ids, reference = nucleus(logits, kernel["temperature"], kernel["top_p"])
    greedy = int(logits.float().argmax().item())
    not_greedy = ids != greedy
    support = (reference > 0) & not_greedy
    rate = float(reference[not_greedy].sum().item())
    if not bool(support.any()):
        return ids, torch.zeros_like(reference), 0., greedy
    # All depths have IDENTICAL support. Only conditional weights change.
    weights = torch.softmax((z / (kernel["temperature"] * factor)).masked_fill(~support, -torch.inf), -1)
    return ids, weights, rate, greedy


def draw_depth(logits, kernel, factor, rng, gate_value, *, force=False):
    ids, probabilities, rate, greedy = depth_distribution(logits, kernel, factor)
    available = rate > 0
    active = available and (force or gate_value < rate)
    selected = int(ids[torch.multinomial(probabilities, 1, generator=rng)].item()) if active else greedy
    z = logits.float()
    return selected, dict(gate_probability=rate, gate_value=gate_value, conditional_available=available,
                          attempted=active, departure=selected != greedy,
                          deficit=float((z.max() - z[selected]).item()),
                          conditional_factor=factor, forced_departure=force)


def gate_uniform(config, index, seed, ordinal):
    return seeded("depth-gate", config, index, seed, ordinal) / float(2**31 - 1)


def direction(vector, config_id, number):
    generator = torch.Generator(device="cpu").manual_seed(seeded("random-direction", config_id, number))
    random = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    random *= vector.float().cpu().norm() / random.norm().clamp_min(1e-12)
    return random.to(device=vector.device, dtype=vector.dtype)


def deficit_w1(x, y):
    # Equal-size samples: exact empirical W1, no histogram approximation.
    if len(x) != len(y) or not x:
        raise ValueError("Paired nonempty deficit samples required")
    return sum(abs(a-b) for a, b in zip(sorted(x), sorted(y))) / len(x)
