import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge
from protocol import plan, key, job, settings, splits, checked, result_path, seeded
from spec import write, digest, data
from kernels import depth_distribution, draw_depth, gate_uniform, direction, deficit_w1
from causal_engine import CausalEngine
from native import nucleus
from causal_stats import interval, mcnemar, headroom_null, wilson
from analysis import validate_micro
from report import gold_answer, evaluate
from counterparts import prepare, score


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    bos_token = None
    def apply_chat_template(self, *args, **kwargs): return "prompt"
    def __call__(self, *args, **kwargs):
        return Batch(input_ids=torch.tensor([[6, 1]]), attention_mask=torch.ones(1, 2, dtype=torch.long))
    def decode(self, tokens, **kwargs): return " ".join(map(str, tokens))


class ToyEngine(CausalEngine):
    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.tokenizer = Tokenizer()
        self.config = dict(id="math15_cps", method="cps", dataset="math")
        self.vector, self.zero = torch.tensor([1.]), torch.tensor([0.])
        self.splits, self.start, self.end, self.eos = {9}, 6, 8, 7
    def reset(self): pass
    def forward(self, ids, mask, cache=None, native=False, **kwargs):
        memory = 0 if cache is None else int(cache[0][0].item())
        token = int(ids[0, -1])
        chosen = 9 if cache is None else 3 if token == 9 and native else 2 if token == 9 else 4 if token in (2, 3) and memory else 5 if token in (2, 3) else 7
        memory = 1 if token == 9 and native else memory
        z = torch.arange(10, dtype=torch.float32).reshape(1, 1, -1)*.01
        z[0, 0, chosen] = 4.
        return SimpleNamespace(logits=z, past_key_values=((torch.tensor(memory), torch.tensor(memory)),))


class CausalTests(unittest.TestCase):
    def test_plan_separation_and_seed_independence(self):
        jobs = plan(["math15_cps"], ["occlusion", "depth", "replay"])
        self.assertEqual(len(jobs), len(set(map(key, jobs))))
        for j in jobs:
            if j["condition"] != "G":
                self.assertIn(j["index"], splits("math")["test"])
        eval_seeds = {j["seed"] for j in jobs if j["condition"] == "whole_boundary_q50"}
        discovery = {j["seed"] for j in jobs if j["condition"] == "discover_boundary_q50"}
        self.assertEqual(len(eval_seeds), 16)
        self.assertFalse(eval_seeds & discovery)
        for j in plan(["math15_cps"], ["depth"], True):
            self.assertIn(j["index"], splits("math")["calibration"])
        with self.assertRaises(ValueError): plan(["math7_cps"], ["depth"])

    def test_fixed_replay_200(self):
        jobs = plan(["math15_seal", "math15_cps", "math15_rebalance"], ["replay"])
        for cfg in ("math15_seal", "math15_cps", "math15_rebalance"):
            self.assertEqual({j["index"] for j in jobs if j["config_id"] == cfg and j["condition"] == "thought_replay"}, set(splits("math")["replay"]))

    def test_matched_reconstructs_reference(self):
        z, kernel = torch.tensor([4., 3.9, 3., 2., -5.]), dict(temperature=.8, top_p=.95)
        ids, cond, rate, greedy = depth_distribution(z, kernel)
        _, refids, ref = nucleus(z, .8, .95)
        self.assertTrue(torch.equal(ids, refids))
        mixture = cond*rate
        mixture[ids == greedy] += 1-rate
        torch.testing.assert_close(mixture, ref)

    def test_depth_same_support_rate_monotone_mean(self):
        z, kernel = torch.tensor([4., 3.9, 3., 2., -5.]), dict(temperature=.8, top_p=.99)
        curves = [depth_distribution(z, kernel, f) for f in (.5, 1, 2)]
        self.assertEqual(len({c[2] for c in curves}), 1)
        for c in curves:
            self.assertTrue(torch.equal(c[1] > 0, curves[0][1] > 0))
        means = [float(((z.max()-z[ids])*p).sum()) for ids, p, _, _ in curves]
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])

    def test_zero_gap_can_be_real_departure(self):
        z = torch.tensor([4., 4., -100.])
        token, detail = draw_depth(z, dict(temperature=1., top_p=1.), 1., torch.Generator().manual_seed(1), .1, force=True)
        self.assertEqual(token, 1)
        self.assertTrue(detail["departure"])
        self.assertEqual(detail["deficit"], 0)

    def test_single_support_never_forces_impossible_action(self):
        token, detail = draw_depth(torch.tensor([100., -100.]), dict(temperature=.1, top_p=.1), 2., torch.Generator(), 0, force=True)
        self.assertEqual(token, 0)
        self.assertFalse(detail["conditional_available"])

    def test_random_directions_fixed_norm_and_seed(self):
        v = torch.ones(1536)
        a, b = direction(v, "math15_cps", 0), direction(v, "math15_cps", 1)
        torch.testing.assert_close(a.norm(), v.norm())
        self.assertTrue(torch.equal(a, direction(v, "math15_cps", 0)))
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(gate_uniform("a", 3, 4, 5), gate_uniform("a", 3, 4, 5))

    def test_cache_only_is_not_forced_baseline_text(self):
        e = ToyEngine()
        g = e.run(dict(index=0, problem="p"), "G", max_tokens=20)
        cache = e.run(dict(index=0, problem="p"), "cache_only", max_tokens=20)
        sham = e.run(dict(index=0, problem="p"), "self_cache", max_tokens=20)
        self.assertEqual(g["token_ids"], [9, 2, 5, 7])
        self.assertEqual(cache["token_ids"], [9, 2, 4, 7])
        self.assertEqual(sham["token_ids"], g["token_ids"])

    def test_whole_and_adjacent_sample_at_different_positions(self):
        e = ToyEngine()
        cal = dict(thresholds={}, thresholds_adjacent={})
        with patch("causal_engine.sample_top_p", side_effect=lambda z, *args: int(z.argmax())):
            boundary = e.run(dict(index=0, problem="p"), "whole_boundary_all", max_tokens=20, calibration=cal)
            adjacent = e.run(dict(index=0, problem="p"), "whole_adjacent_all", max_tokens=20, calibration=cal)
        self.assertEqual([r["position"] for r in boundary["events"] if r["sampled"]], [1])
        self.assertEqual([r["position"] for r in adjacent["events"] if r["sampled"]], [2])
        self.assertEqual(boundary["sampled_tokens"], 3)
        self.assertEqual(adjacent["sampled_tokens"], 2)

    def test_official_sampling_includes_first_generated_token(self):
        e = ToyEngine()
        with patch("causal_engine.sample_top_p", return_value=e.eos) as sampler:
            row = e.run(dict(index=0, problem="p"), "official_full", max_tokens=20)
        self.assertEqual(row["token_ids"], [e.eos])
        self.assertEqual(row["sampled_tokens"], 1)
        self.assertEqual(sampler.call_count, 1)

    def test_combined_sampling_keeps_native_cache(self):
        e = ToyEngine()
        with patch("causal_engine.sample_top_p", side_effect=lambda z, *args: int(z.argmax())):
            row = e.run(dict(index=0, problem="p"), "wholeV_boundary_all", max_tokens=20,
                        calibration=dict(thresholds={}, thresholds_adjacent={}))
        self.assertEqual(row["token_ids"], [9, 3, 4, 7])
        self.assertEqual(row["cache_policy"], "native")
        self.assertTrue(row["events"][0]["native_cache_committed"])

    def test_micro_exact_one_and_global_cap(self):
        e = ToyEngine()
        cal = dict(direct=dict(selected=dict(temperature=1., top_p=1.)))
        base = e.run(dict(index=0, problem="p"), "G", max_tokens=20)
        rows = {}
        for c in ("micro_matched", "micro_shallow", "micro_deep", "micro_greedy"):
            r = e.run(dict(index=0, problem="p"), c, max_tokens=4, calibration=cal, baseline=base, smoke=True)
            r.update(config_id="math15_cps")
            rows[c] = r
            self.assertLessEqual(len(r["token_ids"]), 4)
        validate_micro(rows)
        missing = e.run(dict(index=0, problem="p"), "micro_matched", max_tokens=20, calibration=cal, baseline=base, smoke=False)
        self.assertEqual(missing["status"], "unavailable")

    def test_micro_audit_rejects_different_prefix(self):
        rows = {n: dict(config_id="x", index=1, condition="micro_matched", frozen_prefix_sha256=str(n),
                        events=[dict(forced_departure=True, departure=True)]) for n in range(2)}
        with self.assertRaises(ValueError): validate_micro(rows)

    def test_formal_micro_boundary_ordinal_and_think_mask(self):
        e = ToyEngine()
        tokens = [9, 2]*10
        self.assertEqual(e.frozen_prefix(dict(token_ids=tokens), False), tokens[:15])
        self.assertIsNone(e.frozen_prefix(dict(token_ids=[9, 2, e.end]+tokens), False))

    def test_statistics_not_equivalence(self):
        out = interval([-.1, .1], repeats=100)
        self.assertTrue(out["crosses_zero"])
        self.assertFalse(out["equivalence_test"])
        self.assertAlmostEqual(mcnemar([0, 0, 0], [1, 1, 1])["exact_two_sided_p"], .25)
        self.assertAlmostEqual(deficit_w1([0, 3], [1, 2]), 1.)
        self.assertGreater(wilson(0, 8)[1], .3)
        self.assertLess(wilson(8, 8)[0], .7)

    def test_headroom_crossfit_with_independent_discovery(self):
        rng = np.random.default_rng(10)
        q = np.linspace(.05, .95, 100)
        x = rng.binomial(1, q[:, None], (100, 100))
        y = np.where(x, 1-rng.binomial(1, .1, x.shape), rng.binomial(1, .25, x.shape))
        g, v = [0]*40+[1]*60, [1]*10+[0]*30+[0]*6+[1]*54
        result = headroom_null(q, x, y, np.arange(100)%5, g, v, repeats=100)
        self.assertEqual(result["status"], "ok")
        self.assertLess(abs(result["residual_mean"]), .05)
        self.assertEqual(len(result["coupling_sensitivity"]), 4)

    def test_headroom_does_not_fit_away_the_tested_effect(self):
        q = np.full(100, .5)
        x = np.tile([0., 1.], (100, 4))
        g, v = [0]*50+[1]*50, [1]*100
        result = headroom_null(q, x, x.copy(), np.arange(100)%5, g, v, repeats=100)
        self.assertLess(result["residual_mean"], -.45)
        self.assertFalse(result["slope_identifiable"])
        subset = headroom_null(q, x, x.copy(), np.arange(100)%5, g, v, mask=np.array(g) == 0, repeats=100)
        self.assertEqual(subset["fit_population_n"], 100)
        self.assertEqual(subset["n"], 50)

    def test_gold_and_grading_pipeline(self):
        self.assertTrue(all(gold_answer(item) for item in data("math").values()))
        result = evaluate(dict(text=r"\boxed{42}", response_text=r"\boxed{42}"), "42")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["correct"])
        textual = evaluate(dict(text=r"\boxed{\text{east}}", response_text=r"\boxed{\text{east}}"), "east")
        self.assertTrue(textual["strict_correct"])

    def test_result_rejects_stale_and_debug_records(self):
        with tempfile.TemporaryDirectory() as d:
            j = job("math15_cps", 1, "G")
            p = result_path(Path(d), j)
            meta = dict(run_id="a", smoke=False, runtime={})
            write(p, dict(j, run_id="a", smoke=False, runtime={}, max_new_tokens=256, dtype="bfloat16"))
            with self.assertRaises(ValueError): checked(p, meta, j)

    def test_blind_counterpart_equal_pool_no_leakage(self):
        raw, graded = {}, {}
        for c, ss in (("G", [0]), ("V", [0]), ("official_full", range(8))):
            for s in ss:
                k = ("math15_cps", 0, c, s)
                raw[k] = dict(response_text=f"solution {c} {s}")
                graded[k] = dict(correct=c != "G", strict_correct=c != "G")
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            prepare(output, dict(run_id="test"), raw, graded)
            packet = json.loads((output/"counterparts/blind_packet.json").read_text())
            allocation = json.loads((output/"counterparts/private_allocation_DO_NOT_SEND_TO_JUDGE.json").read_text())
            self.assertEqual(len(packet["entries"]), 2)
            self.assertEqual(packet["entries"][0]["candidates"], packet["entries"][1]["candidates"])
            for a in allocation:
                self.assertEqual(len(a["target_seeds"]), 2)
                self.assertNotIn(a["query_seed"], a["target_seeds"])
            labels = output/"labels.json"
            write(labels, [dict(id=p["id"], matches=[True, False], rationale="same reduction") for p in packet["entries"]])
            with patch("analysis.bundle"):
                score(output, labels, "test fixture, not actual judgments")
            verdict = json.loads((output/"counterparts/scored.json").read_text())
            self.assertEqual(verdict["estimate"]["mean"], 0)

    def test_calibration_is_label_free_and_equal_problem(self):
        from frozen_scan import freeze
        p = dict(grid=np.array([0., 1.]), mass=np.array([[.8, .2]]), native_mass=np.array([.8, .2]),
                 events=2, margins=np.array([1., 3.]), adjacent_margins=np.array([.2, .6]),
                 native_flip=.2, sample_flip=np.array([.2]), random_numbers=np.array([]))
        cal, cdf = freeze([p, p], [(1., .95)])
        self.assertFalse(cal["selection_uses_correctness"])
        self.assertEqual(cal["direct"]["selected"]["wasserstein"], 0)
        self.assertEqual(cal["thresholds"]["q25"], 1.)
        self.assertEqual(cal["thresholds_adjacent"]["q25"], .2)


if __name__ == "__main__":
    unittest.main()
