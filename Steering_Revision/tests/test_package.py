import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spec import verify_assets, configurations, splits, make_job, path_for, digest, write, read, checked_result, signature
from task import plan_jobs, reject_model_overrides
from engine import Engine
from native import nucleus, generation_seed
from calibration import compress_event, merge_profiles, finish, candidate_grid
from report import evaluate, paired_interval, make_reports
import report


class FakeEngine:
    splits, end, eos = {9}, 8, 7
    def __init__(self, length, stop):
        self.length, self.stop, self.steps = length, stop, 0
    def query(self, token): return token
    def extend(self, mask): return mask
    def output(self):
        logits = torch.zeros(1, 1, 10)
        logits[0, 0, self.stop if self.steps == self.length-1 else 1] = 1
        return SimpleNamespace(logits=logits, past_key_values=())
    def forward(self, *args, **kwargs):
        self.steps += 1
        return self.output()


class PackageTests(unittest.TestCase):
    def test_assets(self):
        self.assertGreater(len(verify_assets()), 20)

    def test_no_duplicate_work(self):
        jobs = plan_jobs(list(configurations()), ["replay", "direct", "gsm-whole"])
        self.assertEqual(len(jobs), 90000)
        self.assertEqual(len({digest(j) for j in jobs}), len(jobs))
        self.assertFalse(any("7_cps" in j["config_id"] for j in jobs))
        self.assertEqual(path_for("out", "math15_cps", 1, "G"), path_for("out", "math15_seal", 1, "G"))
        self.assertNotEqual(path_for("out", "math15_rebalance", 1, "G"), path_for("out", "math15_seal", 1, "G"))

    def test_splits(self):
        for ds in ("math", "gsm"):
            s = splits(ds)
            self.assertEqual((len(s["calibration"]), len(s["test"])), (100, 400))
            self.assertFalse(set(s["calibration"]) & set(s["test"]))
        self.assertEqual(len(splits("math")["replay"]), 200)

    def test_full_thought_and_stops(self):
        for length, stop_token, cap, expected in ((2100, 9, 3000, (2100, "split")),
                                                (4, 8, 100, (4, "think_end")),
                                                (4, 7, 100, (4, "eos")),
                                                (2100, 9, 100, (100, "global_budget"))):
            fake = FakeEngine(length, stop_token)
            tokens, reason = Engine.propose_thought(fake, fake.output(), None, cap)
            self.assertEqual((len(tokens), reason), expected)

    def test_reset(self):
        fake = object.__new__(Engine)
        fake.model = SimpleNamespace(_prev_step_mean=torch.tensor([.9]))
        fake.tokenizer, fake.vector = None, None
        fake.config = dict(method="rebalance", layer=19, coefficient=-1)
        with patch("engine.configure_steering"):
            fake.reset()
        self.assertIsNone(fake.model._prev_step_mean)

    def test_native_kernel_ties_and_cdf(self):
        z = torch.tensor([3., 3., 2., -5.])
        v = torch.tensor([2., 4., 1., -5.])
        grid = [(.2, .01), (1., .5), (1.4, .95), (2., 1.)]
        event = compress_event(z, v, grid)
        self.assertEqual(event["native_gap"], 0)
        self.assertEqual(event["native_flip"], 1)
        for i, (t, p) in enumerate(grid):
            _, ids, probs = nucleus(z, t, p)
            deficits = (z.max() - z[ids]).numpy()
            ref = np.array([float(probs[torch.tensor(deficits == d)].sum()) for d in event["grid"]])
            np.testing.assert_allclose(event["mass"][i], ref, atol=1e-7)
        x, mass, native = merge_profiles([event], native_points=True)
        profile = dict(grid=x, mass=mass, native_mass=native, events=1, native_flip=1., sample_flip=event["sample_flip"])
        result, cdf = finish([profile], grid)
        json.dumps(result)
        self.assertAlmostEqual(cdf[1][-1], 1)
        self.assertAlmostEqual(cdf[2][-1], 1, places=6)

    def test_deficit_coordinates_promote_before_subtraction(self):
        for dtype in (torch.float16, torch.bfloat16):
            base = torch.tensor([2., -.0009765625], dtype=dtype)
            target = torch.tensor([0., 3.], dtype=dtype)
            event = compress_event(base, target, [(1., 1.)])
            self.assertEqual(event["native_gap"], float(base.float()[0]-base.float()[1]))
            self.assertIn(event["native_gap"], event["grid"])

    def test_reject_unverified_model_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"model-00001-of-000002.safetensors").touch()
            known = {"model-00001-of-000002.safetensors":{}}
            reject_model_overrides(p, known)
            (p/"model.safetensors").touch()
            with self.assertRaises(ValueError): reject_model_overrides(p, known)

    @unittest.skipUnless(sys.platform == "linux", "Linux parent-death protection")
    def test_worker_exits_when_controller_is_killed(self):
        child_code = "from task import parent_death_guard; import time; parent_death_guard(); print('ready',flush=True); time.sleep(30)"
        parent_code = ("import subprocess,sys,os,time; "
                       f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}], "
                       "env=dict(os.environ,REVISION_PARENT_PID=str(os.getpid())),stdout=subprocess.PIPE,text=True); "
                       "assert p.stdout.readline().strip()=='ready'; print(p.pid,flush=True); time.sleep(30)")
        p = subprocess.Popen([sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True)
        child_pid = None
        try:
            child_pid = int(p.stdout.readline().strip())
            p.kill()
            p.wait(timeout=5)
            for _ in range(50):
                status = Path(f"/proc/{child_pid}/stat")
                if not status.exists() or status.read_text().split()[2] == "Z":
                    break
                time.sleep(.1)
            else:
                self.fail("worker survived controller SIGKILL")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()
            if child_pid:
                try: os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError: pass

    def test_resume_identity_and_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"row.json.gz"
            row = dict(make_job("math15_seal", 1, "G"), run_id="x", smoke=False, max_new_tokens=10000)
            write(p, row)
            self.assertEqual(checked_result(p, "x"), row)
            with self.assertRaises(ValueError): checked_result(p, "y")
            write(p, dict(row, smoke=True))
            with self.assertRaises(ValueError): checked_result(p, "x")

    def test_seed_schedule(self):
        cfg = configurations()["math15_cps"]
        self.assertNotEqual(generation_seed(cfg, 3, 0), generation_seed(cfg, 3, 1))
        self.assertNotEqual(generation_seed(cfg, 3, 0), generation_seed(cfg, 3, 0, True))

    def test_grading_sensitivity_is_explicit(self):
        row = dict(text=r"The answer is \boxed{1}.", response_text=r"The answer is \boxed{1}.")
        result = evaluate(row, "100")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["correct"])
        self.assertFalse(result["strict_correct"])

    def test_negative_point_is_not_significance(self):
        result = paired_interval([-.2, .1])
        self.assertTrue(result["negative_point"])
        self.assertFalse(result["ci_strictly_negative"])

    def test_reports_do_not_substitute_shared_sampling(self):
        # Small synthetic unit fixture, never written into an experiment directory.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            configs = ["math15_seal", "math15_cps"]
            graded = {}
            for cfg in configs:
                write(out/"calibration"/f"{cfg}.json", dict(indices=[2], direct=dict(all_candidates=[{}],
                    selected=dict(temperature=1., top_p=.5))))
                for c in ("G", "V", "S", "SV", "token_replay", "thought_replay"):
                    group = graded.setdefault(str(path_for(out, cfg, 0, c).parent), {})
                    for i in (0, 1):
                        correct = c != "G" and (c != "S" or cfg == "math15_cps")
                        group[i] = [dict(index=i, seed=seed, correct=correct, strict_correct=correct,
                                         token_count=10, hit_max=False, closed=True)
                                    for seed in (range(2) if c in ("S", "SV") else [0])]
            split = dict(test=[0, 1], replay=[0, 1], calibration=[2])
            with patch("report.splits", return_value=split):
                make_reports(out, dict(configs=configs, scopes=["replay", "direct"], run_id="unit"), graded)
            report = read(out/"summary/summary.json")
            sampling = {r["config"]:r["accuracy"] for r in report["tables"]["table6"] if r["condition"] == "S"}
            self.assertEqual(sampling, {"math15_seal":0., "math15_cps":100.})
            self.assertEqual(len(report["tables"]["table2"]), 8)

    def test_cpu_grade_report_and_bundle(self):
        # End-to-end packaging fixture. No GPU generation or scientific result.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"unit"
            cfgid = "math15_seal"
            cfg = configurations()[cfgid]
            cal = dict(indices=[2], run_id="unit", smoke=False, thresholds={},
                       direct=dict(all_candidates=[{}], selected=dict(temperature=1., top_p=.9)))
            write(out/"calibration"/f"{cfgid}.json", cal)
            meta = dict(run_id="unit", smoke=False, code_signature=signature(),
                        configs=[cfgid], scopes=["direct"], runtime={})
            write(out/"RUN.json", meta)
            jobs = []
            for i in (0, 1):
                for c in ("G", "V", "S", "SV"):
                    job = make_job(cfgid, i, c)
                    jobs.append(job)
                    sampling = c in ("S", "SV")
                    row = dict(job, run_id="unit", smoke=False, max_new_tokens=10000,
                               runtime={}, dtype=cfg["dtype"], text=r"\boxed{2}", response_text=r"\boxed{2}",
                               token_count=1, hit_max=False, closed=True,
                               rng_seed=generation_seed(cfg, i, 0, True) if sampling else None,
                               calibration_sha256=digest(cal) if sampling else None,
                               kernel=cal["direct"]["selected"] if sampling else None,
                               cache_policy="native" if c in ("V", "SV") else "unsteered")
                    write(path_for(out, **job), row)
            write(out/"EXPECTED.json", jobs)
            with patch("task.plan_jobs", return_value=jobs), patch("report.splits", return_value=dict(test=[0, 1])), \
                 patch("report.data", return_value={0:dict(answer="2"), 1:dict(answer="2")}), \
                 patch.object(sys, "argv", ["report.py", "--output", str(out), "--cpu-workers", "1"]):
                report.main()
            self.assertTrue((Path(d)/"unit_results.tar.gz").is_file())
            self.assertEqual(read(out/"grading_errors.json"), [])
            summary = read(out/"summary/summary.json")
            self.assertTrue(all(r["accuracy"] == 100. for r in summary["tables"]["table6"]))


if __name__ == "__main__":
    unittest.main()
