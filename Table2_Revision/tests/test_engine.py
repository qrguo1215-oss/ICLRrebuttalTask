"""CPU-only engine contract tests; no model checkpoints or CUDA are needed."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, mock_open, patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_local(name):
    spec = importlib.util.spec_from_file_location("table2_test_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keep these unit tests independent of the parent's task/spec and of other
# experiment modules with the same names in a combined test-discovery process.
spec_stub = ModuleType("spec")
spec_stub.ROOT = ROOT
spec_stub.generation_seed = Mock(return_value=1234)
spec_stub.read = Mock()
with patch.dict(sys.modules, {"spec": spec_stub}):
    native = load_local("native")
    grading = load_local("grading")
    with patch.dict(sys.modules, {"native": native}):
        engine = load_local("engine")

protocol = load_local("spec")
with patch.dict(sys.modules, {"spec": protocol}):
    task = load_local("task")
    with patch.dict(sys.modules, {"task": task}):
        report = load_local("report")


START, END, SPLIT, EOS, A, B, POISON = range(1, 8)
ITEM = dict(index=17, problem="What is one plus one?")


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    eos_token_id, bos_token = EOS, "<bos>"

    def encode(self, text, add_special_tokens=False):
        if text == "<think>":
            return [START]
        if text == "</think>":
            return [END]
        return [A, SPLIT, B, SPLIT]

    def get_vocab(self):
        return {"\u010a\u010a": SPLIT, "plain": A}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return self.bos_token + messages[0]["content"] + "<think>"

    def __call__(self, prompt, return_tensors):
        return Batch(input_ids=torch.tensor([[0, START]]), attention_mask=torch.ones(1, 2, dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=False):
        words = {START: "<think>", END: "</think>", SPLIT: "\n\n", EOS: "<eos>",
                 A: "a", B: "b", POISON: "polluted"}
        return "".join(words[t] for t in tokens if not skip_special_tokens or t not in (START, END, EOS))


class Cache:
    def __init__(self, legacy):
        self.legacy = legacy

    def to_legacy_cache(self):
        return self.legacy


class Model:
    """Greedy scripted CPU model with mutable controller and donor-cache poison."""
    device = torch.device("cpu")
    config = SimpleNamespace(hidden_size=3584)

    def __init__(self, script=None, dynamic_cache=True):
        self.script = script if script is not None else [SPLIT, A, SPLIT, B, END, A, EOS]
        self.dynamic_cache = dynamic_cache
        self.calls = []
        self._prev_step_mean = torch.tensor([999.])

    def set_steering_flag(self, steering_flag, steering_layer, steer_vec, steer_coef, tokenizer, **kwargs):
        self.steering_flag, self.steering_vector = steering_flag, steer_vec
        self.steering_coef, self.steering_layer = steer_coef, steering_layer
        self.dyn_hparams = kwargs.get("dyn_hparams")
        for field in ("_coefs", "_step_prob_sum", "_step_tok_count", "_last_maxprob"):
            setattr(self, field, torch.zeros(1))

    def start_new_round(self):
        self.new_round, self.cur_steps, self.steering_think_flag = True, 0, None
        self.calls.clear()

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache, return_dict):
        ids = input_ids[0].tolist()
        native_call = self.steering_flag and bool(self.steering_vector.any())
        if past_key_values is None:
            past_key_values = ((torch.tensor([0]), torch.tensor([0.])),)
        length, poison = past_key_values[0]
        seen = int(length.item())
        assert attention_mask.shape[1] == seen + len(ids), "a token was forwarded twice or omitted"
        self.calls.append(dict(ids=ids, native=native_call, seen=seen,
                               poison=float(poison.item()), steps=self.cur_steps))
        if native_call:
            # Deliberately mutate the supplied tensors: a shallow donor copy fails.
            poison.add_(100.)
        increment = 10 if native_call else 1
        for field in ("_coefs", "_step_prob_sum", "_step_tok_count", "_last_maxprob"):
            getattr(self, field).add_(increment)
        if self._prev_step_mean is None:
            self._prev_step_mean = torch.zeros(1)
        self._prev_step_mean.add_(increment)
        self.cur_steps += 1
        self.new_round = False
        self.steering_think_flag = torch.tensor([END not in ids])
        position = seen + len(ids) - 2
        chosen = self.script[position] if position < len(self.script) else A
        if native_call and ids[-1] == SPLIT and chosen == A:
            chosen = B
        elif not native_call and poison.item():
            chosen = POISON
        logits = torch.zeros(1, 1, 8)
        logits[0, 0, chosen] = 5
        legacy = ((torch.tensor([seen + len(ids)]), poison.clone()),)
        cache = Cache(legacy) if self.dynamic_cache else legacy
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_engine(vector=1., script=None, dynamic_cache=True, method="seal"):
    config = dict(method=method, layer=20, coefficient=1. if method == "cps" else -1.,
                  size="7", dataset="math", dtype="bfloat16", vector="assets/vectors/7_cps.pt")
    return engine.Engine(Model(script, dynamic_cache), Tokenizer(), torch.tensor([vector]), config)


class Proposal:
    splits, end, eos = {SPLIT}, END, EOS

    def __init__(self, length, stop):
        self.length, self.stop, self.steps = length, stop, 0
        self.native_calls = []

    def query(self, token):
        return token

    def extend(self, mask):
        return mask

    def output(self):
        logits = torch.zeros(1, 1, 8)
        logits[0, 0, self.stop if self.steps == self.length - 1 else A] = 1
        return SimpleNamespace(logits=logits, past_key_values=())

    def forward(self, *args, native):
        self.native_calls.append(native)
        self.steps += 1
        return self.output()


def no_result(connection, *args):
    connection.close()


def slow_result(connection, *args):
    time.sleep(2)
    connection.close()


class EngineTests(unittest.TestCase):
    def assert_state_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for field in left:
            if isinstance(left[field], torch.Tensor):
                self.assertTrue(torch.equal(left[field], right[field]), field)
            else:
                self.assertEqual(left[field], right[field], field)

    def test_noop_token_and_thought_replay_match_greedy(self):
        for dynamic in (False, True):
            for condition in ("token_replay", "thought_replay"):
                with self.subTest(dynamic=dynamic, condition=condition):
                    run = make_engine(vector=0., dynamic_cache=dynamic, method="rebalance")
                    baseline = run.generate(ITEM, "G")
                    state = native.snapshot(run.model)
                    result = run.generate(ITEM, condition)
                    for key in ("token_ids", "text", "text_sha256", "token_count", "eos_reached"):
                        self.assertEqual(result[key], baseline[key], key)
                    self.assert_state_equal(native.snapshot(run.model), state)
                    self.assertEqual(len(result["events"]), 2)

    def test_self_replay_matches_greedy_with_nonzero_vector(self):
        for condition in ("self_token", "self_thought"):
            with self.subTest(condition=condition):
                run = make_engine(method="rebalance")
                baseline = run.generate(ITEM, "G")
                state = native.snapshot(run.model)
                result = run.generate(ITEM, condition)
                self.assertEqual(result["token_ids"], baseline["token_ids"])
                self.assert_state_equal(native.snapshot(run.model), state)
                self.assertFalse(any(call["native"] for call in run.model.calls))

    def test_donor_cache_and_controller_cannot_pollute_target(self):
        for condition in ("token_replay", "thought_replay"):
            for dynamic in (False, True):
                with self.subTest(condition=condition, dynamic=dynamic):
                    run = make_engine(dynamic_cache=dynamic, method="rebalance")
                    result = run.generate(ITEM, condition)
                    self.assertEqual(result["token_ids"], [SPLIT, B, SPLIT, B, END, A, EOS])
                    clean = [call for call in run.model.calls if not call["native"]]
                    self.assertTrue(any(call["native"] for call in run.model.calls))
                    self.assertTrue(all(call["poison"] == 0 for call in clean))
                    self.assertEqual([call["steps"] for call in clean], list(range(len(clean))))
                    self.assertEqual(run.model.cur_steps, len(clean))
                    self.assertEqual(run.model._prev_step_mean.item(), len(clean))
                    self.assertTrue(all(e["native_cache_discarded"] and e["replay_cache"] == "clean"
                                        for e in result["events"]))

    def test_thought_split_boundary_is_processed_once_per_branch(self):
        run = make_engine()
        result = run.generate(ITEM, "thought_replay")
        self.assertEqual([e["position"] for e in result["events"]], [1, 3])
        self.assertEqual([e["thought_stop"] for e in result["events"]], ["split", "think_end"])
        for native_call in (False, True):
            splits = [call["seen"] for call in run.model.calls
                      if call["native"] == native_call and call["ids"] == [SPLIT]]
            self.assertEqual(splits, [2, 4])
        clean_ids = [token for call in run.model.calls[1:] if not call["native"] for token in call["ids"]]
        self.assertEqual(clean_ids, result["token_ids"][:-1])

    def test_proposal_has_no_128_or_2048_token_cap(self):
        for length in (129, 2049, 2100):
            for self_replay in (False, True):
                with self.subTest(length=length, self_replay=self_replay):
                    fake = Proposal(length, SPLIT)
                    tokens, reason = engine.Engine.propose_thought(
                        fake, fake.output(), None, 10000, self_replay=self_replay)
                    self.assertEqual((len(tokens), reason, tokens[-1]), (length, "split", SPLIT))
                    self.assertEqual(fake.steps, length - 1)
                    self.assertEqual(fake.native_calls, [not self_replay] * (length - 1))

    def test_proposal_stops_at_eos_think_end_split_or_global_budget(self):
        for stop, remaining, expected in ((EOS, 20, "eos"), (END, 20, "think_end"),
                                          (SPLIT, 20, "split"), (SPLIT, 3, "global_budget")):
            with self.subTest(stop=stop, remaining=remaining):
                fake = Proposal(5, stop)
                tokens, reason = engine.Engine.propose_thought(fake, fake.output(), None, remaining)
                self.assertEqual((len(tokens), reason), (min(5, remaining), expected))
                self.assertEqual(fake.steps, len(tokens) - 1)

    def test_global_10000_token_budget_for_main_conditions(self):
        for condition in ("G", "V", "token_replay", "thought_replay"):
            with self.subTest(condition=condition):
                run = make_engine(script=[SPLIT])
                result = run.generate(ITEM, condition)
                self.assertEqual(result["max_new_tokens"], 10000)
                self.assertEqual(result["generated_tokens_including_eos"], 10000)
                self.assertEqual(result["token_count"], 10000)
                self.assertTrue(result["hit_max"])
                self.assertFalse(result["eos_reached"])
                if condition == "thought_replay":
                    self.assertEqual(result["events"][0]["thought_tokens"], 9999)
                    self.assertEqual(result["events"][0]["thought_stop"], "global_budget")
                    self.assertEqual(result["replay_budget_truncations"], 1)
                    self.assertEqual(run.model.cur_steps, 10000)

    def test_replay_eos_at_budget_is_not_reported_as_truncation(self):
        run = make_engine(script=[SPLIT, A, EOS])
        result = run.generate(ITEM, "thought_replay", max_tokens=3)
        self.assertEqual(result["token_ids"], [SPLIT, B, EOS])
        self.assertTrue(result["eos_reached"])
        self.assertFalse(result["hit_max"])
        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["replay_budget_truncations"], 0)

    def test_shadow_restores_cache_state_and_steering_on_exception(self):
        run = make_engine(method="rebalance")
        run.reset()
        cache = ((torch.tensor([2]), torch.zeros(1)),)
        original = native.clone_frozen_cache(cache)
        state = native.snapshot(run.model)
        vector, flag = run.model.steering_vector, run.model.steering_flag
        original_call = Model.__call__

        def fail(model, **kwargs):
            original_call(model, **kwargs)
            raise RuntimeError("donor failed")

        with patch.object(Model, "__call__", fail):
            with self.assertRaisesRegex(RuntimeError, "donor failed"):
                run.shadow(run.query(SPLIT), torch.ones(1, 3), cache, native=True)
        self.assert_state_equal(native.snapshot(run.model), state)
        self.assertIs(run.model.steering_vector, vector)
        self.assertEqual(run.model.steering_flag, flag)
        for layer, expected in zip(cache, original):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(layer, expected)))

    def test_cache_cloning_owns_tensor_storage(self):
        cache = ((torch.tensor([1.]), torch.tensor([2.])),)
        for wrapped in (cache, Cache(cache)):
            frozen = native.cache_to_frozen_legacy(wrapped)
            branch = native.clone_frozen_cache(frozen)
            branch[0][0].add_(100)
            self.assertEqual(cache[0][0].item(), 1.)
            self.assertEqual(frozen[0][0].item(), 1.)
        self.assertIsNone(native.clone_frozen_cache(None))

    def test_7b_cps_uses_prescaled_vector_with_positive_one_coefficient(self):
        run = make_engine(vector=3.5, method="cps")
        run.reset()
        self.assertIs(run.model.steering_vector, run.vector)
        self.assertEqual(run.model.steering_vector.item(), 3.5)
        self.assertEqual(run.model.steering_coef, 1.)
        self.assertEqual(run.model.steering_layer, 20)

    def test_model_loader_uses_configured_vector_without_scaling_or_task_import(self):
        run = make_engine(method="cps")
        model, tokenizer, vector = Mock(), Mock(), Mock()
        model.config.hidden_size = 3584
        model.to.return_value.eval.return_value = model
        vector.numel.return_value = 3584
        vector.to.return_value = vector
        seal = SimpleNamespace(from_pretrained=Mock(return_value=model))
        rebalance = SimpleNamespace(from_pretrained=Mock())
        auto = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        modules = {"spec": spec_stub, "task": None,
                   "transformers": SimpleNamespace(AutoTokenizer=auto),
                   "vendor.seal_model": SimpleNamespace(Qwen2ForCausalLM=seal),
                   "vendor.rebalance_model": SimpleNamespace(Qwen2ForCausalLM=rebalance)}
        with patch.dict(sys.modules, modules), patch.object(native, "reject_model_overrides") as guard, \
             patch.object(torch, "load", return_value=vector) as load, \
             patch.object(torch, "isfinite", return_value=SimpleNamespace(all=lambda: True)):
            spec_stub.read.return_value = {"7": {"files": {"config.json": {}}}}
            self.assertEqual(native.load_model(run.config, "/mock/model"), (model, tokenizer, vector))
        guard.assert_called_once_with(Path("/mock/model"), {"config.json": {}})
        load.assert_called_once_with(ROOT / run.config["vector"], map_location="cpu", weights_only=True)
        vector.to.assert_called_once_with("cuda", dtype=torch.bfloat16)
        rebalance.from_pretrained.assert_not_called()
        self.assertTrue(seal.from_pretrained.call_args.kwargs["local_files_only"])

    def test_model_override_guard_preserves_original_rejections(self):
        def entry(name, is_file=True):
            return SimpleNamespace(name=name, suffix=Path(name).suffix, is_file=lambda: is_file)

        expected = {"model-00001-of-00002.safetensors": {}}
        known = [entry(next(iter(expected))), entry("README.md"), entry("tokenizer_backups", False)]
        with patch.object(Path, "iterdir", return_value=iter(known)):
            native.reject_model_overrides(Path("/mock/model"), expected)
        for name in ("model.safetensors", "pytorch_model.bin", "weights.pt", "weights.pth", "weights.gguf",
                     "adapter_config.json", "tokenizer.json", "added_tokens.json", "special_tokens_map.json",
                     "vocab.json", "merges.txt", "other.index.json"):
            with self.subTest(name=name), patch.object(Path, "iterdir", return_value=iter(known + [entry(name)])):
                with self.assertRaisesRegex(ValueError, "unexpected model/tokenizer overrides"):
                    native.reject_model_overrides(Path("/mock/model"), expected)


class GradingTests(unittest.TestCase):
    def test_gold_answers_preserve_numbers_and_text(self):
        self.assertEqual(grading.gold_answer(dict(answer=42)), "42")
        self.assertEqual(grading.gold_answer(dict(index=1, solution=r"Thus $\boxed{42}$.")), "42")
        self.assertEqual(grading.gold_answer(dict(index=1, solution=r"Thus $\boxed{\text{east}}$.")), "east")
        with self.assertRaises(ValueError):
            grading.gold_answer(dict(index=1, solution=""))

    def test_legacy_percentage_and_strict_sensitivity(self):
        row = dict(text=r"\boxed{1}", response_text=r"\boxed{1}")
        result = grading.evaluate(row, "100")
        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(result["correct"])
        self.assertFalse(result["strict_correct"])

    def test_strict_grading_uses_untrimmed_response(self):
        row = dict(text=r"\boxed{1}", response_text=r"\boxed{1} Question: correction \boxed{2}")
        result = grading.evaluate(row, "2")
        self.assertEqual(result["status"], "ok", result)
        self.assertFalse(result["correct"])
        self.assertTrue(result["strict_correct"])

    def test_cps383_interval_uses_current_cpu_grade_not_historical_label(self):
        row = protocol.reuse()[protocol.key("math15_cps", "V", 383)]
        self.assertFalse(row["audited_correct"])
        result = grading.evaluate(row, grading.gold_answer(protocol.data()[383]))
        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(result["correct"], result)

    def test_timeout_and_child_exit_are_not_incorrect_answers(self):
        row = dict(text=r"\boxed{1}", response_text=r"\boxed{1}")
        for child, status in ((slow_result, "timeout"), (no_result, "error")):
            with self.subTest(status=status), patch.object(grading, "compare_child", child):
                result = grading.evaluate(row, "1", timeout=.05)
                self.assertEqual(result["status"], status)
                self.assertNotIn("correct", result)

    def test_child_parser_errors_are_reported_and_pipe_closed(self):
        connection = Mock()
        grading.compare_child(connection, None, None, "1")
        self.assertEqual(connection.send.call_args.args[0]["status"], "error")
        connection.close.assert_called_once_with()

    def test_evaluator_identity_hashes_own_code_and_vendor(self):
        identity = grading.evaluator_identity()
        self.assertEqual(identity["files"]["grading.py"], grading.sha(ROOT / "grading.py"))
        self.assertIn("vendor/evaluation/grader.py", identity["files"])
        self.assertIn("sympy", identity["packages"])

    def test_grading_import_does_not_import_torch_or_task(self):
        code = ("import sys, types; from pathlib import Path; "
                f"sys.path.insert(0, {str(ROOT)!r}); "
                "spec = types.ModuleType('spec'); "
                f"spec.ROOT = Path({str(ROOT)!r}); sys.modules['spec'] = spec; "
                "import grading; "
                "assert 'torch' not in sys.modules; assert 'task' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-c", code], check=True)


class TaskQueueTests(unittest.TestCase):
    def test_worker_waits_for_reader_lock_and_stops_only_at_sentinel(self):
        work = mp.get_context("spawn").Queue()
        jobs = [(f"math7_seal/{condition}/0017",
                 dict(config_id="math7_seal", condition=condition, index=17, seed=0))
                for condition in ("token_replay", "thought_replay")]
        for entry in jobs + [None]:
            work.put(entry)
        generated = Mock()
        generated.generate.side_effect = lambda *args, **kwargs: dict(token_count=2)
        entered = threading.Event()
        proxy = Mock(spec=["get", "get_nowait"])

        def receive():
            entered.set()
            return work.get(timeout=3)

        proxy.get.side_effect = receive
        proxy.get_nowait.side_effect = work.get_nowait
        modules = {"native": SimpleNamespace(load_model=Mock(return_value=(None, None, None))),
                   "engine": SimpleNamespace(Engine=Mock(return_value=generated))}
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, modules))
                stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
                stack.enter_context(patch.object(task.ctypes, "CDLL", return_value=SimpleNamespace(prctl=lambda *args: 0)))
                stack.enter_context(patch.object(Path, "exists", return_value=False))
                stack.enter_context(patch.multiple(task, configure_runtime=Mock(return_value={}),
                    data=Mock(return_value={17: ITEM}), safety_gate=Mock(return_value={}),
                    sha=Mock(return_value="vector-hash"), validate_result=Mock(), write=Mock()))
                stack.enter_context(patch("builtins.print"))
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                self.assertTrue(work._reader.poll(3), "jobs must be buffered before contention")
                work._rlock.acquire()
                try:
                    future = pool.submit(task.worker, "", "math7_seal", "/mock/model", proxy,
                                         "/mock/output", "protocol", False, os.getppid())
                    self.assertTrue(entered.wait(3), "worker never attempted a blocking read")
                    proxy.get_nowait.assert_not_called()
                    self.assertFalse(future.done(), "worker exited while another reader held the lock")
                finally:
                    work._rlock.release()
                future.result(timeout=5)
                self.assertEqual(generated.generate.call_args_list,
                                 [call(ITEM, job["condition"], max_tokens=10000) for _, job in jobs])
                self.assertEqual(proxy.get.call_args_list, [call(), call(), call()])
                self.assertEqual(task.validate_result.call_count, 2)
        finally:
            work.close()
            work.join_thread()

    def check_controller_shutdown(self, command="run", fail=False):
        cfgid = "math7_seal"
        config = protocol.configurations()[cfgid]
        todo = {protocol.key(cfgid, "token_replay", i):
                dict(config_id=cfgid, condition="token_replay", index=i, seed=0) for i in range(3)}
        baseline = {protocol.key(cfgid, "G", i): dict(token_count=i + 1) for i in range(3)}
        context, work = Mock(), Mock()
        context.Queue.return_value = work
        processes = []

        def process(**kwargs):
            child = Mock(exitcode=1 if fail else 0)
            child.is_alive.return_value = False
            processes.append(child)
            return child

        context.Process.side_effect = process
        args = SimpleNamespace(command=command, configs=cfgid, output="/mock/output", gpus="0,1",
                               workers_per_gpu=2, model_root="/mock/models", cpu_workers=1)
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "mkdir"))
            stack.enter_context(patch.object(Path, "open", mock_open()))
            stack.enter_context(patch.object(Path, "exists", return_value=False))
            stack.enter_context(patch.object(task.fcntl, "flock"))
            stack.enter_context(patch.object(task.mp, "get_context", return_value=context))
            launch = stack.enter_context(patch.object(task.subprocess, "run"))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(patch.multiple(task, verify=Mock(), write=Mock(),
                configurations=Mock(return_value={cfgid: config}), signature=Mock(return_value="protocol"),
                splits=Mock(return_value={"replay": list(range(3))}), sha=Mock(return_value="asset-hash"),
                pending=Mock(side_effect=[todo, {}, {}]), reuse=Mock(return_value=baseline),
                ensure_models=Mock(return_value={"7": "/mock/model"})))
            if fail:
                with self.assertRaisesRegex(RuntimeError, "Worker failure"):
                    task.run(args)
                launch.assert_not_called()
            else:
                task.run(args)
                self.assertEqual(launch.call_count, int(command == "run"))
        slots = 2 if command == "smoke" else 3
        self.assertEqual(len(processes), slots)
        ordered = sorted(todo.items(), key=lambda entry: -entry[1]["index"])
        self.assertEqual(work.put.call_args_list, [call(entry) for entry in ordered] + [call(None)] * slots)
        self.assertEqual(work.method_calls[-2:], [call.cancel_join_thread(), call.close()])
        for child in processes:
            child.start.assert_called_once_with()
            child.join.assert_called_once_with(timeout=15)

    def test_controller_sends_one_sentinel_per_actual_worker(self):
        self.check_controller_shutdown()

    def test_smoke_shutdown_cancels_join_for_unconsumed_jobs(self):
        self.check_controller_shutdown(command="smoke")

    def test_failed_worker_cancels_queue_join_during_shutdown(self):
        self.check_controller_shutdown(fail=True)


class ReportCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {"grading": grading}))
        self.exists = self.stack.enter_context(patch.object(Path, "exists", return_value=False))
        self.read = self.stack.enter_context(patch.object(report, "read"))
        self.write = self.stack.enter_context(patch.object(report, "write"))
        self.sha = self.stack.enter_context(patch.object(report, "sha", return_value="report-v1"))
        self.evaluate = self.stack.enter_context(patch.object(grading, "evaluate", side_effect=lambda *args:
            dict(status="ok", correct=True, strict_correct=True)))
        self.stack.enter_context(patch.object(grading, "gold_answer", return_value="2"))
        self.row = dict(index=17, condition="V", config_id="math15_cps", text=r"\boxed{2}",
                        response_text=r"\boxed{2}", text_sha256="same-text",
                        provenance=dict(kind="new_table2_generation", source="first-source"))

    def grade(self, row=None):
        payload = ("math15_cps/V/0017", self.row if row is None else row,
                   dict(answer="2"), "/mock/output", "same-evaluator")
        return report.grade(payload)[1]

    def cache(self, result):
        self.exists.return_value = True
        self.read.return_value = result
        self.evaluate.reset_mock()
        self.write.reset_mock()

    def test_unchanged_successful_grade_is_reused(self):
        original = self.grade()
        self.cache(original)
        self.assertEqual(self.grade(), original)
        self.evaluate.assert_not_called()
        self.write.assert_not_called()

    def test_report_hash_change_invalidates_cached_grade(self):
        original = self.grade()
        self.cache(dict(original, correct=False, strict_correct=False))
        self.sha.return_value = "report-v2"
        result = self.grade()
        self.evaluate.assert_called_once()
        self.write.assert_called_once()
        self.assertNotEqual(result["identity"], original["identity"])
        self.assertTrue(result["correct"])
        self.sha.assert_called_with(Path(report.__file__))

    def test_provenance_change_invalidates_cached_grade(self):
        original = self.grade()
        self.cache(original)
        row = dict(self.row, provenance=dict(kind="new_table2_generation", source="replacement-source"))
        result = self.grade(row)
        self.evaluate.assert_called_once()
        self.write.assert_called_once()
        self.assertNotEqual(result["identity"], original["identity"])
        self.assertEqual(result["provenance"], row["provenance"])

    def test_missing_raw_text_is_rejected_without_using_historical_label(self):
        row = dict(self.row, audited_correct=True)
        del row["text"]
        with self.assertRaisesRegex(ValueError, "Raw text required"):
            self.grade(row)
        self.evaluate.assert_not_called()
        self.write.assert_not_called()

    def test_historical_labels_are_diagnostic_and_current_grades_are_primary(self):
        for config in ("math15_cps", "math7_seal"):
            for historical, current in ((False, True), (True, False), (True, True), (False, False)):
                with self.subTest(config=config, historical=historical, current=current):
                    row = dict(self.row, config_id=config, audited_correct=historical,
                               provenance=dict(kind="historical_native_raw", source="recovered-raw-text"))
                    self.evaluate.side_effect = lambda *args: dict(status="ok", correct=current, strict_correct=current)
                    self.evaluate.reset_mock()
                    result = self.grade(row)
                    self.evaluate.assert_called_once_with(row, "2")
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["correct"], current)
                    self.assertEqual(result["historical_correct"], historical)
                    self.assertEqual(result["historical_grading_changed"], historical != current)
                    self.assertEqual(result["grading_origin"], "current_cpu_evaluator")
                    self.assertIsNone(result["strict_correct"])

    def test_historical_labels_do_not_mask_current_evaluator_failures(self):
        for status in ("error", "timeout"):
            with self.subTest(status=status):
                self.evaluate.side_effect = lambda *args: dict(status=status)
                result = self.grade(dict(self.row, audited_correct=True))
                self.assertEqual(result["status"], status)
                self.assertNotIn("correct", result)
                self.assertTrue(result["historical_correct"])


if __name__ == "__main__":
    unittest.main()
