"""Additional causal interventions; native wrappers and replay remain shared."""
import hashlib
import time

import torch

import bridge
from engine import Engine, owned_cache, sample_top_p
from native import make_prompt, trim_output
from kernels import direction, draw_depth, gate_uniform
from protocol import MICRO_ORDINAL, seeded


class CausalEngine(Engine):
    def forward(self, ids, mask, cache=None, **kwargs):
        self.calls = getattr(self, "calls", 0) + 1
        self.processed_tokens = getattr(self, "processed_tokens", 0) + ids.numel()
        return super().forward(ids, mask, cache, **kwargs)

    def run(self, item, condition, seed=0, max_tokens=10000, calibration=None,
            baseline=None, smoke=False):
        self.calls = self.processed_tokens = 0
        started = time.monotonic()
        original = self.vector
        provenance = {}
        try:
            if condition == "reverse" or condition.startswith("random_"):
                if condition == "reverse":
                    self.vector = -original
                else:
                    number = int(condition.rsplit("_", 1)[1])
                    scale = (calibration["random_scales"][str(number)]["selected_scale"]
                             if condition.startswith("random_calibrated_") else 1.)
                    self.vector = direction(original, self.config["id"], number) * scale
                    provenance = dict(direction_index=number, scale=scale)
                provenance.update(vector_norm=float(self.vector.float().norm().item()),
                                  original_norm=float(original.float().norm().item()),
                                  vector_sha256=hashlib.sha256(self.vector.float().cpu().numpy().tobytes()).hexdigest())
                row = self.generate(item, "V", seed=seed, max_tokens=max_tokens)
            elif condition in ("G", "G_off", "V", "token_replay", "thought_replay", "self_token", "self_thought"):
                row = self.generate(item, condition, seed=seed, max_tokens=max_tokens)
            else:
                row = self.extra(item, condition, seed, max_tokens, calibration, baseline, smoke)
        finally:
            self.vector = original
        row.update(condition=condition, forward_calls=self.calls, processed_input_tokens=self.processed_tokens,
                   wall_seconds=time.monotonic()-started, intervention=provenance,
                   timing_scope="instrumented generation including shadow forwards and replay")
        return row

    def frozen_prefix(self, baseline, smoke):
        ordinal = 1 if smoke else MICRO_ORDINAL
        think, count = True, 0
        for n, token in enumerate(baseline["token_ids"]):
            if token == self.start:
                think = True
            if token == self.end:
                think = False
            if think and token in self.splits:
                count += 1
                if count == ordinal:
                    return baseline["token_ids"][:n+1]
        return None

    @torch.inference_mode()
    def extra(self, item, condition, seed, maximum, calibration, baseline, smoke):
        self.reset()
        micro = condition.startswith("micro_")
        depth = condition.startswith("depth_") or micro and condition != "micro_greedy"
        whole = condition.startswith(("whole_", "wholeV_", "discover_"))
        native = condition.startswith("wholeV_") or condition == "cache_only"
        adjacent = whole and "_adjacent_" in condition
        stage = condition.rsplit("_", 1)[-1]
        threshold = calibration["thresholds_adjacent" if adjacent else "thresholds"].get(stage) if whole else None
        if whole and stage != "all" and threshold is None:
            raise ValueError("Missing frozen uncertainty threshold")
        kernel = calibration["direct"]["selected"] if depth or micro else None
        # Sampling streams never depend on steering, direction, worker or GPU.
        family = "micro" if micro else "depth" if depth else "whole" if whole else condition
        resolved = seeded(self.config["id"], item["index"], seed, family)
        rng = torch.Generator(device=self.model.device).manual_seed(resolved)
        prompt = self.tokenizer(make_prompt(self.tokenizer, item["problem"]), return_tensors="pt").to(self.model.device)
        mask = prompt["attention_mask"]
        out = self.forward(prompt["input_ids"], mask, native=native)
        cache = owned_cache(out.past_key_values)
        next_token = (sample_top_p(out.logits[0, -1], .6, .95, rng) if condition == "official_full"
                      else int(out.logits[0, -1].argmax().item()))
        think = bool((prompt["input_ids"] == self.start).any().item())
        generated, events = [], []
        sampled_tokens = int(condition == "official_full")
        segment, previous_boundary = False, False
        micro_pending = micro
        prefix = None
        if micro:
            prefix = self.frozen_prefix(baseline, smoke)
            if prefix is None or len(prefix) >= maximum:
                return self.unavailable(item, seed, maximum, "requested_boundary_absent_or_at_cap")
            # Rebuild a clean cache from raw token ids; keep the split pending.
            for token in prefix[:-1]:
                generated.append(token)
                mask = self.extend(mask)
                out = self.forward(self.query(token), mask, cache)
                cache = owned_cache(out.past_key_values)
            next_token = prefix[-1]
        del out
        while len(generated) < maximum:
            current = next_token
            generated.append(current)
            if current == self.eos or len(generated) == maximum:
                break
            if current == self.start:
                think = True
            if current == self.end:
                think, segment = False, False
            boundary = think and current in self.splits
            gate_site = boundary if not adjacent else think and previous_boundary and not boundary
            mask = self.extend(mask)
            clean_logits = None
            if boundary and (native or condition == "self_cache"):
                clean = self.shadow(self.query(current), mask, cache, native=False)
                clean_logits = clean.logits[0, -1].float().detach()
                del clean
            out = self.forward(self.query(current), mask, cache, native=native)
            cache = owned_cache(out.past_key_values)
            logits = out.logits[0, -1]
            base = clean_logits if clean_logits is not None else logits.float()
            if condition == "cache_only" and boundary:
                logits = base
            if boundary:
                segment = False
            event = None
            if boundary or gate_site:
                top = torch.topk(base, 2).values
                event = dict(position=len(generated), ordinal=1+sum(e["boundary"] for e in events),
                             boundary=boundary, gate_site=gate_site,
                             margin=float((top[0]-top[1]).item()), baseline_top1=int(base.argmax().item()),
                             sampled=False, native_cache_committed=native)
            if whole and gate_site:
                segment = stage == "all" or event["margin"] <= threshold
                event["sampled"] = segment
            draw_depth_now = boundary and (condition.startswith("depth_") or micro_pending)
            if draw_depth_now:
                factor = {"matched": 1., "shallow": .5, "deep": 2., "greedy": 1.}[stage]
                chosen, details = draw_depth(base, kernel, factor, rng,
                                             gate_uniform(self.config["id"], item["index"], seed, event["ordinal"]),
                                             force=micro)
                if micro and not details["conditional_available"]:
                    return self.unavailable(item, seed, maximum, "no_non_greedy_reference_support", prefix)
                next_token = int(base.argmax().item()) if condition == "micro_greedy" else chosen
                event.update(details)
                if condition == "micro_greedy":
                    event.update(attempted=False, departure=False, deficit=0.)
                sampled_tokens += int(details["attempted"] and condition != "micro_greedy")
                micro_pending = False
            elif (whole and segment and think) or condition == "official_full" or condition == "boundary_official" and boundary:
                next_token = sample_top_p(logits, .6, .95, rng)
                sampled_tokens += 1
                if event is not None and condition == "boundary_official":
                    event["sampled"] = True
            else:
                next_token = int(logits.argmax().item())
            if event is not None:
                event.update(chosen_token=next_token, departure=next_token != event["baseline_top1"],
                             deficit=float((base.max()-base[next_token]).item()))
                events.append(event)
            previous_boundary = boundary
            del out
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = trim_output(response)
        return dict(index=int(item["index"]), seed=seed, status="ok", text=text, response_text=response,
                    raw_text=self.tokenizer.decode(generated, skip_special_tokens=False), token_ids=generated,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    token_count=len(generated)-int(generated[-1] == self.eos), generated_tokens_including_eos=len(generated),
                    hit_max=len(generated) == maximum and generated[-1] != self.eos,
                    eos_reached=generated[-1] == self.eos, closed="boxed" in text,
                    sampled_tokens=sampled_tokens, sampled_thoughts=sum(e["sampled"] for e in events) if whole else 0,
                    events=events, rng_seed=resolved, max_new_tokens=maximum, threshold=threshold,
                    kernel=kernel, cache_policy="native" if native else "unsteered",
                    frozen_prefix_sha256=hashlib.sha256(str(prefix).encode()).hexdigest() if micro else None,
                    frozen_prefix_tokens=len(prefix) if micro else None,
                    micro_ordinal=(1 if smoke else MICRO_ORDINAL) if micro else None)

    @staticmethod
    def unavailable(item, seed, maximum, reason, prefix=None):
        return dict(index=item["index"], seed=seed, status="unavailable", reason=reason,
                    max_new_tokens=maximum, token_ids=[], text="", response_text="", events=[],
                    frozen_prefix_sha256=hashlib.sha256(str(prefix).encode()).hexdigest() if prefix else None)
