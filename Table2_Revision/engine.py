"""Batch-one native inference with isolated cache AND dynamic-controller branches."""
from __future__ import annotations

import hashlib
import torch

from native import (
    cache_to_frozen_legacy, clone_frozen_cache, configure_steering, make_prompt,
    split_ids_for, trim_output, snapshot, restore, generation_seed,
)
from native import sample_top_p as legacy_sample_top_p


def sample_top_p(logits, temperature, top_p, generator):
    # PyTorch 2.3 cannot mark CUDA cumsum strictly deterministic. Preserve the
    # original GPU nucleus implementation rather than silently switch kernels.
    strict = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(strict, warn_only=True)
        return legacy_sample_top_p(logits, temperature, top_p, generator)
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn)


def owned_cache(cache):
    # These tensors have a single committed owner. Copies are needed only when
    # a shadow/donor branches, not after every autoregressive token.
    return cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache


def thought_end(token, split_ids, think_end, eos):
    if token == eos:
        return "eos"
    if token == think_end:
        return "think_end"
    if token in split_ids:
        return "split"
    return None


class Engine:
    def __init__(self, model, tokenizer, vector, config):
        self.model, self.tokenizer, self.vector, self.config = model, tokenizer, vector, config
        self.zero = torch.zeros_like(vector)
        self.splits = split_ids_for(tokenizer)
        start = tokenizer.encode("<think>", add_special_tokens=False)
        end = tokenizer.encode("</think>", add_special_tokens=False)
        if len(start) != 1 or len(end) != 1:
            raise ValueError("native wrapper requires single-token think delimiters")
        self.start, self.end = start[0], end[0]
        self.eos = tokenizer.eos_token_id

    def reset(self):
        configure_steering(self.model, self.tokenizer, self.config["method"], self.vector,
                           self.config["layer"], self.config["coefficient"])
        # Native set_steering_flag resets the other accumulators but omits this
        # one. A preceding problem must not supply the current problem's history.
        if hasattr(self.model, "_prev_step_mean"):
            self.model._prev_step_mean = None

    def forward(self, ids, mask, cache=None, *, native=False, copy_cache=False, disabled=False):
        # A zero vector advances ReBalance's controller on the CLEAN branch.
        # Merely disabling its flag would leave its confidence history stale.
        old_flag, old_vector = self.model.steering_flag, self.model.steering_vector
        self.model.steering_flag = not disabled
        self.model.steering_vector = self.vector if native else self.zero
        try:
            return self.model(input_ids=ids, attention_mask=mask,
                              past_key_values=clone_frozen_cache(cache) if copy_cache else cache,
                              use_cache=True, return_dict=True)
        finally:
            self.model.steering_flag, self.model.steering_vector = old_flag, old_vector

    def shadow(self, ids, mask, cache, *, native):
        before = snapshot(self.model)
        try:
            return self.forward(ids, mask, cache, native=native, copy_cache=True)
        finally:
            restore(self.model, before)

    def query(self, token):
        return torch.tensor([[token]], dtype=torch.long, device=self.model.device)

    def extend(self, mask):
        return torch.cat([mask, mask.new_ones((1, 1))], dim=1)

    def propose_thought(self, initial, mask, remaining, *, self_replay=False):
        cache = owned_cache(initial.past_key_values)
        logits = initial.logits[0, -1]
        tokens = []
        for _ in range(remaining):
            token = int(logits.argmax().item())
            tokens.append(token)
            reason = thought_end(token, self.splits, self.end, self.eos)
            if reason:
                return tokens, reason
            if len(tokens) == remaining:
                break
            mask = self.extend(mask)
            out = self.forward(self.query(token), mask, cache, native=not self_replay)
            cache = owned_cache(out.past_key_values)
            logits = out.logits[0, -1]
        return tokens, "global_budget"

    @torch.inference_mode()
    def generate(self, item, condition, *, seed=0, max_tokens=10000, threshold=None,
                 direct=None, record_shadow=True):
        self.reset()
        native = condition in ("V", "SV") or condition.startswith("EV_")
        replay = condition in ("token_replay", "thought_replay", "self_token", "self_thought")
        whole = condition.startswith(("E_", "EV_"))
        boundary_sample = condition in ("S", "SV")
        disabled = condition == "G_off"
        if not (native or replay or whole or boundary_sample or condition in ("G", "G_off")):
            raise ValueError(f"unknown generation condition: {condition}")
        if whole and not condition.endswith("_all") and threshold is None:
            raise ValueError("quantile condition requires frozen threshold")
        if boundary_sample and direct is None:
            raise ValueError("direct sampling requires calibrated T/top-p")
        prompt = make_prompt(self.tokenizer, item["problem"])
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        mask = encoded["attention_mask"]
        prefill = self.forward(encoded["input_ids"], mask, native=native, disabled=disabled)
        cache = owned_cache(prefill.past_key_values)
        next_token = int(prefill.logits[0, -1].argmax().item())
        # Same RNG for paired S/SV and E/EV; independent of worker or shard.
        resolved_seed = generation_seed(self.config, item["index"], seed, boundary_sample)
        rng = torch.Generator(device=self.model.device).manual_seed(
            resolved_seed)
        think = bool((encoded["input_ids"] == self.start).any().item())
        generated, events = [], []
        sampling_segment = False
        sampled_tokens = 0
        replay_budget_truncations = 0
        native_boundary_calls = 0
        while len(generated) < max_tokens:
            current = next_token
            generated.append(current)
            if current == self.eos or len(generated) >= max_tokens:
                break
            if current == self.start:
                think = True
            elif current == self.end:
                think = False
                sampling_segment = False
            boundary = think and current in self.splits
            query = self.query(current)
            mask = self.extend(mask)
            before = snapshot(self.model) if boundary and replay else None
            old_cache = cache
            base_logits = None
            if boundary and native and (record_shadow or whole):
                shadow = self.shadow(query, mask, cache, native=False)
                base_logits = shadow.logits[0, -1].detach().float()
                del shadow
            out = self.forward(query, mask, cache, native=native, disabled=disabled)
            cache = owned_cache(out.past_key_values)
            logits = out.logits[0, -1]
            if boundary:
                if base_logits is None:
                    base_logits = logits.detach().float()
                top = torch.topk(base_logits, 2).values
                margin = float((top[0] - top[1]).item())
                event = {"ordinal": len(events) + 1, "position": len(generated),
                         "margin": margin, "baseline_top1": int(base_logits.argmax().item()),
                         "sampled": False}
                if replay:
                    clean_state = snapshot(self.model)
                    restore(self.model, before)
                    try:
                        donor = self.forward(query, mask, old_cache, native=not condition.startswith("self_"), copy_cache=True)
                        event["donor_top1"] = int(donor.logits[0, -1].argmax().item())
                        if condition in ("token_replay", "self_token"):
                            proposal, stop = [event["donor_top1"]], "one_token"
                        else:
                            proposal, stop = self.propose_thought(
                                donor, mask, max_tokens - len(generated), self_replay=condition.startswith("self_"))
                        del donor
                    finally:
                        restore(self.model, clean_state)
                    # The final token remains pending: in particular the next split
                    # must be processed once, by the next closed-loop intervention.
                    for token in proposal[:-1]:
                        generated.append(token)
                        mask = self.extend(mask)
                        clean = self.forward(self.query(token), mask, cache)
                        cache = owned_cache(clean.past_key_values)
                        del clean
                    next_token = proposal[-1]
                    event.update(thought_tokens=len(proposal), thought_stop=stop,
                                 replay_cache="clean", native_cache_discarded=True)
                    replay_budget_truncations += int(stop == "global_budget")
                    events.append(event)
                    del out
                    continue
                if native:
                    native_boundary_calls += 1
                if whole:
                    sampling_segment = condition.endswith("_all") or margin <= threshold
                    event["sampled"] = sampling_segment
                elif boundary_sample:
                    event["sampled"] = True
                events.append(event)
            draw = (boundary_sample and boundary) or (whole and sampling_segment and think)
            if draw:
                t, p = (direct["temperature"], direct["top_p"]) if boundary_sample else (0.6, 0.95)
                next_token = sample_top_p(logits, t, p, rng)
                sampled_tokens += 1
            else:
                next_token = int(logits.argmax().item())
            if boundary:
                events[-1].update(chosen_token=next_token,
                                  chosen_deficit=float((base_logits.max() - base_logits[next_token]).item()),
                                  native_cache_committed=native)
            del out
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        response_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = trim_output(response_text)
        return {"index": int(item["index"]), "condition": condition, "seed": seed,
                "text": text, "response_text": response_text, "raw_text": raw, "token_ids": generated,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_count": len(generated) - int(bool(generated) and generated[-1] == self.eos),
                "generated_tokens_including_eos": len(generated),
                "hit_max": len(generated) == max_tokens and generated[-1] != self.eos,
                "eos_reached": bool(generated and generated[-1] == self.eos),
                "closed": "boxed" in text, "events": events,
                "sampled_tokens": sampled_tokens, "sampled_thoughts": sum(e["sampled"] for e in events) if whole else 0,
                "replay_budget_truncations": replay_budget_truncations,
                "native_boundary_calls": native_boundary_calls,
                "cache_policy": "clean_replay" if replay else "native" if native else "unsteered",
                "rng_seed": resolved_seed if whole or boundary_sample else None,
                "max_new_tokens": max_tokens, "threshold": threshold,
                "kernel": dict(direct) if boundary_sample else {"temperature": 0.6, "top_p": 0.95} if whole else None}

    @torch.inference_mode()
    def audit(self, item):
        """Non-result same-prefix audit: null identity, cache and state isolation."""
        self.reset()
        reset_ok = getattr(self.model, "_prev_step_mean", None) is None
        prompt = make_prompt(self.tokenizer, item["problem"])
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        mask = encoded["attention_mask"]
        out = self.forward(encoded["input_ids"], mask)
        cache = cache_to_frozen_legacy(out.past_key_values)
        # Walk a fixed clean prefix token-by-token, exercising ReBalance history.
        suffix = self.tokenizer.encode("Let us calculate carefully.\n\nCheck the result.\n\n", add_special_tokens=False)
        for token in suffix[:-1]:
            mask = self.extend(mask)
            out = self.forward(self.query(token), mask, cache)
            cache = cache_to_frozen_legacy(out.past_key_values)
        if suffix[-1] not in self.splits:
            raise ValueError("audit prefix does not end at a native split")
        before = snapshot(self.model)
        original = clone_frozen_cache(cache)
        mask = self.extend(mask)
        query = self.query(suffix[-1])
        zero = self.shadow(query, mask, cache, native=False)
        off = self.forward(query, mask, cache, copy_cache=True, disabled=True)
        restore(self.model, before)
        native1 = self.shadow(query, mask, cache, native=True)
        self.shadow(query, mask, cache, native=False)
        native2 = self.shadow(query, mask, cache, native=True)

        def equal(a, b):
            if isinstance(a, torch.Tensor):
                return bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))).item()) if a.is_floating_point() else torch.equal(a, b)
            return a == b

        cache_ok = all(torch.equal(a, b) for aa, bb in zip(cache, original) for a, b in zip(aa, bb))
        state_ok = all(equal(before[k], snapshot(self.model)[k]) for k in before)
        zero_error = float((zero.logits.float() - off.logits.float()).abs().max().item())
        repeat_error = float((native1.logits.float() - native2.logits.float()).abs().max().item())
        native_difference = float((native1.logits.float() - zero.logits.float()).abs().max().item())
        result = {"zero_vs_disabled_max_logit_error": zero_error,
                  "native_shadow_repeat_max_logit_error": repeat_error,
                  "native_vs_zero_max_logit_difference": native_difference,
                  "reset_clears_previous_problem_history": reset_ok,
                  "input_cache_unchanged": cache_ok, "controller_state_unchanged": state_ok}
        if zero_error != 0 or repeat_error != 0 or not cache_ok or not state_ok or not reset_ok:
            raise AssertionError(result)
        return result
