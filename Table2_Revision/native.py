"""Minimal native-method adapter; no dependencies on the original workspace."""
from pathlib import Path
import sys

import torch
from spec import generation_seed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
PROMPT_PREFIX = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
FIELDS = ("_coefs", "_step_prob_sum", "_step_tok_count", "_last_maxprob",
          "_prev_step_mean", "steering_think_flag", "new_round", "cur_steps")
DEFAULTS = {
    "15": dict(q25c=.662293, q75c=.94805, low_val_1=-1.02, q25v=.000560, q75v=.011597, low_val_2=-1.91, high_val_2=.1),
    "7": dict(q25c=.666017, q75c=.927745, low_val_1=-1.19, q25v=.000488, q75v=.009931, low_val_2=-2.34, high_val_2=.1),
}


def snapshot(model):
    return {k: getattr(model, k, None).detach().clone() if isinstance(getattr(model, k, None), torch.Tensor)
            else getattr(model, k, None) for k in FIELDS}


def restore(model, state):
    for k, v in state.items():
        setattr(model, k, v.clone() if isinstance(v, torch.Tensor) else v)


def cache_to_frozen_legacy(cache):
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple(tuple(t.detach().clone() for t in layer) for layer in cache)


clone_frozen_cache = cache_to_frozen_legacy


def configure_steering(model, tokenizer, method, vector, layer, coefficient):
    kwargs = dict(steering_flag=True, steering_layer=layer, steer_vec=vector,
                  steer_coef=coefficient, tokenizer=tokenizer)
    if method == "rebalance":
        size = "15" if model.config.hidden_size == 1536 else "7"
        kwargs["dyn_hparams"] = DEFAULTS[size]
    model.set_steering_flag(**kwargs)
    model.start_new_round()


def make_prompt(tokenizer, problem):
    text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT_PREFIX + "Question: " + problem.strip()}],
                                         tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    return text


def split_ids_for(tokenizer):
    return {int(i) for word, i in tokenizer.get_vocab().items() if "\u010a\u010a" in word}


def trim_output(text):
    for marker in ("Answer the following question", "Question:", "Comment:"):
        text = text.split(marker)[0]
    return text


def nucleus_from_sorted(sorted_logits, temperature, top_p):
    probs = torch.softmax(sorted_logits / temperature, dim=-1)
    strict, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(strict, warn_only=True)
        cumulative = torch.cumsum(probs, dim=-1)
    finally:
        torch.use_deterministic_algorithms(strict, warn_only=warn)
    probs = probs.masked_fill(cumulative - probs > top_p, 0.)
    return probs / probs.sum().clamp_min(1e-12)


def nucleus(logits, temperature, top_p):
    sorted_logits, indices = torch.sort(logits.float(), descending=True)
    return sorted_logits, indices, nucleus_from_sorted(sorted_logits, temperature, top_p)


def sample_top_p(logits, temperature, top_p, generator):
    _, ids, probs = nucleus(logits, temperature, top_p)
    return int(ids[torch.multinomial(probs, 1, generator=generator)].item())


def reject_model_overrides(path, expected):
    # Transformers prefers monolithic weights and optional tokenizer sidecars.
    suspicious = []
    for f in Path(path).iterdir():
        if not f.is_file() or f.name in expected:
            continue
        if (f.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf") or
            f.name.startswith(("adapter", "tokenizer", "added_tokens", "special_tokens", "vocab", "merges")) or
            f.name.endswith(".index.json")):
            suspicious.append(f.name)
    if suspicious:
        raise ValueError(f"unexpected model/tokenizer overrides in {path}: {sorted(suspicious)}; use a clean model directory")


def load_model(config, model_path):
    from transformers import AutoTokenizer
    from vendor.seal_model import Qwen2ForCausalLM as SEAL
    from vendor.rebalance_model import Qwen2ForCausalLM as ReBalance
    cls = ReBalance if config["method"] == "rebalance" else SEAL
    dtype = torch.float16 if config["dtype"] == "float16" else torch.bfloat16
    from spec import read
    reject_model_overrides(Path(model_path), read(ROOT/"assets/models.json")[config["size"]]["files"])
    model = cls.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
                               attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    vector = torch.load(ROOT / config["vector"], map_location="cpu", weights_only=True).to("cuda", dtype=dtype)
    if vector.numel() != model.config.hidden_size or not torch.isfinite(vector).all():
        raise ValueError("invalid vector")
    return model, tokenizer, vector
