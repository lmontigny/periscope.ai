from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.hooks import routing_hooks
from src.analysis.extractor import _infer_top_k, _softmax


@dataclass
class GenerationData:
    prompt_tokens: list[str]        # decoded prompt tokens
    generated_tokens: list[str]     # decoded generated tokens (not including prompt)
    # shape: [total_tokens, num_moe_layers, num_experts]
    router_logits: np.ndarray
    # shape: [total_tokens, num_moe_layers, top_k]
    expert_ids: np.ndarray
    expert_weights: np.ndarray
    num_experts: int
    top_k: int

    @property
    def all_tokens(self) -> list[str]:
        return self.prompt_tokens + self.generated_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def num_layers(self) -> int:
        return self.router_logits.shape[1]


def generate_with_routing(
    prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 60,
    temperature: float = 0.0,
    progress_fn=None,
) -> GenerationData:
    """
    Run greedy/sampled generation while capturing MoE routing for every token.

    Phase 1 — prefill: one forward pass on the full prompt, captures routing
    for all prompt tokens simultaneously.
    Phase 2 — decode: one forward pass per new token (KV-cached), captures
    routing for each generated token.

    progress_fn(step, total): optional callable for a Streamlit progress bar.
    """
    device = next(model.parameters()).device
    top_k = _infer_top_k(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # ── Phase 1: prefill ──────────────────────────────────────────────────────
    with torch.inference_mode():
        with routing_hooks(model) as store:
            outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    num_experts, sorted_layers = _read_store_meta(store)

    # Collect routing for all prompt tokens: shape [prompt_len, layers, experts]
    prefill_logits = _collect_logits(store, sorted_layers, num_experts, seq_len=prompt_len)

    # ── Phase 2: decode ───────────────────────────────────────────────────────
    decode_logits_list: list[np.ndarray] = []
    generated_tokens: list[str] = []

    logits_last = outputs.logits[:, -1, :]  # [1, vocab]

    for step in range(max_new_tokens):
        if progress_fn:
            progress_fn(step, max_new_tokens)

        next_id = _sample(logits_last, temperature)  # [1, 1]
        token_str = tokenizer.decode(next_id[0])
        generated_tokens.append(token_str)

        if next_id[0, 0].item() == tokenizer.eos_token_id:
            break

        with torch.inference_mode():
            with routing_hooks(model) as store:
                outputs = model(
                    input_ids=next_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        past_key_values = outputs.past_key_values
        logits_last = outputs.logits[:, -1, :]

        # Each decode step processes exactly 1 token
        step_logits = _collect_logits(store, sorted_layers, num_experts, seq_len=1)
        decode_logits_list.append(step_logits)

    if progress_fn:
        progress_fn(max_new_tokens, max_new_tokens)

    # ── Assemble ──────────────────────────────────────────────────────────────
    if decode_logits_list:
        all_logits = np.concatenate(
            [prefill_logits] + [l[np.newaxis] for l in decode_logits_list], axis=0
        )
    else:
        all_logits = prefill_logits

    # all_logits: [total_tokens, num_layers, num_experts]
    total = all_logits.shape[0]
    num_layers = all_logits.shape[1]

    expert_ids = np.zeros((total, num_layers, top_k), dtype=np.int32)
    expert_weights = np.zeros((total, num_layers, top_k), dtype=np.float32)

    for t in range(total):
        for l in range(num_layers):
            probs = _softmax(all_logits[t, l])
            idx = np.argsort(probs)[-top_k:][::-1]
            expert_ids[t, l] = idx
            expert_weights[t, l] = probs[idx]

    return GenerationData(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        router_logits=all_logits,
        expert_ids=expert_ids,
        expert_weights=expert_weights,
        num_experts=num_experts,
        top_k=top_k,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_store_meta(store) -> tuple[int, list[int]]:
    sorted_layers = store.sorted_layers
    num_experts = store.logits[sorted_layers[0]].reshape(-1, store.logits[sorted_layers[0]].shape[-1]).shape[-1]
    return num_experts, sorted_layers


def _collect_logits(
    store, sorted_layers: list[int], num_experts: int, seq_len: int
) -> np.ndarray:
    """Return [seq_len, num_layers, num_experts] float32 array."""
    num_layers = len(sorted_layers)
    out = np.zeros((seq_len, num_layers, num_experts), dtype=np.float32)
    for out_idx, layer_idx in enumerate(sorted_layers):
        raw = store.logits[layer_idx].reshape(-1, num_experts)[:seq_len]
        out[:, out_idx, :] = raw
    return out


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return [1, 1] next-token id. Greedy when temperature == 0."""
    if temperature == 0.0 or temperature < 1e-6:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
