from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.model.hooks import routing_hooks


@dataclass
class RoutingData:
    tokens: list[str]           # decoded subword tokens (length = seq_len)
    router_logits: np.ndarray   # [num_moe_layers, seq_len, num_experts]
    expert_ids: np.ndarray      # [num_moe_layers, seq_len, top_k]
    expert_weights: np.ndarray  # [num_moe_layers, seq_len, top_k]  — softmax weights
    num_experts: int
    top_k: int
    # next_token_topk[i] = [(token_str, prob), ...] top-5 predictions after token i
    next_token_topk: list[list[tuple[str, float]]] = None

    @property
    def num_layers(self) -> int:
        return self.router_logits.shape[0]

    @property
    def seq_len(self) -> int:
        return self.router_logits.shape[1]


def extract(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> RoutingData:
    """Run a single forward pass and collect MoE routing data for every token."""
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    tokens = [tokenizer.decode([tok]) for tok in input_ids[0].tolist()]

    with torch.inference_mode():
        with routing_hooks(model) as store:
            outputs = model(**inputs)

    if not store.logits:
        raise RuntimeError(
            "No MoE routing data captured. "
            "The model may not use a supported MoE architecture."
        )

    # Infer top_k and num_experts from the first layer's logits
    first_layer_logits = next(iter(store.logits.values()))  # [batch*seq, num_experts]
    num_experts = first_layer_logits.shape[-1]
    top_k = _infer_top_k(model)

    sorted_layers = store.sorted_layers
    num_moe_layers = len(sorted_layers)

    logits_arr = np.zeros((num_moe_layers, seq_len, num_experts), dtype=np.float32)
    expert_ids_arr = np.zeros((num_moe_layers, seq_len, top_k), dtype=np.int32)
    weights_arr = np.zeros((num_moe_layers, seq_len, top_k), dtype=np.float32)

    for out_idx, layer_idx in enumerate(sorted_layers):
        raw = store.logits[layer_idx]  # [batch*seq, num_experts] or [seq, num_experts]
        # Flatten batch dim and take the first (and only) batch item's tokens
        logits_2d = raw.reshape(-1, num_experts)[:seq_len]
        logits_arr[out_idx] = logits_2d

        probs = _softmax(logits_2d)                          # [seq, num_experts]
        top_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]  # [seq, top_k]
        top_weights = np.take_along_axis(probs, top_indices, axis=-1)

        expert_ids_arr[out_idx] = top_indices
        weights_arr[out_idx] = top_weights

    next_token_topk = _decode_top_predictions(outputs.logits[0], tokenizer, k=5)

    return RoutingData(
        tokens=tokens,
        router_logits=logits_arr,
        expert_ids=expert_ids_arr,
        expert_weights=weights_arr,
        num_experts=num_experts,
        top_k=top_k,
        next_token_topk=next_token_topk,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _decode_top_predictions(
    lm_logits: torch.Tensor,          # [seq_len, vocab_size]
    tokenizer: PreTrainedTokenizerBase,
    k: int = 5,
) -> list[list[tuple[str, float]]]:
    """
    For each token position i, return the top-k predicted next tokens
    (what the model thinks comes after token i given the full context up to i).
    Returns a list of length seq_len; each entry is [(token_str, prob), ...].
    """
    probs = torch.softmax(lm_logits.float(), dim=-1)   # [seq, vocab]
    top_probs, top_ids = probs.topk(k, dim=-1)         # both [seq, k]
    result = []
    for pos in range(lm_logits.shape[0]):
        entry = [
            (tokenizer.decode([int(top_ids[pos, j])]), float(top_probs[pos, j]))
            for j in range(k)
        ]
        result.append(entry)
    return result


def _infer_top_k(model: PreTrainedModel) -> int:
    cfg = model.config
    for attr in ("num_experts_per_tok", "top_k_experts", "top_k"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return 2  # Mixtral default
