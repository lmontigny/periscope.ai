from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.analysis.extractor import RoutingData


@dataclass
class RoutingMetrics:
    # [num_moe_layers, seq_len] — Shannon entropy of softmax distribution
    entropy: np.ndarray
    # [num_moe_layers, num_experts] — fraction of tokens routed to each expert
    expert_load: np.ndarray
    # [num_experts, num_experts] — how often each expert pair is co-selected
    coactivation: np.ndarray


def compute_metrics(data: RoutingData) -> RoutingMetrics:
    entropy = _routing_entropy(data.router_logits)
    load = _expert_load(data.expert_ids, data.num_experts)
    coact = _expert_coactivation(data.expert_ids, data.num_experts)
    return RoutingMetrics(entropy=entropy, expert_load=load, coactivation=coact)


def _routing_entropy(logits: np.ndarray) -> np.ndarray:
    """Shannon entropy of the softmax distribution per token per layer."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-9, 1.0)
    return -(probs * np.log(probs)).sum(axis=-1)  # [layers, seq]


def _expert_load(expert_ids: np.ndarray, num_experts: int) -> np.ndarray:
    """Fraction of (token, top-k slot) assignments per expert per layer."""
    num_layers, seq_len, top_k = expert_ids.shape
    load = np.zeros((num_layers, num_experts), dtype=np.float32)
    for layer in range(num_layers):
        flat = expert_ids[layer].ravel()  # [seq * top_k]
        counts = np.bincount(flat, minlength=num_experts).astype(np.float32)
        load[layer] = counts / counts.sum()
    return load


def _expert_coactivation(expert_ids: np.ndarray, num_experts: int) -> np.ndarray:
    """
    Symmetric [num_experts, num_experts] matrix: entry (i,j) is the fraction
    of tokens where expert i and j are both in the top-k, summed across layers.
    """
    num_layers, seq_len, top_k = expert_ids.shape
    matrix = np.zeros((num_experts, num_experts), dtype=np.float32)
    total = 0
    for layer in range(num_layers):
        for tok in range(seq_len):
            selected = expert_ids[layer, tok]
            for a in range(len(selected)):
                for b in range(a + 1, len(selected)):
                    i, j = int(selected[a]), int(selected[b])
                    matrix[i, j] += 1
                    matrix[j, i] += 1
            total += 1
    if total > 0:
        matrix /= total
    return matrix
