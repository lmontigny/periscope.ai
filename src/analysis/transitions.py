from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.analysis.extractor import RoutingData


@dataclass
class TransitionData:
    # [num_layers-1, seq_len] — 1 where top-1 expert changed between adjacent layers
    changed: np.ndarray
    # [num_layers-1] — fraction of tokens that switched expert at each layer boundary
    switch_rate: np.ndarray
    # [num_layers-1, num_experts, num_experts] — transition frequency matrices
    transition_matrices: np.ndarray
    layer_labels: list[str]   # e.g. ["L0→L1", "L1→L2", ...]


def compute_transitions(data: RoutingData) -> TransitionData:
    top1 = data.expert_ids[:, :, 0]          # [layers, seq]  top-1 expert per token per layer
    num_layers, seq_len = top1.shape
    n = data.num_experts

    changed = (top1[:-1] != top1[1:]).astype(np.float32)   # [layers-1, seq]
    switch_rate = changed.mean(axis=-1)                      # [layers-1]

    # Transition matrix: entry (e1, e2) = P(expert goes from e1 at layer i to e2 at layer i+1)
    matrices = np.zeros((num_layers - 1, n, n), dtype=np.float32)
    for i in range(num_layers - 1):
        for t in range(seq_len):
            e1, e2 = int(top1[i, t]), int(top1[i + 1, t])
            matrices[i, e1, e2] += 1
        if seq_len > 0:
            matrices[i] /= seq_len

    labels = [f"L{i}→L{i+1}" for i in range(num_layers - 1)]
    return TransitionData(
        changed=changed,
        switch_rate=switch_rate,
        transition_matrices=matrices,
        layer_labels=labels,
    )
