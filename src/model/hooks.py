"""
PyTorch forward hooks that capture MoE router logits during a forward pass.

Supports both OLMoE (layers[i].mlp → OlmoeMoE) and Mixtral
(layers[i].block_sparse_moe → MixtralSparseMoeBlock).  Both architectures
return (hidden_states, router_logits) from their MoE block forward, where
router_logits has shape [batch * seq_len, num_experts].
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RoutingStore:
    """Accumulates per-layer router logits during a single forward pass."""
    # layer_idx → raw gate logits [batch*seq, num_experts] (CPU numpy)
    logits: dict[int, np.ndarray] = field(default_factory=dict)

    def clear(self) -> None:
        self.logits.clear()

    @property
    def sorted_layers(self) -> list[int]:
        return sorted(self.logits.keys())


def _get_moe_block(layer: nn.Module) -> nn.Module | None:
    """Return the MoE sub-module from a decoder layer, or None."""
    # OLMoE: OlmoeDecoderLayer.mlp is OlmoeMoE and has a .gate attribute
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "gate") and isinstance(mlp.gate, nn.Linear):
        return mlp
    # Mixtral: MistralDecoderLayer.block_sparse_moe has a .gate attribute
    moe = getattr(layer, "block_sparse_moe", None)
    if moe is not None and hasattr(moe, "gate") and isinstance(moe.gate, nn.Linear):
        return moe
    return None


def _make_moe_output_hook(store: RoutingStore, layer_idx: int):
    def hook(module: nn.Module, inputs, output):
        # Both OlmoeMoE and MixtralSparseMoeBlock return (hidden_states, router_logits)
        if not (isinstance(output, (tuple, list)) and len(output) >= 2):
            return
        logits = output[1]
        if not isinstance(logits, torch.Tensor):
            return
        store.logits[layer_idx] = logits.detach().float().cpu().numpy()
    return hook


@contextmanager
def routing_hooks(model: nn.Module) -> Iterator[RoutingStore]:
    """Context manager: register hooks, yield store, remove hooks on exit."""
    store = RoutingStore()
    handles = []
    for i, layer in enumerate(model.model.layers):
        moe = _get_moe_block(layer)
        if moe is not None:
            h = moe.register_forward_hook(_make_moe_output_hook(store, i))
            handles.append(h)
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
