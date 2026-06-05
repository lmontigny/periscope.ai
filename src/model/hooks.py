"""
PyTorch forward hooks that capture MoE router logits during a forward pass.

Supported architectures (auto-detected):
- OLMoE:    layers[i].mlp        (OlmoeMoE)          router at .gate
- Mixtral:  layers[i].block_sparse_moe               router at .gate
- PhiMoE:   layers[i].mlp        (PhimoeSparseMoeBlock) router at .router

We hook the router linear layer directly rather than the MoE block output,
so we are independent of each model's return-value conventions.
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
    # layer_idx → raw router logits [batch*seq, num_experts] (CPU numpy)
    logits: dict[int, np.ndarray] = field(default_factory=dict)

    def clear(self) -> None:
        self.logits.clear()

    @property
    def sorted_layers(self) -> list[int]:
        return sorted(self.logits.keys())


# Router attribute names to probe, in priority order
_ROUTER_ATTRS = ("gate", "router")
# MoE block attribute names to probe on the decoder layer
_BLOCK_ATTRS = ("block_sparse_moe", "mlp", "moe")


def _find_router(layer: nn.Module) -> tuple[nn.Linear, nn.Module] | None:
    """
    Return (router_linear, moe_block) for a decoder layer, or None.
    Tries all known block-attribute and router-attribute name combinations.
    """
    for block_attr in _BLOCK_ATTRS:
        block = getattr(layer, block_attr, None)
        if block is None:
            continue
        for router_attr in _ROUTER_ATTRS:
            router = getattr(block, router_attr, None)
            if router is not None and isinstance(router, nn.Linear):
                return router, block
    return None


def _make_router_hook(store: RoutingStore, layer_idx: int):
    """Hook on the router nn.Linear.

    Most architectures return a plain tensor [batch*seq, num_experts].
    PhiMoE returns (logits, weights, indices) — we always take the first element.
    """
    def hook(module: nn.Module, inputs, output):
        if isinstance(output, torch.Tensor):
            logits = output
        elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
            logits = output[0]
        else:
            return
        store.logits[layer_idx] = logits.detach().float().cpu().numpy()
    return hook


@contextmanager
def routing_hooks(model: nn.Module) -> Iterator[RoutingStore]:
    """Context manager: register hooks on all MoE router layers, yield store."""
    store = RoutingStore()
    handles = []
    for i, layer in enumerate(model.model.layers):
        result = _find_router(layer)
        if result is not None:
            router_linear, _ = result
            h = router_linear.register_forward_hook(_make_router_hook(store, i))
            handles.append(h)
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
