from __future__ import annotations

import numpy as np

from src.analysis.extractor import RoutingData


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def routing_feature_matrix(data: RoutingData) -> np.ndarray:
    """
    Build a [seq_len, num_layers * num_experts] feature matrix.
    Each row is a token's routing signature: the full softmax probability
    distribution flattened across all MoE layers.
    """
    # router_logits: [layers, seq, experts] → probs same shape
    probs = _softmax(data.router_logits)              # [layers, seq, experts]
    # transpose to [seq, layers, experts] then flatten last two dims
    return probs.transpose(1, 0, 2).reshape(data.seq_len, -1).astype(np.float32)


def compute_umap(data: RoutingData, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Project each token's routing signature to 2D via UMAP.
    Returns [seq_len, 2].  Raises ValueError when seq_len is too small.
    """
    import umap  # imported here so the rest of the app loads without umap installed

    seq_len = data.seq_len
    if seq_len < 4:
        raise ValueError(
            f"Need at least 4 tokens for a meaningful UMAP projection (got {seq_len}). "
            "Try a longer input."
        )

    features = routing_feature_matrix(data)
    n_neighbors = min(n_neighbors, seq_len - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(features)
