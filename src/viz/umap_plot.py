from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.analysis.extractor import RoutingData
from src.analysis.metrics import RoutingMetrics

_COLOR_BY_OPTIONS = ("position", "entropy", "top_expert")


def routing_umap_scatter(
    data: RoutingData,
    embedding: np.ndarray,
    metrics: RoutingMetrics,
    color_by: str = "position",
    layer_idx: int = 0,
) -> go.Figure:
    """
    Scatter plot of 2D UMAP embedding.  Each point is a token; hovering shows
    its text, position, and top expert per layer.

    color_by: "position" | "entropy" | "top_expert"
      - position:  gradient from first to last token
      - entropy:   mean routing entropy across all layers
      - top_expert: top-1 expert in layer_idx
    """
    x, y = embedding[:, 0], embedding[:, 1]
    tokens = data.tokens
    seq_len = data.seq_len

    # Build colour values and scale
    if color_by == "position":
        color_vals = np.arange(seq_len, dtype=float)
        colorscale = "Viridis"
        colorbar_title = "Token position"

    elif color_by == "entropy":
        # mean entropy across layers per token
        color_vals = metrics.entropy.mean(axis=0)   # [seq]
        colorscale = "RdYlGn_r"
        colorbar_title = "Mean entropy"

    else:  # top_expert
        color_vals = data.expert_ids[layer_idx, :, 0].astype(float)
        colorscale = "Turbo"
        colorbar_title = f"Top expert (L{layer_idx})"

    # Hover text: token + top experts per layer (show first 8 layers)
    hover = []
    n_show = min(8, data.num_layers)
    for tok in range(seq_len):
        lines = [f"<b>{repr(tokens[tok])}</b>  (pos {tok})"]
        for layer in range(n_show):
            experts = data.expert_ids[layer, tok]
            weights = data.expert_weights[layer, tok]
            pair = ", ".join(f"E{e}({w:.2f})" for e, w in zip(experts, weights))
            lines.append(f"L{layer}: {pair}")
        if data.num_layers > n_show:
            lines.append(f"… ({data.num_layers - n_show} more layers)")
        hover.append("<br>".join(lines))

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[t.strip() or "·" for t in tokens],
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(
                size=14,
                color=color_vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title, thickness=14),
                line=dict(width=1, color="white"),
            ),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Token routing signatures — UMAP projection",
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="#f8f9fa",
    )
    return fig
