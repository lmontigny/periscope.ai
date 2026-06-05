from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.analysis.extractor import RoutingData

# 64 visually distinct colours (wraps for models with fewer experts)
_EXPERT_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
    "#000000", "#e6beff", "#aa6e28", "#fffac8", "#800080", "#808000", "#00ff00",
    "#ff00ff", "#00ffff", "#ff0000", "#0000ff", "#ff8000", "#00ff80", "#8000ff",
    "#ff0080", "#80ff00", "#0080ff", "#ff8080", "#80ff80", "#8080ff", "#ffff00",
    "#00ff00", "#0000ff", "#ff00ff", "#ff4000", "#00ff40", "#4000ff", "#ff0040",
    "#40ff00", "#0040ff", "#ff8040", "#40ff80", "#8040ff", "#ff4080", "#80ff40",
    "#4080ff", "#ffaa00", "#00ffaa", "#aa00ff", "#ffaa80", "#80ffaa", "#aa80ff",
    "#ffaaaa",
]


def expert_assignment_heatmap(data: RoutingData) -> go.Figure:
    """
    Heatmap: x = tokens, y = MoE layer index.
    Each cell shows the top-1 expert (colour) and routing weight (opacity).
    """
    num_layers, seq_len, _ = data.expert_ids.shape
    tokens = [t.replace(" ", "·") for t in data.tokens]

    # Build z (expert index), text (hover), and colour arrays
    z = data.expert_ids[:, :, 0].astype(float)           # top-1 expert per cell
    top1_weights = data.expert_weights[:, :, 0]           # weight for top-1

    # Map expert IDs to distinct colours
    palette = _EXPERT_PALETTE
    colorscale = [
        [i / (data.num_experts - 1), palette[i % len(palette)]]
        for i in range(data.num_experts)
    ]

    # Build hover text
    hover = []
    for layer in range(num_layers):
        row = []
        for tok in range(seq_len):
            experts = data.expert_ids[layer, tok]
            weights = data.expert_weights[layer, tok]
            expert_str = ", ".join(
                f"E{e}({w:.2f})" for e, w in zip(experts, weights)
            )
            row.append(
                f"Token: {repr(data.tokens[tok])}<br>"
                f"Layer: {layer}<br>"
                f"Experts: {expert_str}"
            )
        hover.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=tokens,
            y=[f"Layer {i}" for i in range(num_layers)],
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=data.num_experts - 1,
            showscale=True,
            colorbar=dict(title="Expert ID", tickvals=list(range(0, data.num_experts, max(1, data.num_experts // 8)))),
        )
    )
    fig.update_layout(
        title="Top-1 Expert Assignment per Token per Layer",
        xaxis_title="Token",
        yaxis_title="MoE Layer",
        height=max(300, 40 * num_layers),
        margin=dict(l=80, r=40, t=60, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig


def entropy_heatmap(data: RoutingData, entropy: np.ndarray) -> go.Figure:
    """
    Heatmap of routing entropy [layers × tokens].
    High entropy = router is uncertain; low = strongly prefers one expert.
    """
    tokens = [t.replace(" ", "·") for t in data.tokens]
    fig = go.Figure(
        go.Heatmap(
            z=entropy,
            x=tokens,
            y=[f"Layer {i}" for i in range(data.num_layers)],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Entropy"),
            hovertemplate="Token: %{x}<br>Layer: %{y}<br>Entropy: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Routing Entropy per Token per Layer",
        xaxis_title="Token",
        yaxis_title="MoE Layer",
        height=max(300, 40 * data.num_layers),
        margin=dict(l=80, r=40, t=60, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig
