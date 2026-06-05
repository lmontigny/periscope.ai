from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.generation import GenerationData

_EXPERT_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3",
]


def generation_heatmap(data: GenerationData, show_up_to: int | None = None) -> go.Figure:
    """
    Heatmap of top-1 expert assignments.
    x = tokens (prompt in lighter shade, generated in full colour)
    y = MoE layer
    Vertical dashed line separates prompt from generated tokens.
    show_up_to: how many total tokens to display (for replay slider).
    """
    n = show_up_to if show_up_to is not None else data.total_tokens
    n = max(1, min(n, data.total_tokens))

    tokens = [t.replace(" ", "·") for t in data.all_tokens[:n]]
    # [layers, n_tokens]
    z = data.expert_ids[:n, :, 0].T.astype(float)   # expert_ids is [tokens, layers, top_k]

    num_layers = data.num_layers
    num_experts = data.num_experts
    palette = _EXPERT_PALETTE
    colorscale = [
        [i / max(num_experts - 1, 1), palette[i % len(palette)]]
        for i in range(num_experts)
    ]

    # Hover text
    hover = []
    for layer in range(num_layers):
        row = []
        for tok in range(n):
            experts = data.expert_ids[tok, layer]
            weights = data.expert_weights[tok, layer]
            row.append(
                f"Token: {repr(data.all_tokens[tok])}<br>"
                f"Layer {layer}: "
                + ", ".join(f"E{e}({w:.2f})" for e, w in zip(experts, weights))
            )
        hover.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=tokens,
            y=[f"L{i}" for i in range(num_layers)],
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=num_experts - 1,
            showscale=True,
            colorbar=dict(title="Expert", thickness=12),
        )
    )

    # Dashed vertical line at prompt/generation boundary
    if data.num_prompt_tokens < n:
        boundary = data.num_prompt_tokens - 0.5
        fig.add_vline(
            x=boundary,
            line_dash="dash",
            line_color="white",
            line_width=2,
            annotation_text="↑ generation starts",
            annotation_position="top right",
            annotation_font_color="white",
        )

    fig.update_layout(
        title="Expert assignment — prompt + generation",
        xaxis_title="Token",
        yaxis_title="MoE Layer",
        height=max(300, 22 * num_layers),
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig


def generation_entropy_chart(data: GenerationData, show_up_to: int | None = None) -> go.Figure:
    """
    Line chart of mean routing entropy per token.
    Entropy = how uncertain the router is (averaged across MoE layers).
    Vertical line separates prompt from generated tokens.
    """
    n = show_up_to if show_up_to is not None else data.total_tokens
    n = max(1, min(n, data.total_tokens))

    logits = data.router_logits[:n]       # [n, layers, experts]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    probs = np.clip(probs, 1e-9, 1.0)
    # entropy per token per layer, then mean over layers
    entropy = -(probs * np.log(probs)).sum(axis=-1).mean(axis=-1)  # [n]

    tokens = [t.replace(" ", "·") for t in data.all_tokens[:n]]
    colors = ["#aec6e8"] * data.num_prompt_tokens + ["#e8634e"] * (n - data.num_prompt_tokens)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=tokens,
            y=entropy,
            mode="lines+markers",
            line=dict(color="#5b9bd5", width=2),
            marker=dict(color=colors[:n], size=8, line=dict(width=1, color="white")),
            hovertemplate="Token: %{x}<br>Entropy: %{y:.3f}<extra></extra>",
            name="Routing entropy",
        )
    )

    if data.num_prompt_tokens < n:
        boundary_token = tokens[data.num_prompt_tokens] if data.num_prompt_tokens < len(tokens) else None
        if boundary_token:
            fig.add_vline(
                x=boundary_token,
                line_dash="dash",
                line_color="gray",
                annotation_text="generation →",
                annotation_position="top right",
            )

    fig.update_layout(
        title="Mean routing entropy per token",
        xaxis_title="Token",
        yaxis_title="Entropy (nats)",
        height=280,
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=80),
        xaxis=dict(tickangle=-45),
        plot_bgcolor="#f8f9fa",
    )
    return fig
