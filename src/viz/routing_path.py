"""
Routing path grid chart — replicates the diagram from "The Complexity of Routing":
  - columns = MoE layers
  - rows    = experts
  - faded circles = inactive experts
  - bold circles  = active (top-k) experts, size ∝ routing weight
  - lines         = one path per top-k slot connecting active experts across layers
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# One colour per top-k rank (top-1 is darkest)
_PATH_COLORS = ["#5c00a3", "#9b30e8", "#c87eff", "#e2b8ff"]


def routing_path_chart(
    expert_ids: np.ndarray,      # [num_layers, top_k]
    expert_weights: np.ndarray,  # [num_layers, top_k]
    num_experts: int,
    title: str = "Routing path",
) -> go.Figure:
    """
    Grid routing path chart.  Works for any number of layers / experts / top-k.
    """
    num_layers, top_k = expert_ids.shape
    marker_size = max(6, min(14, 160 // max(num_experts, 1)))

    fig = go.Figure()

    # ── Background: all inactive experts (faded circles) ─────────────────────
    bg_x, bg_y = [], []
    for layer in range(num_layers):
        for e in range(num_experts):
            bg_x.append(layer)
            bg_y.append(e)

    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y,
        mode="markers",
        marker=dict(size=marker_size, color="#c8a8e8", opacity=0.20, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── One path + active circles per top-k rank ─────────────────────────────
    for k in range(top_k):
        color = _PATH_COLORS[k % len(_PATH_COLORS)]
        path_x = list(range(num_layers))
        path_y = expert_ids[:, k].tolist()
        weights = expert_weights[:, k]

        # Path line
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode="lines",
            line=dict(width=max(1, 4 - k), color=color),
            name=f"Expert #{k + 1}",
            legendgroup=f"k{k}",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Active expert markers (size ∝ weight)
        hover = [
            f"Layer {layer}<br>Expert {int(expert_ids[layer, k])}"
            f"<br>Weight {weights[layer]:.3f}"
            for layer in range(num_layers)
        ]
        sizes = [marker_size + w * marker_size for w in weights]

        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.90,
                line=dict(width=2, color="white"),
            ),
            name=f"Expert #{k + 1}",
            legendgroup=f"k{k}",
            showlegend=False,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="MoE Layer",
            tickmode="array",
            tickvals=list(range(num_layers)),
            ticktext=[f"L{i}" for i in range(num_layers)],
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
        ),
        yaxis=dict(
            title="Expert",
            tickmode="array",
            tickvals=list(range(num_experts)),
            ticktext=[f"E{i}" for i in range(num_experts)],
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
            autorange="reversed",   # E0 at top, matching the diagram style
        ),
        height=max(350, 18 * num_experts),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


# ── Convenience wrappers ──────────────────────────────────────────────────────

def routing_path_from_analysis(data, token_idx: int) -> go.Figure:
    """Extract the right slice from RoutingData ([layers, seq, top_k])."""
    return routing_path_chart(
        expert_ids=data.expert_ids[:, token_idx, :],
        expert_weights=data.expert_weights[:, token_idx, :],
        num_experts=data.num_experts,
        title=f"Routing path — token {token_idx}: {repr(data.tokens[token_idx])}",
    )


def routing_path_from_generation(gd, token_idx: int) -> go.Figure:
    """Extract the right slice from GenerationData ([tokens, layers, top_k])."""
    token_label = repr(gd.all_tokens[token_idx])
    phase = "prompt" if token_idx < gd.num_prompt_tokens else "generated"
    return routing_path_chart(
        expert_ids=gd.expert_ids[token_idx],
        expert_weights=gd.expert_weights[token_idx],
        num_experts=gd.num_experts,
        title=f"Routing path — {phase} token {token_idx}: {token_label}",
    )
