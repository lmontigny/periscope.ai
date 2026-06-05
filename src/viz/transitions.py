from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.analysis.extractor import RoutingData
from src.analysis.transitions import TransitionData


def token_stability_heatmap(data: RoutingData, td: TransitionData) -> go.Figure:
    """
    Heatmap: x = layer transition, y = token.
    Blue = expert stayed the same, red = expert switched.
    """
    tokens = [t.replace(" ", "·") for t in data.tokens]

    # changed is [layers-1, seq] — transpose to [seq, layers-1] for the heatmap
    z = td.changed.T

    hover = []
    for tok in range(data.seq_len):
        row = []
        for i, label in enumerate(td.layer_labels):
            switched = bool(td.changed[i, tok])
            e_before = data.expert_ids[i, tok, 0]
            e_after  = data.expert_ids[i + 1, tok, 0]
            row.append(
                f"Token: {repr(data.tokens[tok])}<br>"
                f"{label}: E{e_before} → E{e_after} "
                f"({'switched' if switched else 'same'})"
            )
        hover.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=td.layer_labels,
            y=tokens,
            text=np.array(hover),
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0, "#bde0fe"], [1, "#e63946"]],
            zmin=0, zmax=1,
            showscale=True,
            colorbar=dict(
                title="Switched",
                tickvals=[0, 1],
                ticktext=["same", "switched"],
                thickness=12,
            ),
        )
    )
    fig.update_layout(
        title="Expert switch per token per layer transition",
        xaxis_title="Layer transition",
        yaxis_title="Token",
        height=max(280, 22 * data.seq_len),
        margin=dict(l=80, r=20, t=60, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig


def switch_rate_chart(td: TransitionData) -> go.Figure:
    """
    Bar chart: fraction of tokens that switched their top expert at each layer boundary.
    """
    fig = go.Figure(
        go.Bar(
            x=td.layer_labels,
            y=td.switch_rate,
            marker_color=[
                f"hsl({int(r * 30)}, 75%, 50%)" for r in td.switch_rate
            ],
            hovertemplate="%{x}<br>Switch rate: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=td.switch_rate.mean(),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"mean {td.switch_rate.mean():.1%}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Expert switch rate across layer transitions",
        xaxis_title="Layer transition",
        yaxis_title="Fraction of tokens that switched",
        yaxis_tickformat=".0%",
        height=300,
        margin=dict(l=60, r=20, t=60, b=80),
        xaxis=dict(tickangle=-45),
        plot_bgcolor="#f8f9fa",
    )
    return fig


def transition_matrix_chart(td: TransitionData, layer_pair_idx: int) -> go.Figure:
    """
    Heatmap of expert-to-expert transitions at a given layer boundary.
    Diagonal = token kept the same expert. Off-diagonal = expert changed.
    """
    matrix = td.transition_matrices[layer_pair_idx]
    n = matrix.shape[0]
    labels = [f"E{i}" for i in range(n)]
    label = td.layer_labels[layer_pair_idx]

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Fraction", thickness=12),
            hovertemplate="From %{y} → To %{x}<br>Rate: %{z:.3f}<extra></extra>",
        )
    )
    # Highlight diagonal (same expert retained)
    diag_vals = np.diag(matrix)
    fig.add_trace(go.Scatter(
        x=labels, y=labels,
        mode="markers",
        marker=dict(symbol="square", size=6, color="rgba(0,0,0,0)",
                    line=dict(width=2, color="#e63946")),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Expert transition matrix — {label}",
        xaxis_title="Expert at next layer",
        yaxis_title="Expert at this layer",
        height=max(300, 14 * n),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig
