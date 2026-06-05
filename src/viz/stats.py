from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.extractor import RoutingData
from src.analysis.metrics import RoutingMetrics


def expert_load_chart(metrics: RoutingMetrics, layer_idx: int) -> go.Figure:
    """Bar chart: fraction of tokens routed to each expert for a given layer."""
    load = metrics.expert_load[layer_idx]
    num_experts = len(load)
    colors = [f"hsl({(i * 37) % 360}, 65%, 55%)" for i in range(num_experts)]
    fig = go.Figure(
        go.Bar(
            x=[f"E{i}" for i in range(num_experts)],
            y=load,
            marker_color=colors,
            hovertemplate="Expert %{x}<br>Load: %{y:.2%}<extra></extra>",
        )
    )
    uniform = 1.0 / num_experts
    fig.add_hline(
        y=uniform,
        line_dash="dash",
        line_color="gray",
        annotation_text="uniform",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Expert Load — Layer {layer_idx}",
        xaxis_title="Expert",
        yaxis_title="Fraction of tokens",
        yaxis_tickformat=".1%",
        height=350,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def coactivation_heatmap(metrics: RoutingMetrics) -> go.Figure:
    """Symmetric heatmap of expert pair co-activation frequency."""
    matrix = metrics.coactivation
    n = matrix.shape[0]
    labels = [f"E{i}" for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            colorbar=dict(title="Co-activation rate"),
            hovertemplate="Expert %{x} + %{y}<br>Rate: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Expert Co-activation Matrix",
        height=max(300, 12 * n),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def load_across_layers(metrics: RoutingMetrics, num_experts: int) -> go.Figure:
    """
    Stacked area chart: expert load fractions across all MoE layers.
    Shows how routing distribution shifts with depth.
    """
    num_layers = metrics.expert_load.shape[0]
    layer_labels = [f"L{i}" for i in range(num_layers)]

    fig = go.Figure()
    for expert in range(num_experts):
        fig.add_trace(
            go.Scatter(
                x=layer_labels,
                y=metrics.expert_load[:, expert],
                name=f"E{expert}",
                mode="lines",
                stackgroup="one",
                hovertemplate=f"Expert {expert}<br>Layer %{{x}}<br>Load: %{{y:.2%}}<extra></extra>",
                line=dict(width=0.5),
                fillcolor=f"hsla({(expert * 37) % 360}, 65%, 55%, 0.7)",
            )
        )
    fig.update_layout(
        title="Expert Load Across Layers",
        xaxis_title="MoE Layer",
        yaxis_title="Fraction of tokens",
        yaxis_tickformat=".0%",
        height=400,
        showlegend=num_experts <= 16,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig
