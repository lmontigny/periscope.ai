from __future__ import annotations

import plotly.graph_objects as go

from src.analysis.extractor import RoutingData


def next_token_predictions_chart(data: RoutingData, token_idx: int) -> go.Figure:
    """
    Horizontal bar chart of the top-5 next-token predictions after token_idx.
    Shows what the model predicts will follow the selected token given full context.
    """
    entries = data.next_token_topk[token_idx]          # [(token_str, prob), ...]
    # Highest probability at the top
    entries = sorted(entries, key=lambda x: x[1])
    labels = [repr(t) for t, _ in entries]
    probs   = [p for _, p in entries]

    colors = [
        f"hsl(265, {40 + int(p * 200)}%, {65 - int(p * 30)}%)"
        for p in probs
    ]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )

    source_token = repr(data.tokens[token_idx])
    next_actual = repr(data.tokens[token_idx + 1]) if token_idx + 1 < data.seq_len else "—"

    fig.update_layout(
        title=f"Next-token predictions after {source_token}  (actual next: {next_actual})",
        xaxis=dict(
            title="Probability",
            tickformat=".0%",
            range=[0, max(probs) * 1.25],
        ),
        yaxis=dict(title=""),
        height=260,
        margin=dict(l=80, r=60, t=60, b=40),
        plot_bgcolor="#f8f9fa",
        showlegend=False,
    )
    return fig
