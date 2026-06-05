from __future__ import annotations

import plotly.graph_objects as go

from src.analysis.extractor import RoutingData


def routing_sankey(data: RoutingData, token_idx: int) -> go.Figure:
    """
    Sankey diagram showing how a single token is routed through every MoE layer.
    Nodes: (layer, expert_id) pairs.  Edges: routing weight between adjacent layers.
    """
    num_layers = data.num_layers
    token_label = repr(data.tokens[token_idx])

    # Build node list: one node per (layer, expert) combination that is selected
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[int, int], int] = {}

    def _get_node(layer: int, expert: int) -> int:
        key = (layer, expert)
        if key not in node_index:
            node_index[key] = len(node_labels)
            node_labels.append(f"L{layer}·E{expert}")
            node_colors.append(f"hsl({(expert * 37) % 360}, 70%, 55%)")
        return node_index[key]

    sources, targets, values, link_labels = [], [], [], []

    for layer in range(num_layers):
        experts = data.expert_ids[layer, token_idx]
        weights = data.expert_weights[layer, token_idx]

        if layer < num_layers - 1:
            next_experts = data.expert_ids[layer + 1, token_idx]
            next_weights = data.expert_weights[layer + 1, token_idx]
            for e_src, w_src in zip(experts, weights):
                for e_dst, w_dst in zip(next_experts, next_weights):
                    src = _get_node(layer, int(e_src))
                    dst = _get_node(layer + 1, int(e_dst))
                    sources.append(src)
                    targets.append(dst)
                    values.append(float(w_src * w_dst))
                    link_labels.append(
                        f"L{layer}·E{int(e_src)} → L{layer+1}·E{int(e_dst)}<br>"
                        f"combined weight: {w_src * w_dst:.4f}"
                    )
        else:
            # Terminal layer: create self-loops so the nodes are still visible
            for e, w in zip(experts, weights):
                node = _get_node(layer, int(e))
                node_labels[node]  # ensure created
                # Add a small self-referencing value so the node renders
                sources.append(node)
                targets.append(node)
                values.append(float(w) * 0.01)
                link_labels.append(f"Final: L{layer}·E{int(e)} w={w:.3f}")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                label=node_labels,
                color=node_colors,
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=link_labels,
                hovertemplate="%{label}<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=f"Token routing path — token {token_idx}: {token_label}",
        height=max(400, 80 * num_layers),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig
