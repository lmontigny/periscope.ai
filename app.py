"""
periscope.ai — MoE interpretability dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
import torch

from config import Config
from src.model.loader import detect_device, load_model
from src.analysis.extractor import RoutingData, extract
from src.analysis.metrics import compute_metrics
from src.viz.heatmap import expert_assignment_heatmap, entropy_heatmap
from src.viz.flow import routing_sankey
from src.viz.stats import coactivation_heatmap, expert_load_chart, load_across_layers
from src.viz.umap_plot import routing_umap_scatter
from src.viz.generation_plot import generation_heatmap, generation_entropy_chart

st.set_page_config(
    page_title="periscope.ai — MoE Interpretability",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("periscope.ai")
    st.caption("MoE routing interpretability")

    st.divider()
    st.subheader("Model")

    model_options = {
        "Phi-tiny-MoE 3.8B — fits 16 GB, no quant needed": "microsoft/Phi-tiny-MoE-instruct",
        "OLMoE 1B/7B — 64 experts, needs int8 quant on 16 GB": "allenai/OLMoE-1B-7B-0924",
        "Mixtral 8×7B — needs 36 GB+": "mistralai/Mixtral-8x7B-v0.1",
        "Mixtral 8×22B — needs 64 GB+": "mistralai/Mixtral-8x22B-v0.1",
    }
    model_label = st.selectbox("Model", list(model_options.keys()))
    model_id = model_options[model_label]

    quantize = st.toggle("Quantize (int8 on MPS, int4 on CUDA)", value=False)

    auto_device = detect_device()
    st.caption(f"Detected device: **{auto_device}**")

    if st.button("Load / reload model", type="primary"):
        st.session_state.pop("model", None)
        st.session_state.pop("tokenizer", None)
        st.session_state.pop("routing_data", None)
        st.session_state.pop("metrics", None)
        st.session_state["load_requested"] = True

    st.divider()
    st.caption("After loading, paste text in the **Input** tab and click **Analyze**.")


# ── Model loading ──────────────────────────────────────────────────────────────

if "model" not in st.session_state or st.session_state.get("load_requested"):
    st.session_state.pop("load_requested", None)
    cfg = Config(model_id=model_id, load_in_quantized=quantize)
    with st.spinner(f"Loading {model_label} …"):
        try:
            model, tokenizer = load_model(cfg)
            st.session_state["model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.session_state["model_label"] = model_label
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

model = st.session_state["model"]
tokenizer = st.session_state["tokenizer"]

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_input, tab_routing, tab_stats, tab_gen = st.tabs(["Input", "Token Routing", "Expert Stats", "Generation"])

# ── Tab 1: Input ───────────────────────────────────────────────────────────────

with tab_input:
    st.header("Analyze text")
    st.caption(
        "Enter any text and click **Analyze** to run a forward pass and extract "
        "per-token expert routing data."
    )

    default_text = (
        "The transformer architecture has revolutionized natural language processing "
        "by enabling models to capture long-range dependencies through attention."
    )
    text = st.text_area("Input text", value=default_text, height=120)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("Analyze", type="primary", disabled=not text.strip())

    if run and text.strip():
        with st.spinner("Running forward pass …"):
            try:
                routing_data = extract(text, model, tokenizer)
                metrics = compute_metrics(routing_data)
                st.session_state["routing_data"] = routing_data
                st.session_state["metrics"] = metrics
                st.success(
                    f"Done — {routing_data.seq_len} tokens, "
                    f"{routing_data.num_layers} MoE layers, "
                    f"{routing_data.num_experts} experts (top-{routing_data.top_k})"
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if "routing_data" in st.session_state:
        rd: RoutingData = st.session_state["routing_data"]
        st.subheader("Tokens")
        st.write(" | ".join(f"`{t}`" for t in rd.tokens))

# ── Tab 2: Token Routing ───────────────────────────────────────────────────────

with tab_routing:
    st.header("Token routing")

    if "routing_data" not in st.session_state:
        st.info("Run an analysis on the **Input** tab first.")
    else:
        rd: RoutingData = st.session_state["routing_data"]
        m = st.session_state["metrics"]

        st.subheader("Expert assignment heatmap")
        st.caption("Color = top-1 expert.  Hover for full top-k detail.")
        st.plotly_chart(expert_assignment_heatmap(rd), use_container_width=True)

        st.subheader("Routing entropy")
        st.caption(
            "High entropy → router is uncertain (weight spread across many experts).  "
            "Low entropy → strong preference for one expert."
        )
        st.plotly_chart(entropy_heatmap(rd, m.entropy), use_container_width=True)

        st.subheader("Token routing path (Sankey)")
        token_idx = st.slider(
            "Token index",
            min_value=0,
            max_value=rd.seq_len - 1,
            value=0,
            format="%d",
            help="Select which token's routing path to visualise across all MoE layers.",
        )
        st.caption(f"Selected token: `{rd.tokens[token_idx]}`")
        st.plotly_chart(routing_sankey(rd, token_idx), use_container_width=True)

        st.divider()
        st.subheader("Routing signature UMAP")
        st.caption(
            "Each token is projected to 2D based on its full routing signature "
            "(softmax probabilities across all layers and experts). "
            "Tokens that cluster together make similar routing decisions throughout the network. "
            "Works best with 20+ tokens."
        )

        umap_col1, umap_col2 = st.columns([1, 3])
        with umap_col1:
            color_by = st.radio(
                "Color by",
                options=["position", "entropy", "top_expert"],
                format_func=lambda x: {
                    "position": "Token position",
                    "entropy": "Routing entropy",
                    "top_expert": "Top expert (layer)",
                }[x],
            )
            umap_layer = 0
            if color_by == "top_expert":
                umap_layer = st.slider("Layer", 0, rd.num_layers - 1, 0)

        with umap_col2:
            if rd.seq_len < 4:
                st.warning("Need at least 4 tokens. Use a longer input text.")
            else:
                with st.spinner("Computing UMAP…"):
                    try:
                        from src.analysis.embedding import compute_umap
                        embedding = compute_umap(rd)
                        st.plotly_chart(
                            routing_umap_scatter(rd, embedding, m, color_by, umap_layer),
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"UMAP failed: {e}")

# ── Tab 3: Expert Stats ────────────────────────────────────────────────────────

with tab_stats:
    st.header("Expert statistics")

    if "routing_data" not in st.session_state:
        st.info("Run an analysis on the **Input** tab first.")
    else:
        rd: RoutingData = st.session_state["routing_data"]
        m = st.session_state["metrics"]

        st.subheader("Expert load across all layers")
        st.caption("Stacked area — each colour is one expert.  Uniform line = perfect balance.")
        st.plotly_chart(load_across_layers(m, rd.num_experts), use_container_width=True)

        st.subheader("Expert load for a single layer")
        layer_idx = st.slider(
            "Layer index",
            min_value=0,
            max_value=rd.num_layers - 1,
            value=0,
        )
        st.plotly_chart(expert_load_chart(m, layer_idx), use_container_width=True)

        st.subheader("Expert co-activation matrix")
        st.caption(
            "How often each pair of experts is selected together for the same token.  "
            "Dark cells = frequently co-activated."
        )
        st.plotly_chart(coactivation_heatmap(m), use_container_width=True)

# ── Tab 4: Generation ─────────────────────────────────────────────────────────

with tab_gen:
    st.header("Generation mode")
    st.caption(
        "Watch how expert routing evolves token by token as the model generates. "
        "Blue dots = prompt tokens, red dots = generated tokens."
    )

    gen_prompt = st.text_area(
        "Prompt",
        value="The key difference between transformers and recurrent networks is",
        height=80,
    )

    gcol1, gcol2 = st.columns(2)
    with gcol1:
        max_new = st.slider("Max new tokens", min_value=10, max_value=150, value=40, step=5)
    with gcol2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1,
                                help="0 = greedy (deterministic).  >0 adds randomness.")

    if st.button("Generate", type="primary", key="gen_btn"):
        st.session_state.pop("gen_data", None)
        progress_bar = st.progress(0, text="Generating…")

        def _progress(step, total):
            progress_bar.progress(
                min(step / max(total, 1), 1.0),
                text=f"Generating token {step}/{total}…",
            )

        try:
            from src.analysis.generation import generate_with_routing
            gen_data = generate_with_routing(
                gen_prompt, model, tokenizer,
                max_new_tokens=max_new,
                temperature=temperature,
                progress_fn=_progress,
            )
            st.session_state["gen_data"] = gen_data
            progress_bar.empty()
        except Exception as e:
            progress_bar.empty()
            st.error(f"Generation failed: {e}")

    if "gen_data" in st.session_state:
        gd = st.session_state["gen_data"]

        st.subheader("Generated text")
        prompt_part = "".join(gd.prompt_tokens)
        gen_part = "".join(gd.generated_tokens)
        st.markdown(f"{prompt_part}**{gen_part}**")

        st.divider()
        st.subheader("Routing replay")
        st.caption("Drag the slider to step through the generation token by token.")
        replay_step = st.slider(
            "Show routing up to token",
            min_value=1,
            max_value=gd.total_tokens,
            value=gd.total_tokens,
            key="replay_slider",
        )

        st.plotly_chart(
            generation_heatmap(gd, show_up_to=replay_step),
            use_container_width=True,
        )

        st.subheader("Routing entropy over time")
        st.caption(
            "Mean routing entropy per token across all MoE layers. "
            "Spikes = the router is uncertain which experts to use for that token."
        )
        st.plotly_chart(
            generation_entropy_chart(gd, show_up_to=replay_step),
            use_container_width=True,
        )
