# periscope.ai

A local interpretability tool for visualizing how Mixture-of-Experts (MoE) language models route tokens through their expert layers.

![periscope.ai dashboard](docs/screenshot.png)

## What is MoE routing?

In a Mixture-of-Experts LLM, each transformer layer contains multiple independent feed-forward networks called **experts**. A lightweight **router** (a single linear layer) decides, for every token, which top-k experts should process it. Only those k experts run — the rest are skipped.

This sparse activation is what makes large MoE models efficient: a model like Phi-tiny-MoE has 3.8B total parameters but only activates ~1.1B per token. Each layer has its own router, so the same token can take a completely different path through experts at different depths.

## Views

### Token Routing tab
| View | What it shows |
|------|--------------|
| **Expert assignment heatmap** | Which expert each token is routed to, at every MoE layer — color = top-1 expert, hover for full top-k detail |
| **Routing entropy heatmap** | How confident the router is per token per layer — low = one expert dominates, high = weight is spread across experts |
| **Routing path grid** | Grid of experts × layers for a single token: faded dots = inactive experts, bold dots = selected experts (size ∝ weight), lines trace the path across depth |
| **Routing signature UMAP** | 2D projection of each token's full routing signature (softmax probabilities across all layers) — tokens that cluster share routing behaviour throughout the network |

### Expert Stats tab
| View | What it shows |
|------|--------------|
| **Expert load across layers** | Stacked area chart of how token load is distributed across experts at every layer |
| **Expert load (single layer)** | Bar chart for a chosen layer — dashed line = perfect uniform load |
| **Co-activation matrix** | How often each expert pair is selected together for the same token |
| **Token stability heatmap** | Which tokens switch their top expert between adjacent layers (blue = same, red = switched) |
| **Switch rate chart** | Fraction of tokens that changed expert at each layer boundary — spikes reveal where the model reorganises routing |
| **Expert transition matrix** | For a chosen layer boundary: expert-to-expert handoff frequencies — diagonal = expert retained, off-diagonal = systematic handoffs |

### Generation tab
| View | What it shows |
|------|--------------|
| **Routing heatmap** | Expert assignments for the full prompt + generated sequence; dashed line marks where generation starts |
| **Replay slider** | Step through the generation token by token to watch routing evolve |
| **Entropy over time** | Mean routing entropy per token — spikes = router uncertain, flat low = confident routing |
| **Routing path grid** | Per-token expert path for any chosen prompt or generated token |

## Supported models

| Model | Total params | Active params | Memory | Notes |
|-------|-------------|---------------|--------|-------|
| `microsoft/Phi-tiny-MoE-instruct` | 3.8B | 1.1B | ~8 GB bfloat16 | **Default — fits 16 GB M-series Mac, no quant needed** |
| `allenai/OLMoE-1B-7B-0924` | 7B | 1B | ~7 GB int8 | 64 experts, top-8 — richer routing patterns |
| `mistralai/Mixtral-8x7B-v0.1` | 46.7B | ~12B | ~26 GB int4 | Requires 36 GB+ |

Change model by editing `MODEL_ID` in `.env`.

## Setup

```bash
git clone https://github.com/lmontigny/periscope.ai
cd periscope.ai

uv venv
uv pip install -r requirements.txt

cp .env.example .env   # optionally edit MODEL_ID
```

## Run

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).  
The model downloads from HuggingFace on first run (~8 GB for the default).

## Usage

1. **Input tab** — paste any text and click **Analyze** to run a forward pass and extract routing data for every token.
2. **Token Routing tab** — explore the expert assignment heatmap, routing entropy, per-token routing path grid, and the UMAP projection. Use the token slider to compare routing paths across tokens.
3. **Expert Stats tab** — inspect load balance, expert co-activation, and the layer transition heatmaps. The transition matrix shows how experts "hand off" tokens between adjacent layers.
4. **Generation tab** — enter a prompt, set max tokens and temperature, then click **Generate**. Use the replay slider to step through the generation and watch routing evolve token by token.

## How it works

Routing data is captured via PyTorch **forward hooks** registered directly on the router linear layer inside each MoE block. This works for all supported architectures regardless of their return format:

- **OLMoE** — router at `layers[i].mlp.gate`
- **PhiMoE** — router at `layers[i].mlp.router` (returns a tuple; hook takes `output[0]`)
- **Mixtral** — router at `layers[i].block_sparse_moe.gate`

For the **Analyze** flow, a single forward pass (no generation) captures routing for the full input sequence. For **Generation** mode, a prefill pass captures all prompt tokens, then each decode step (KV-cached) captures the newly generated token.

No model weights are modified. Hooks are removed immediately after each run.
