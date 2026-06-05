import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    model_id: str = field(
        default_factory=lambda: os.getenv("MODEL_ID", "microsoft/Phi-tiny-MoE-instruct")
    )
    load_in_quantized: bool = field(
        default_factory=lambda: os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    )
    # None = auto-detect (mps → cuda → cpu)
    device: str | None = field(
        default_factory=lambda: os.getenv("DEVICE") or None
    )

    SUPPORTED_MODELS = {
        # ~7.6 GB bfloat16 — fits 16 GB M3 without quantization
        "microsoft/Phi-tiny-MoE-instruct": {"num_experts": 16, "top_k": 2, "label": "Phi-tiny-MoE 3.8B (recommended for 16 GB)"},
        # ~14 GB bfloat16 — needs int8 quantization on 16 GB M3
        "allenai/OLMoE-1B-7B-0924": {"num_experts": 64, "top_k": 8, "label": "OLMoE 1B/7B (64 experts, needs quant)"},
        # ~26 GB int4 — requires 36 GB+ hardware
        "mistralai/Mixtral-8x7B-v0.1": {"num_experts": 8, "top_k": 2, "label": "Mixtral 8×7B (needs 36 GB+)"},
        "mistralai/Mixtral-8x22B-v0.1": {"num_experts": 8, "top_k": 2, "label": "Mixtral 8×22B (needs 64 GB+)"},
    }
