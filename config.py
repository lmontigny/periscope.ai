import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    model_id: str = field(
        default_factory=lambda: os.getenv("MODEL_ID", "allenai/OLMoE-1B-7B-0924")
    )
    load_in_quantized: bool = field(
        default_factory=lambda: os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    )
    # None = auto-detect (mps → cuda → cpu)
    device: str | None = field(
        default_factory=lambda: os.getenv("DEVICE") or None
    )

    SUPPORTED_MODELS = {
        "allenai/OLMoE-1B-7B-0924": {"num_experts": 64, "top_k": 8, "label": "OLMoE 1B/7B"},
        "mistralai/Mixtral-8x7B-v0.1": {"num_experts": 8, "top_k": 2, "label": "Mixtral 8×7B"},
        "mistralai/Mixtral-8x22B-v0.1": {"num_experts": 8, "top_k": 2, "label": "Mixtral 8×22B"},
    }
