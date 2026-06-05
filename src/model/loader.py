import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

_cache: dict = {}


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(config: Config) -> tuple:
    """Load model + tokenizer, apply quantization, cache across Streamlit reruns."""
    key = (config.model_id, config.load_in_quantized, config.device)
    if key in _cache:
        return _cache[key]

    device = config.device or detect_device()
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32

    if device == "cuda" and config.load_in_quantized:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                quantization_config=bnb,
                device_map="auto",
            )
        except Exception:
            # bitsandbytes not available — fall back to plain float16
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id, torch_dtype=dtype, device_map="auto"
            )
    elif device == "mps" and config.load_in_quantized:
        # Load in float16 first (7B × 2 bytes = ~14 GB peak), then quantize
        # to int8 (~7 GB) before moving to MPS.  low_cpu_mem_usage avoids
        # the 2× memory spike that from_pretrained normally causes.
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
        except Exception:
            pass  # torchao unavailable — stay in float16
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=dtype, device_map=device
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    _cache[key] = (model, tokenizer)
    return model, tokenizer


def clear_cache() -> None:
    _cache.clear()
