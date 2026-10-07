"""Models we have measured end to end. `scores` = accuracy (%) from `localllm eval` keyed "lang/suite" (see bench.py).
gb = weights in GiB; kv_kb_per_token = KV cache per token at q8 (from the GGUF attention layout).
Speeds: llama-server decode tok/s on an RX 9070 XT 16 GB (Windows, Vulkan) with the tuned launch in runtime.py.
Add a model only after measuring it with `localllm eval`."""

MODELS = {
    "qwen3.8-27b-q3": {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-Q3_K_XL.gguf", "gb": 12.2,
        "kv_kb_per_token": 34.8, "fixed_cache_gb": 0.15, "max_ctx": 262144, "tok_s_9070xt": 50, "mtp": True,
        "cpu_mapped_gb": 0.51,  # measured: ~521 MiB of the 12.2 GiB model stays CPU-mapped (large 248k vocab)
        "scores": {"en/global": 81.5, "zh/global": 76.2, "zh/regional": 74.7, "es/global": 80.2, "es/regional": 76.8, "hi/global": 69.0, "hi/regional": 74.3, "ar/global": 70.8, "ar/regional": 71.2, "ja/global": 73.5, "ja/regional": 87.6, "th/regional": 67.1},
        "note": "dense 27B; built-in MTP head drafts 2 tokens",
    },
    "gemma4-26b-a4b-qat": {
        "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "file": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "gb": 13.3,
        "kv_kb_per_token": 10.9, "fixed_cache_gb": 0.11, "max_ctx": 262144, "tok_s_9070xt": 69, "mtp": False,
        "moe": {"layers": 30, "expert_gb_per_layer": 0.4},  # from the GGUF: 11.96 GiB of experts over 30 layers
        "scores": {"en/global": 82.2, "zh/global": 73.5, "zh/regional": 66.5, "es/global": 74.5, "es/regional": 75.2, "hi/global": 69.5, "hi/regional": 71.0, "ar/global": 71.5, "ar/regional": 73.6, "ja/global": 74.5, "ja/regional": 81.9, "th/regional": 65.7},
        "note": "MoE with ~4B active params: fastest",
    },
    "qwen3.8-27b-iq2": {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "file": "Qwen3.8-27B-UD-IQ2_S.gguf", "gb": 7.8,
        "kv_kb_per_token": 34.8, "fixed_cache_gb": 0.15, "max_ctx": 262144, "tok_s_9070xt": 40, "mtp": False,
        "scores": {"en/global": 74.2, "zh/global": 67.8, "zh/regional": 67.8, "es/global": 70.8, "es/regional": 69.2, "hi/global": 56.2, "hi/regional": 55.5, "ar/global": 60.8, "ar/regional": 57.2, "ja/global": 65.8, "ja/regional": 77.9, "th/regional": 54.2},
        "note": "for 10-12 GB cards only: 2-bit costs 8-13 points, most in Hindi, Arabic, Thai",
    },
}


def score(key: str, lang: str | None = None) -> float:
    """Mean accuracy over the user's language tests if we have them, else over everything measured."""
    s = MODELS[key]["scores"]
    mine = [v for k, v in s.items() if lang and k.startswith(lang + "/")]
    pool = mine or list(s.values())
    return sum(pool) / len(pool) if pool else 0.0


MIN_CTX = 8192   # a model only "fits" if it also leaves room for an 8k-token conversation


def fits(key: str, vram_gb: float) -> bool:
    from .sizing import context_tokens
    return context_tokens(vram_gb, MODELS[key]) >= MIN_CTX


def cpu_moe_layers(key: str, vram_gb: float, ram_free_gb: float) -> int | None:
    """MoE models that don't fit the card whole: how many layers' experts to keep in RAM (llama.cpp --n-cpu-moe)
    so the rest plus an 8k context fit on the GPU. 0 = fits whole; None = not possible on this PC."""
    from .sizing import context_tokens
    m = MODELS[key]
    if fits(key, vram_gb):
        return 0
    if "moe" not in m:
        return None
    per = m["moe"]["expert_gb_per_layer"]
    for n in range(1, m["moe"]["layers"] + 1):
        if n * per > 0.7 * ram_free_gb:
            return None
        if context_tokens(vram_gb, {**m, "gb": m["gb"] - n * per}) >= MIN_CTX:
            return n
    return None


TIE_POINTS = 2.0   # accuracy gaps this small are inside the benchmark's margin: prefer the faster model


OFFLOAD_SPEED = 0.5   # placeholder share of full-GPU speed with experts in RAM; replaced by measurements per model


def speed(key: str, vram_gb: float, ram_free_gb: float = 0.0) -> float:
    """Expected decode tok/s on this card: measured speed, scaled down when experts have to stay in RAM."""
    m = MODELS[key]
    n = cpu_moe_layers(key, vram_gb, ram_free_gb) or 0
    if not n:
        return m["tok_s_9070xt"]
    measured = m.get("tok_s_offload", {})          # {layers in RAM: tok/s} measured on the reference card
    if measured:
        nearest = min(measured, key=lambda k: abs(int(k) - n))
        return measured[nearest]
    return m["tok_s_9070xt"] * OFFLOAD_SPEED


def pick(vram_gb: float, lang: str | None = None, ram_free_gb: float = 0.0) -> str | None:
    """Most accurate model (for `lang` when measured) that runs on this PC - whole on the GPU, or a MoE with some
    experts in RAM - with an 8k context; near-ties go to the faster one."""
    ok = [k for k in MODELS if cpu_moe_layers(k, vram_gb, ram_free_gb) is not None]
    if not ok:
        return None
    best = max(score(k, lang) for k in ok)
    return max((k for k in ok if score(k, lang) >= best - TIE_POINTS), key=lambda k: speed(k, vram_gb, ram_free_gb))
