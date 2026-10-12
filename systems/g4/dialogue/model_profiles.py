"""
Model profiles for comparative experiments across Qwen model sizes.

Each profile encapsulates ALL parameters that differ between model sizes:
  - Model identifier (HuggingFace path)
  - Token budgets (context window, max output, memory limits)
  - vLLM serving parameters (GPU memory, quantization, max_model_len)
  - Agent budgets (steps, wallclock) scaled to model capability

Usage:
    # In .env or shell:
    THESIS_MODEL_PROFILE=32b

    # In code:
    from model_profiles import get_profile, apply_profile
    profile = get_profile("32b")
    config = apply_profile(load_config(), profile)

    # Or CLI-style:
    THESIS_MODEL_PROFILE=32b bash scripts/run_single.sh hello-world

Design rationale:
    Fair comparison requires that each model gets proportional resources.
    A 7B model doesn't need 32K context (it can't use it well), but giving
    it only 8K when the 32B gets 32K would be unfair.  The profiles below
    use the maximum context each model can physically support on its
    target GPU, so the comparison measures *capability* not *configuration*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    """Complete configuration profile for a specific model size."""

    # ---- Identity ----
    name: str                          # Human-readable: "7b", "14b", "32b", "72b"
    description: str                   # For logging/reports

    # ---- Model ----
    model_id: str                      # HuggingFace model path
    quantization: Optional[str]        # "awq", "gptq", None (fp16/bf16)

    # ---- vLLM Serving ----
    gpu: str                           # Target GPU: "T4", "A100-40GB", "A100-80GB"
    max_model_len: int                 # vLLM --max-model-len
    gpu_memory_utilization: float      # vLLM --gpu-memory-utilization
    dtype: str = "auto"                # vLLM --dtype
    tensor_parallel: int = 1           # vLLM --tensor-parallel-size

    # ---- Generation ----
    max_tokens: int = 2048             # Max output tokens per LLM call
    temperature: float = 0.0           # Deterministic for fair comparison

    # ---- Agent Context Budget ----
    # Formula: max_context_tokens = max_model_len - max_tokens - safety_margin
    # The agent trims conversation to fit within this budget.
    max_context_tokens: int = 28000

    # ---- Agent Execution Budget ----
    # Scaled to model capability — smaller models need fewer steps
    # (they can't productively use 60 steps with limited reasoning)
    max_steps: int = 60
    max_wallclock_seconds: int = 600

    # ---- Per-call LLM read-timeout ----
    # Upper bound (seconds) for a SINGLE LLM HTTP call. Must exceed the
    # slowest legitimate generation so slower models (e.g. Devstral) are
    # not killed mid-thought. The agent-level max_wallclock_seconds remains
    # the real task budget; this is only a hung-connection ceiling.
    llm_timeout_seconds: int = 300

    # ---- Token Estimation ----
    # Qwen models with 151K vocab average ~4.0 chars/token for code
    chars_per_token: float = 4.0
    token_safety_factor: float = 1.20

    # ---- Tool Calling ----
    tool_calling: str = "prompt"       # "prompt" for fair comparison across sizes

    # ---- Provider (for non-vLLM models) ----
    provider: Optional[str] = None     # None = use env default; "openai", "vllm", etc.
    api_base: Optional[str] = None     # None = use env default
    api_key_env: Optional[str] = None  # Env var name holding the API key (e.g. "GEMINI_API_KEY")

    # ---- Reasoning / Thinking (Qwen3 hybrid models) ----
    # None  = send nothing (server default) — zero effect for non-Qwen3 models
    # True  = force reasoning ON   (per-request chat_template_kwargs)
    # False = force reasoning OFF
    enable_thinking: Optional[bool] = None


# ============================================================
# Profile Definitions
# ============================================================

PROFILES: dict[str, ModelProfile] = {
    # ---- Qwen 2.5 Coder 7B (AWQ) — T4 16GB ----
    "7b": ModelProfile(
        name="7b",
        description="Qwen2.5-Coder-7B-Instruct-AWQ on T4 16GB",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        quantization="awq",
        gpu="T4",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        max_tokens=1024,
        max_context_tokens=14000,    # 16384 - 1024 - 1360 margin
        max_steps=60,                # uniform 60-step budget (was 40 for 7B)
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen 2.5 Coder 14B (AWQ) — A100 40GB ----
    "14b": ModelProfile(
        name="14b",
        description="Qwen2.5-Coder-14B-Instruct-AWQ on A100 40GB",
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
        quantization="awq",          # AWQ 4-bit — consistent with 7b/32b/72b
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.92,
        max_tokens=2048,
        max_context_tokens=28000,    # 32768 - 2048 - 2720 margin
        max_steps=60,                # uniform 60-step budget (was 50 for 14B)
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen 2.5 Coder 32B (AWQ) — A100 40GB ----
    "32b": ModelProfile(
        name="32b",
        description="Qwen2.5-Coder-32B-Instruct-AWQ on A100 40GB",
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,    # 32768 - 2048 - 2720 margin
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen 2.5 Coder 32B (fp16) — A100 80GB ----
    "32b-fp16": ModelProfile(
        name="32b-fp16",
        description="Qwen2.5-Coder-32B-Instruct (fp16) on A100 80GB",
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization=None,
        gpu="A100-80GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen 2.5 72B (AWQ) — A100 80GB ----
    # NOTE: No "Coder" variant exists for 72B — using the general Qwen2.5-72B
    "72b": ModelProfile(
        name="72b",
        description="Qwen2.5-72B-Instruct-AWQ on A100 80GB (no Coder variant)",
        model_id="Qwen/Qwen2.5-72B-Instruct-AWQ",
        quantization="awq",
        gpu="A100-80GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),


    # ---- Llama 3.3 70B — Jean-Zay 2× A100-80GB (bf16) ----
    "llama-3.3-70b": ModelProfile(
        name="llama-3.3-70b",
        description="Llama-3.3-70B-Instruct (bf16) on 2× A100-80GB",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        quantization=None,
        gpu="2×A100-80GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        tensor_parallel=2,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Mistral-Nemo-12B (AWQ) — A100 40GB ----
    # casperhansen/mistral-nemo-instruct-2407-awq — standard text-only, no VLM issues.
    # tekken.json tokenizer included → --tokenizer-mode mistral in vLLM.
    # 12B AWQ ≈ 6GB weights → fits on A100 40GB (Colab). Apache 2.0.
    "mistral-nemo-12b": ModelProfile(
        name="mistral-nemo-12b",
        description="Mistral-Nemo-Instruct-2407 (AWQ 4-bit, bartowski) on 1× A100-40GB",
        model_id="casperhansen/mistral-nemo-instruct-2407-awq",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- DeepSeek-R1-Distill-Qwen-32B — Jean-Zay 1× A100-80GB (bf16) ----
    # Reasoning model: generates <think>...</think> tokens before answering.
    # Extra max_tokens budget to accommodate thinking tokens.
    "deepseek-r1-32b": ModelProfile(
        name="deepseek-r1-32b",
        description="DeepSeek-R1-Distill-Qwen-32B (bf16) on 1× A100-80GB",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quantization=None,
        gpu="A100-80GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=4096,               # larger budget for <think>...</think> tokens
        max_context_tokens=26000,      # 32768 - 4096 - 2672 margin
        max_steps=60,
        max_wallclock_seconds=1800,     # aligned to other large models (20min); nudge-only budget
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),


    # ---- Devstral-Small-2-24B-Instruct-2512 (AWQ) — A100 40GB ----
    # Agent-optimized model: fine-tuned from Mistral-Small-3.x-24B by Mistral AI + All Hands AI.
    # SWE-bench Verified: 68.0% (open-weight SOTA at this size; Dec 2025). Was 53.6% on 2507.
    # mistral3 arch + Llama4-style rope-scaling & scalable-softmax → needs a RECENT vLLM
    # (Dec-2025 era; model card recommends nightly) AND --tokenizer-mode mistral.
    # 24B AWQ ≈ 12-14GB weights → fits on A100 40GB. Apache 2.0. Native ctx 256K (capped 32K below).
    "devstral-small": ModelProfile(
        name="devstral-small",
        description="Devstral-Small-2-24B-Instruct-2512 (AWQ 4-bit, cyankiwi) on 1× A100-40GB",
        model_id="cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit",
        quantization="compressed-tensors",  # checkpoint is compressed-tensors (pack-quantized W4 g32), NOT plain awq
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Phi-4 (AWQ) — A100 40GB ----
    # Microsoft Phi-4 (14B dense, Dec 2024). 400+ citation tech report (arxiv:2412.08905).
    # SOTA for its size on reasoning/math/code benchmarks. MIT license.
    # Uses phi3 architecture → vLLM 0.6.6 compatible (same install as Qwen/DeepSeek).
    # 14B AWQ ≈ 7GB weights → fits easily on A100 40GB.
    # Context: 16K only (smaller than Qwen's 32K — set max_model_len accordingly).
    "phi-4": ModelProfile(
        name="phi-4",
        description="Microsoft Phi-4 (AWQ 4-bit, curiousmind147) on 1× A100-40GB",
        model_id="curiousmind147/microsoft-phi-4-AWQ-4bit-GEMM",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=12000,      # 16384 - 2048 - 2336 margin
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),


    # ---- Llama-3.1-8B-Instruct (AWQ) — L4 22GB / A100 40GB ----
    # Meta Llama 3.1 8B (Jul 2024). Most cited open-source baseline in 2024-2025
    # agent benchmarks. Standard llama architecture → vLLM 0.6.6 compatible.
    # 8B AWQ ≈ 4GB weights → fits comfortably on L4 22GB.
    # Llama 3.1 license (permissive for research).
    "llama-3.1-8b": ModelProfile(
        name="llama-3.1-8b",
        description="Meta Llama-3.1-8B-Instruct (AWQ 4-bit, hugging-quants) on 1× L4/A100-40GB",
        model_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        quantization="awq",
        gpu="L4-22GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Llama-3.1-70B-Instruct (AWQ) — A100 40GB (tight fit) ----
    # Meta Llama 3.1 70B (Jul 2024). Strongest Llama for agent tasks.
    # AWQ 4-bit reduces ~140GB (fp16) to ~35GB → fits on single A100 40GB.
    # Needs reduced max_model_len=8192 to leave room for KV cache.
    # Standard llama architecture → vLLM 0.6.6 compatible.
    "llama-3.1-70b": ModelProfile(
        name="llama-3.1-70b",
        description="Meta Llama-3.1-70B-Instruct (AWQ 4-bit, hugging-quants) on 1× A100-40GB",
        model_id="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=8192,           # Tight: 35GB weights leaves ~5GB for KV cache
        gpu_memory_utilization=0.97,  # Push limit to fit 70B AWQ on 40GB
        max_tokens=2048,
        max_context_tokens=5000,      # 8192 - 2048 - 1144 margin
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen3-Coder-30B-A3B — Jean-Zay 1× A100-80GB (bf16, MoE) ----
    # MoE: 30B total params, ~3B active per token. Memory ~60GB (loads all params).
    # ---- Qwen3-32B (AWQ, non-thinking) — A100 40GB ----
    # Generational counterpart to Qwen2.5-Coder-32B: SAME size (32B), SAME quant
    # (AWQ 4-bit), SAME context (32K) -> isolates the Qwen2.5->Qwen3 generation jump.
    # NOTE: Qwen3-32B is a GENERAL (not "Coder") model — no dense Qwen3-Coder-32B
    # exists (the Qwen3-Coder line is MoE). State this confound in the write-up.
    # Qwen3 is a HYBRID reasoning model; thinking is ON by default. For a fair
    # comparison vs non-reasoning Qwen2.5 we serve it NON-THINKING. Enforced at
    # serve time in the Colab notebook via vLLM (>= 0.9.0; arch needs >= 0.8.5):
    #   --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
    # The harness sends plain chat-completions, so the server default keeps every
    # turn non-thinking. 32B AWQ ~18-20GB weights -> fits A100 40GB.
    "qwen3-32b": ModelProfile(
        name="qwen3-32b",
        description="Qwen3-32B (AWQ 4-bit, non-thinking) on 1x A100-40GB",
        enable_thinking=False,           # reasoning OFF (per-request override)
        model_id="Qwen/Qwen3-32B-AWQ",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,    # 32768 - 2048 - 2720 margin (== Qwen2.5-32B)
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen3-32B-THINKING (AWQ, reasoning ON) — A100 40GB ----
    # Identical serving to "qwen3-32b" but with reasoning ENABLED, to isolate the
    # thinking axis (B-depth). enable_thinking=True is sent per-request via
    # chat_template_kwargs; size/quant/context/budgets held constant vs the
    # non-thinking control above. max_tokens raised to 4096 because Qwen3 thinking
    # traces routinely exceed 2048 tokens before emitting the answer/tool-call.
    "qwen3-32b-thinking": ModelProfile(
        name="qwen3-32b-thinking",
        description="Qwen3-32B (AWQ 4-bit, reasoning ON) on 1x A100-40GB",
        enable_thinking=True,            # reasoning ON (per-request override)
        model_id="Qwen/Qwen3-32B-AWQ",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=4096,                 # headroom for reasoning trace
        max_context_tokens=26000,        # 32768 - 4096 - margin
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Qwen3.6-27B (AWQ, non-thinking) — A100 40GB ----
    # Latest open agentic-coding flagship that fits 40GB (dense 27B, Apr 2026).
    # Terminal-Bench 2.0 = 59.3, SWE-bench Verified 77.2 (beats Qwen3.5-397B-A17B).
    # AWQ-INT4 ~16-21GB weights -> fits A100 40GB. Native ctx 262K (we cap at 32K
    # for KV-cache fit + parity with the rest of the grid). Hybrid reasoning model:
    # served NON-THINKING (notebook: --reasoning-parser qwen3
    # --default-chat-template-kwargs '{"enable_thinking": false}'). Needs a RECENT
    # vLLM supporting the Qwen3.6 arch (Apr-2026) — newer than the Qwen3-32B floor.
    # This is a separate "latest-SOTA-that-fits-40GB" data point, NOT the controlled
    # Qwen2.5->Qwen3 generational pair (it changes both generation AND size).
    "qwen3.6-27b": ModelProfile(
        name="qwen3.6-27b",
        description="Qwen3.6-27B (AWQ 4-bit, non-thinking) on 1x A100-40GB",
        model_id="cyankiwi/Qwen3.6-27B-AWQ-INT4",
        quantization="awq",
        gpu="A100-40GB",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        max_tokens=2048,
        max_context_tokens=28000,    # 32768 - 2048 - 2720 margin
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    "qwen3-coder-30b": ModelProfile(
        name="qwen3-coder-30b",
        description="Qwen3-Coder-30B-A3B-Instruct (bf16, MoE) on 1× A100-80GB",
        model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        quantization=None,
        gpu="A100-80GB",
        max_model_len=32768,
        gpu_memory_utilization=0.88,
        max_tokens=2048,
        max_context_tokens=28000,
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
    ),

    # ---- Gemini 3.1 Flash-Lite — Google AI API (switched from 2.5-flash 2026-07-09: 2.5 retired by Google) ----
    "gemini-flash": ModelProfile(
        name="gemini-flash",
        description="Gemini 3.1 Flash-Lite via Google AI OpenAI-compatible API",
        model_id="gemini-3.1-flash-lite",
        quantization=None,
        gpu="cloud",
        max_model_len=1048576,          # 1M context window
        gpu_memory_utilization=0.0,     # N/A for cloud
        max_tokens=8192,
        max_context_tokens=128000,      # conservative limit for agent use
        max_steps=60,
        max_wallclock_seconds=1800,
        chars_per_token=4.0,
        token_safety_factor=1.20,
        tool_calling="prompt",           # prompt-based for FAIRNESS (all local models use prompt); was native
        provider="openai",              # OpenAI-compatible endpoint
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",   # reads from this env var
    ),
}

# Aliases for convenience
PROFILES["7"] = PROFILES["7b"]
PROFILES["14"] = PROFILES["14b"]
PROFILES["32"] = PROFILES["32b"]
PROFILES["72"] = PROFILES["72b"]
PROFILES["gemini"] = PROFILES["gemini-flash"]
# GENCI aliases (shorter names for CLI use)
PROFILES["llama8b"] = PROFILES["llama-3.1-8b"]
PROFILES["llama70b-awq"] = PROFILES["llama-3.1-70b"]
PROFILES["llama70b"] = PROFILES["llama-3.3-70b"]
PROFILES["mistral-nemo"] = PROFILES["mistral-nemo-12b"]
PROFILES["mistral12b"] = PROFILES["mistral-nemo-12b"]
PROFILES["deepseek32b"] = PROFILES["deepseek-r1-32b"]
PROFILES["qwen3-30b"] = PROFILES["qwen3-coder-30b"]
PROFILES["qwen3-32b-awq"] = PROFILES["qwen3-32b"]
PROFILES["qwen3-32"] = PROFILES["qwen3-32b"]
PROFILES["qwen3-32b-think"] = PROFILES["qwen3-32b-thinking"]
PROFILES["qwen3-thinking"] = PROFILES["qwen3-32b-thinking"]
PROFILES["qwen3.6-27b-awq"] = PROFILES["qwen3.6-27b"]
PROFILES["qwen3.6"] = PROFILES["qwen3.6-27b"]
PROFILES["qwen36-27b"] = PROFILES["qwen3.6-27b"]
PROFILES["devstral"] = PROFILES["devstral-small"]
PROFILES["phi4"] = PROFILES["phi-4"]





def get_profile(name: str) -> ModelProfile:
    """
    Get a model profile by name.

    Args:
        name: Profile name (e.g., "7b", "14b", "32b", "72b", "32b-fp16")

    Raises:
        ValueError: If profile not found.
    """
    name = name.lower().strip()
    if name not in PROFILES:
        available = sorted(set(p.name for p in PROFILES.values()))
        raise ValueError(
            f"Unknown model profile: '{name}'. "
            f"Available profiles: {', '.join(available)}"
        )
    return PROFILES[name]


def list_profiles() -> list[ModelProfile]:
    """Return all unique profiles (no aliases)."""
    seen = set()
    result = []
    for p in PROFILES.values():
        if p.name not in seen:
            seen.add(p.name)
            result.append(p)
    return result


def apply_profile(config, profile: ModelProfile):
    """
    Apply a model profile to a Config object, overriding relevant settings.

    Environment variables STILL take priority if set — the profile provides
    defaults that the env vars can override.

    Args:
        config: A Config object from config.load_config()
        profile: The ModelProfile to apply

    Returns:
        The modified config (same object, mutated in place)
    """
    import os

    # Only apply if env var not explicitly set (env vars win)
    def _env_or(env_key: str, profile_value, setter):
        if not os.environ.get(env_key):
            setter(profile_value)

    # LLM settings
    _env_or("THESIS_LLM_MODEL", profile.model_id,
            lambda v: setattr(config.llm, "model", v))
    _env_or("THESIS_LLM_MAX_TOKENS", profile.max_tokens,
            lambda v: setattr(config.llm, "max_tokens", v))
    _env_or("THESIS_LLM_TEMPERATURE", profile.temperature,
            lambda v: setattr(config.llm, "temperature", v))
    _env_or("THESIS_LLM_TOOL_CALLING", profile.tool_calling,
            lambda v: setattr(config.llm, "tool_calling", v))
    _env_or("THESIS_LLM_TIMEOUT", profile.llm_timeout_seconds,
            lambda v: setattr(config.llm, "timeout", v))
    _env_or("THESIS_LLM_ENABLE_THINKING", profile.enable_thinking,
            lambda v: setattr(config.llm, "enable_thinking", v))

    # Provider overrides (for cloud API profiles like Gemini)
    # These FORCE-override env vars because provider/api_base are integral
    # to a cloud profile's identity — the vLLM defaults must not leak through.
    if profile.provider:
        config.llm.provider = profile.provider
    if profile.api_base:
        config.llm.api_base = profile.api_base
    if profile.api_key_env:
        key = os.environ.get(profile.api_key_env, "")
        if key:
            config.llm.api_key = key

    # Agent settings
    _env_or("THESIS_AGENT_MAX_STEPS", profile.max_steps,
            lambda v: setattr(config.agent, "max_steps", v))
    _env_or("THESIS_AGENT_MAX_WALLCLOCK", profile.max_wallclock_seconds,
            lambda v: setattr(config.agent, "max_wallclock_seconds", v))
    _env_or("THESIS_AGENT_MAX_CONTEXT_TOKENS", profile.max_context_tokens,
            lambda v: setattr(config.agent, "max_context_tokens", v))

    return config
