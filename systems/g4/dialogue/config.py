"""
Central configuration for the thesis agent system.

All settings can be overridden via environment variables prefixed with THESIS_.
Defaults target local Ollama serving with Qwen2.5-Coder.
"""
from __future__ import annotations

import os
from typing import Optional, Literal

from pydantic import BaseModel, Field

# Auto-load .env from project root (no-op if file not present or dotenv not installed)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # env vars already exported take priority
except ImportError:
    pass


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""

    # Provider routing — which OpenAI-compatible server to target
    provider: Literal["ollama", "vllm", "openai", "bedrock"] = "ollama"
    model: str = "qwen2.5-coder:32b"

    # Connection
    api_base: Optional[str] = None   # Auto-resolved per provider if None
    api_key: Optional[str] = None    # Auto-resolved per provider if None

    # Generation defaults
    temperature: float = 0.0
    max_tokens: int = 4096

    # Reliability
    timeout: int = 300
    max_retries: int = 6

    # Tool calling strategy
    #   "native"  — use OpenAI-format tool_calls (Qwen2.5, Llama3.1+, etc.)
    #   "prompt"  — inject tool descriptions into system prompt, parse JSON from output
    tool_calling: Literal["native", "prompt"] = "native"

    # Reasoning/thinking toggle for hybrid models (Qwen3 family).
    #   None  — send nothing; server default applies (zero effect for models
    #           that ignore it, e.g. Qwen2.5 / Llama / Gemini)
    #   True  — force reasoning ON  (per-request chat_template_kwargs)
    #   False — force reasoning OFF
    enable_thinking: Optional[bool] = None

    def get_api_base(self) -> str:
        """Resolve API base URL."""
        if self.api_base:
            return self.api_base
        defaults = {
            "ollama": "http://localhost:11434/v1",
            "vllm": "http://localhost:8000/v1",
            "openai": "https://api.openai.com/v1",
            "bedrock": "",  # Bedrock uses boto3, not HTTP base
        }
        return defaults.get(self.provider, "http://localhost:11434/v1")

    def get_api_key(self) -> str:
        """Resolve API key (many local providers don't need one)."""
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("THESIS_LLM_API_KEY")
        if env_key:
            return env_key
        # Ollama and vLLM don't require real keys
        if self.provider in ("ollama", "vllm"):
            return "not-needed"
        return "not-set"

    def get_model_string(self) -> str:
        """Model identifier for the OpenAI client."""
        return self.model


class AgentConfig(BaseModel):
    """Configuration for agent execution budgets and behavior."""

    # Step budgets
    max_steps: int = 60
    max_wallclock_seconds: int = 1800   # 30min TOTAL (uniform with G1/G2/G3); divided across turns

    # Context management
    max_output_chars: int = 16000        # Truncate terminal output per command
    max_context_turns: int = 40          # Sliding window size for conversation

    # ── Ablation knobs (env var controlled) ──────────────────────────
    # Prompt variant: "v1" (original) | "stage0" (data-grounding + verify)
    prompt_version: str = "v1"
    # Tool set: "default" (6 tools) | "enhanced" (+search_files, +list_dir)
    tool_set: str = "default"
    # Context mode: "default" (narrative LLM summary) | "enhanced" (factual
    # extraction + categorized drops + auto-pinned key facts)
    context_mode: str = "default"

    # Token budget for the input side of every LLM call.
    # Formula:  max_context_tokens = max_model_len - max_tokens - small_margin
    #
    # The agent memory trims conversation history so that:
    #   (memory_tokens + tool_schema_overhead≈2000) * 1.15 ≤ max_context_tokens
    #
    # Examples:
    #   Ollama 32B (32768 ctx, max_tokens=4096):  32768 - 4096 - 672  ≈ 28000  ← default
    #   vLLM 32B  (32768 ctx, max_tokens=2048):  32768 - 2048 - 720  ≈ 30000  ← Colab A100
    #   vLLM  7B  (16384 ctx, max_tokens=800):   16384 -  800 - 584  ≈ 15000  (deprecated)
    #
    # Override with env var: THESIS_AGENT_MAX_CONTEXT_TOKENS
    max_context_tokens: int = 28000

    # Terminal defaults
    command_timeout: float = 30.0        # Default per-command timeout
    long_command_timeout: float = 120.0  # For pip install, make, etc.

    # Pipeline architecture.
    #   "single"   — single ReAct agent (baseline)
    #   "dialogue" — G4: two agents taking turns
    # Override with env var: THESIS_AGENT_PIPELINE
    pipeline: str = "single"

    # Dialogue variant: "cooperation" (relay) | "debate" (adversarial critique)
    # Override with env var: THESIS_DIALOGUE_VARIANT
    dialogue_variant: str = "cooperation"

    # Steps each agent gets per turn before handing off.
    # Total turns = max_steps / dialogue_steps_per_turn.
    # Override with env var: THESIS_DIALOGUE_STEPS_PER_TURN
    dialogue_steps_per_turn: int = 7


class Config(BaseModel):
    """Root configuration container."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Supported env vars (all optional, sensible defaults used):
        THESIS_LLM_PROVIDER     — ollama | vllm | openai | bedrock
        THESIS_LLM_MODEL        — model name/tag
        THESIS_LLM_API_BASE     — server URL
        THESIS_LLM_API_KEY      — API key
        THESIS_LLM_TEMPERATURE  — float
        THESIS_LLM_MAX_TOKENS   — int
        THESIS_LLM_TIMEOUT      — int (seconds)
        THESIS_LLM_TOOL_CALLING — native | prompt
        THESIS_AGENT_MAX_STEPS  — int
        THESIS_AGENT_MAX_WALLCLOCK — int (seconds)
        THESIS_AGENT_MAX_OUTPUT — int (chars)
    """
    llm_kwargs: dict = {}

    if v := os.environ.get("THESIS_LLM_PROVIDER"):
        llm_kwargs["provider"] = v
    if v := os.environ.get("THESIS_LLM_MODEL"):
        llm_kwargs["model"] = v
    if v := os.environ.get("THESIS_LLM_API_BASE"):
        llm_kwargs["api_base"] = v
    if v := os.environ.get("THESIS_LLM_API_KEY"):
        llm_kwargs["api_key"] = v
    if v := os.environ.get("THESIS_LLM_TEMPERATURE"):
        llm_kwargs["temperature"] = float(v)
    if v := os.environ.get("THESIS_LLM_MAX_TOKENS"):
        llm_kwargs["max_tokens"] = int(v)
    if v := os.environ.get("THESIS_LLM_TIMEOUT"):
        llm_kwargs["timeout"] = int(v)
    if v := os.environ.get("THESIS_LLM_TOOL_CALLING"):
        llm_kwargs["tool_calling"] = v

    agent_kwargs: dict = {}

    if v := os.environ.get("THESIS_AGENT_MAX_STEPS"):
        agent_kwargs["max_steps"] = int(v)
    if v := os.environ.get("THESIS_AGENT_MAX_WALLCLOCK"):
        agent_kwargs["max_wallclock_seconds"] = int(v)
    if v := os.environ.get("THESIS_AGENT_MAX_OUTPUT"):
        agent_kwargs["max_output_chars"] = int(v)
    if v := os.environ.get("THESIS_AGENT_MAX_CONTEXT_TOKENS"):
        agent_kwargs["max_context_tokens"] = int(v)
    if v := os.environ.get("THESIS_AGENT_PROMPT_VERSION"):
        agent_kwargs["prompt_version"] = v
    if v := os.environ.get("THESIS_AGENT_TOOLS"):
        agent_kwargs["tool_set"] = v
    if v := os.environ.get("THESIS_AGENT_CONTEXT"):
        agent_kwargs["context_mode"] = v
    if v := os.environ.get("THESIS_AGENT_PIPELINE"):
        agent_kwargs["pipeline"] = v
    if v := os.environ.get("THESIS_DIALOGUE_VARIANT"):
        agent_kwargs["dialogue_variant"] = v.lower().strip()
    if v := os.environ.get("THESIS_DIALOGUE_STEPS_PER_TURN"):
        agent_kwargs["dialogue_steps_per_turn"] = int(v)

    return Config(
        llm=LLMConfig(**llm_kwargs),
        agent=AgentConfig(**agent_kwargs),
    )
