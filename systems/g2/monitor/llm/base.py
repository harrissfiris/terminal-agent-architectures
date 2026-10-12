"""
Abstract LLM interface and shared data types.

All LLM providers implement BaseLLM, returning structured LLMResponse objects
that cleanly separate text content from tool calls.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class ContextOverflowError(Exception):
    """Raised when the server rejects a request because the prompt exceeds
    the model's maximum context length.  The agent loop catches this, trims
    history harder, and retries — so a single oversized prompt never kills
    a task outright."""


@dataclass
class ToolCall:
    """A tool/function call requested by the LLM."""

    id: str                     # Unique call ID (from API or generated)
    name: str                   # Tool function name
    arguments: dict[str, Any]   # Parsed arguments


@dataclass
class LLMResponse:
    """Structured response from an LLM chat completion."""

    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    stop_reason: Optional[str] = None

    # True when the model produced output but no tool call could be parsed
    # from it.  The agent loop uses this to send an explicit "your tool call
    # could not be parsed" correction instead of silently dropping the turn.
    parse_error: bool = False

    # The model's FULL untouched output text (before tool-call extraction /
    # reasoning-stripping).  Captured into the conversation trace so we can see
    # exactly what primary material the parser received — essential for
    # debugging format mismatches (e.g. Phi-4 / DeepSeek-R1).
    raw_content: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def first_tool_call(self) -> Optional[ToolCall]:
        return self.tool_calls[0] if self.tool_calls else None

    @property
    def text(self) -> str:
        """Return content or empty string (convenience)."""
        return self.content or ""


# Type alias for messages in OpenAI-compatible format
Message = dict[str, Any]


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    All providers expose an OpenAI-compatible chat interface with
    optional tool/function calling support.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: Conversation in OpenAI message format.
            tools: Optional tool definitions (OpenAI function-calling schema).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @property
    def total_usage(self) -> tuple[int, int]:
        """Return cumulative (input_tokens, output_tokens). Override in subclasses."""
        return (0, 0)
