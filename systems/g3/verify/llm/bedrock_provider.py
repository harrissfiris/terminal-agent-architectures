"""
Legacy Bedrock provider (optional, requires boto3 + AWS credentials).

Wraps AWS Bedrock Claude API behind the BaseLLM interface so existing
code can still run if AWS access is available.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from config import LLMConfig
from .base import BaseLLM, LLMResponse, ToolCall, Message

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


class BedrockProvider(BaseLLM):
    """LLM provider using AWS Bedrock (Anthropic Claude)."""

    def __init__(self, config: LLMConfig):
        import boto3

        self.config = config
        self._client = boto3.client("bedrock-runtime")
        self._model_id = config.model or ANTHROPIC_MODEL_ID
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def total_usage(self) -> tuple[int, int]:
        return (self._total_input_tokens, self._total_output_tokens)

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat completion to Bedrock.

        Note: tool calling is not supported in this legacy provider.
        Convert tool-calling workflows to prompt-based format externally.
        """
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        # Separate system prompt from messages (Anthropic format)
        system_prompt = ""
        api_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tok,
            "messages": api_messages,
        }
        if system_prompt:
            body["system"] = system_prompt
        if temperature is not None:
            body["temperature"] = temperature

        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        self._call_count += 1

        # Token tracking
        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        # Extract text
        content_blocks = response_body.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        content = "\n".join(text_parts).strip() or None

        return LLMResponse(
            content=content,
            tool_calls=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=self._model_id,
            stop_reason=response_body.get("stop_reason"),
        )
