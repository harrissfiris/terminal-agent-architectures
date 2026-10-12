"""
LLM provider abstraction layer.

Usage:
    from config import load_config
    from llm import create_llm

    config = load_config()
    llm = create_llm(config.llm)
    response = llm.chat(messages=[{"role": "user", "content": "Hello"}])
"""
from .base import BaseLLM, LLMResponse, ToolCall, Message
from config import LLMConfig


def create_llm(config: LLMConfig) -> BaseLLM:
    """
    Factory: create an LLM provider from configuration.

    Routing:
        provider="bedrock" → BedrockProvider (requires boto3)
        everything else    → OpenAICompatProvider (Ollama, vLLM, OpenAI, TGI, ...)
    """
    if config.provider == "bedrock":
        try:
            from .bedrock_provider import BedrockProvider
            return BedrockProvider(config)
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock provider. "
                "Install with: pip install boto3"
            )

    from .openai_compat import OpenAICompatProvider
    return OpenAICompatProvider(config)


__all__ = [
    "BaseLLM",
    "LLMResponse",
    "ToolCall",
    "Message",
    "create_llm",
]
