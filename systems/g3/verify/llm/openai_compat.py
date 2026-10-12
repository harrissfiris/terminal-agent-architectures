"""
OpenAI-compatible LLM provider.

Works with ANY server that speaks the OpenAI chat completions API:
  - Ollama       → http://localhost:11434/v1
  - vLLM         → http://localhost:8000/v1
  - OpenAI       → https://api.openai.com/v1
  - TGI          → (with --api-type openai)
  - Azure OpenAI → (with appropriate base_url)

Supports two tool-calling strategies:
  1. "native"  — real OpenAI-format tool_calls (Qwen2.5, Llama3.1+, Mistral, etc.)
  2. "prompt"  — inject tool definitions into system prompt, parse JSON from output
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
)

from config import LLMConfig
from .base import BaseLLM, LLMResponse, ToolCall, Message, ContextOverflowError

logger = logging.getLogger(__name__)

# Substrings that identify a context-length-exceeded 400 from common servers
# (vLLM / OpenAI / TGI).  Used to convert it into a recoverable error.
_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "longer than the maximum",
    "exceeds the maximum",
    "reduce the length",
    "too many tokens",
    "max_tokens must be at least",
)


class OpenAICompatProvider(BaseLLM):
    """
    LLM provider using the OpenAI Python SDK.

    Handles retries, token tracking, and both native and
    prompt-based tool calling.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = OpenAI(
            base_url=config.get_api_base(),
            api_key=config.get_api_key(),
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._model = config.get_model_string()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

        # ---- Dynamic chars/token calibration ----
        # Learn the model's REAL chars/token from the server's own tokenizer
        # (usage.prompt_tokens vs characters sent).  Model-agnostic, no deps.
        # Memory trimming reads this to avoid context overflow.
        self._chars_per_token: float = 4.0
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def total_usage(self) -> tuple[int, int]:
        """Cumulative (input_tokens, output_tokens)."""
        return (self._total_input_tokens, self._total_output_tokens)

    @property
    def chars_per_token(self) -> float:
        """Current chars/token ratio, calibrated from the server's tokenizer."""
        return self._chars_per_token

    @staticmethod
    def _payload_chars(messages: list[Message]) -> int:
        """Approximate character count of the request payload actually sent."""
        total = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
        return total

    def _calibrate(self, payload_chars: int, prompt_tokens: int) -> None:
        """Update chars/token from the server's real prompt_tokens count (EMA)."""
        if prompt_tokens < 200 or payload_chars < 400:
            return
        observed = max(2.0, min(payload_chars / prompt_tokens, 6.0))
        if not self._calibrated:
            self._chars_per_token = observed
            self._calibrated = True
        else:
            self._chars_per_token = 0.7 * self._chars_per_token + 0.3 * observed

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send a chat completion with automatic retry and tool-calling support."""
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        # Prepare messages (may modify for prompt-based tool calling)
        final_messages = list(messages)
        use_native_tools = False

        if tools and self.config.tool_calling == "native":
            use_native_tools = True
        elif tools and self.config.tool_calling == "prompt":
            final_messages = self._convert_messages_for_prompt_mode(final_messages)
            final_messages = self._inject_tools_into_prompt(final_messages, tools)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": final_messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        # ---- Reasoning/thinking toggle (Qwen3 hybrid models) ----
        # Sent only when explicitly configured. None => omit entirely, so the
        # server default applies and non-Qwen3 backends never see an unknown
        # kwarg. Routed via vLLM per-request chat_template_kwargs.
        _enable_thinking = getattr(self.config, "enable_thinking", None)
        if _enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": _enable_thinking}
            }

        if use_native_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        payload_chars = self._payload_chars(final_messages)

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                parsed = self._parse_response(response, tools)
                # Learn the model's true chars/token from the server tokenizer
                if response.usage and response.usage.prompt_tokens:
                    self._calibrate(payload_chars, response.usage.prompt_tokens)
                return parsed

            except BadRequestError as e:
                # If native tool calling fails, try without tools
                if use_native_tools and "tool" in str(e).lower():
                    logger.warning(
                        f"Native tool calling failed ({e}), retrying without tools"
                    )
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    # Inject tools into prompt instead
                    kwargs["messages"] = self._inject_tools_into_prompt(
                        self._convert_messages_for_prompt_mode(list(messages)),
                        tools,
                    )
                    use_native_tools = False
                    continue
                # A context-length 400 is recoverable: signal the agent loop
                # to trim history and retry instead of failing the task.
                if any(mk in str(e).lower() for mk in _CONTEXT_OVERFLOW_MARKERS):
                    raise ContextOverflowError(str(e)) from e
                raise

            except (APITimeoutError, APIConnectionError) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)

            except RateLimitError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s: {e}")
                    time.sleep(wait)

            except Exception as e:
                logger.error(f"LLM call failed with unexpected error: {type(e).__name__}: {e}")
                raise

        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries + 1} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: Any,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Parse OpenAI API response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        # Token tracking
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._call_count += 1

        content = message.content
        raw_content = content  # preserve the full untouched output for tracing
        tool_calls: list[ToolCall] = []
        parse_error = False

        # Native tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # Prompt-based tool call parsing (fallback)
        elif tools and self.config.tool_calling == "prompt" and content:
            parsed = self._parse_tool_call_from_text(content)
            if parsed:
                tool_calls.append(parsed)
                content = None  # Tool call replaces content
            else:
                # Could not extract a tool call.  We DON'T silently fabricate a
                # think() any more — that hid the failure from the model and
                # caused infinite parse-loops on models that wrap calls in
                # markdown fences / verbose reasoning (e.g. Phi-4).  Instead we
                # flag it so the agent can send an explicit correction.
                logger.warning(
                    f"Failed to parse tool call from prompt output "
                    f"({len(content)} chars). Flagging parse_error."
                )
                parse_error = True

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=getattr(response, "model", self._model) or self._model,
            stop_reason=choice.finish_reason,
            parse_error=parse_error,
            raw_content=raw_content,
        )

    # ------------------------------------------------------------------
    # Prompt-based tool calling (for models without native support)
    # ------------------------------------------------------------------

    def _convert_messages_for_prompt_mode(
        self, messages: list[Message]
    ) -> list[Message]:
        """
        Convert tool-calling message history to plain text format.

        - 'tool' role messages → 'user' role with header
        - 'assistant' messages with tool_calls → 'assistant' with JSON text
        """
        converted: list[Message] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                tool_name = msg.get("name", "unknown")
                content = msg.get("content", "")
                converted.append({
                    "role": "user",
                    "content": f"[Tool Result: {tool_name}]\n{content}",
                })

            elif role == "assistant" and msg.get("tool_calls"):
                parts: list[str] = []
                if msg.get("content"):
                    parts.append(msg["content"])
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")
                    parts.append(f'{{"tool": "{name}", "args": {args}}}')
                converted.append({
                    "role": "assistant",
                    "content": "\n".join(parts),
                })

            else:
                converted.append(msg)

        return converted

    def _inject_tools_into_prompt(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> list[Message]:
        """Inject tool descriptions into the system prompt."""
        tool_desc_parts = ["You have access to the following tools:\n"]

        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_lines: list[str] = []
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_lines.append(f"    - {pname}: {ptype}{req} — {pdesc}")

            tool_desc_parts.append(f"## {name}")
            tool_desc_parts.append(f"{desc}")
            if param_lines:
                tool_desc_parts.append("Parameters:")
                tool_desc_parts.extend(param_lines)
            tool_desc_parts.append("")

        tool_desc_parts.append(
            "To use a tool, respond with EXACTLY this JSON on a SINGLE line:\n"
            '{"tool": "<tool_name>", "args": {<arguments>}}\n\n'
            "If you want to think or reason without calling a tool:\n"
            '{"tool": "think", "args": {"content": "<your reasoning>"}}\n\n'
            "IMPORTANT: Your entire response must be a single JSON object."
        )

        tool_system_msg = "\n".join(tool_desc_parts)
        new_messages = list(messages)

        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0] = {
                "role": "system",
                "content": new_messages[0]["content"] + "\n\n" + tool_system_msg,
            }
        else:
            new_messages.insert(0, {"role": "system", "content": tool_system_msg})

        return new_messages

    # Reasoning wrappers some models emit before/around their tool call.
    # Matched closed OR unclosed (DeepSeek-R1 may emit only the closing tag).
    _REASONING_TAGS = ("think", "thinking", "reasoning", "thought", "reflection")

    @classmethod
    def _strip_reasoning(cls, text: str) -> str:
        """Remove <think>…</think> style reasoning blocks (model-agnostic).

        Handles closed pairs, a leading unclosed close tag (``…</think>``),
        and an open tag with no close (keeps text before it)."""
        for tag in cls._REASONING_TAGS:
            text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(rf"^.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(rf"<{tag}>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _iter_json_objects(text: str):
        """Yield every top-level balanced {...} substring, in order.

        String-aware (braces inside quoted strings don't affect depth), so a
        JSON value that contains '}' no longer truncates extraction."""
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        yield text[start : i + 1]
                        start = -1

    # Matches the opening of a tool-call-shaped object: {"tool"|"action"|"name":
    _TOOL_ANCHOR = re.compile(r'\{\s*"(?:tool|action|name)"\s*:')
    # Mistral-specific tokenizer special token marking a native tool call.
    # Used to gate the [TOOL_CALLS] fallback parser so non-Mistral output
    # (which never contains this token) is never affected.
    _MISTRAL_TOOL_TOKEN = "[TOOL_CALLS]"

    def _iter_tool_anchored(self, text: str):
        """Yield balanced {...} objects starting at each tool-call signature.

        Starting AT the '{' of a tool-call object means quotes/braces in any
        preceding prose or code fence are skipped entirely — only the object's
        own (valid-JSON, double-quoted) string tracking applies."""
        for m in self._TOOL_ANCHOR.finditer(text):
            depth = 0
            in_str = False
            esc = False
            for i in range(m.start(), len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[m.start():i + 1]
                        break

    def _first_object_toolcall(self, text: str) -> Optional[ToolCall]:
        """Coerce a tool call from the FIRST balanced JSON object at the start
        of `text` (which is known to begin with '{').  Tolerates trailing
        content after the object (the DeepSeek lead-then-ramble case)."""
        try:
            return self._coerce_tool_call(json.loads(text))
        except json.JSONDecodeError:
            for blob in self._iter_json_objects(text):
                try:
                    return self._coerce_tool_call(json.loads(blob))
                except json.JSONDecodeError:
                    return None
            return None

    def _coerce_tool_call(self, obj: Any) -> Optional[ToolCall]:
        """Build a ToolCall from a parsed dict, accepting key-name variants."""
        if not isinstance(obj, dict):
            return None
        tool_name = obj.get("tool") or obj.get("action") or obj.get("name")
        if not tool_name or str(tool_name).lower() in ("null", "none", ""):
            return None
        args = (
            obj.get("args")
            or obj.get("action_input")
            or obj.get("arguments")
            or obj.get("parameters")
            or {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return ToolCall(
            id=f"prompt_{self._call_count}",
            name=str(tool_name),
            arguments=args if isinstance(args, dict) else {"_raw": str(args)},
        )

    def _parse_mistral_tool_calls(self, text: str) -> Optional[ToolCall]:
        """Recover a tool call emitted in Mistral's NATIVE format.

        Devstral / Mistral-Small are trained to emit tool calls via the special
        token ``[TOOL_CALLS]`` (usually after a line of prose), followed EITHER by
        ``<tool_name>{<json args>}`` (Devstral's usual form, e.g.
        ``[TOOL_CALLS]bash{"command": "ls"}``) OR by the OpenAI-style array
        ``[{"name": <tool>, "arguments": {<args>}}, ...]``.  The harness asks for
        ``{"tool":...,"args":...}`` JSON (tool_calling="prompt") and Devstral
        mostly complies but reverts to this native format intermittently; the
        inner object then carries no tool/action/name key, so the generic parser
        above cannot recover it and the step is wasted on a correction.

        GATED on the literal ``[TOOL_CALLS]`` token (a Mistral tokenizer special
        token NO other model emits) so non-Mistral output never reaches here.
        """
        if not text or self._MISTRAL_TOOL_TOKEN not in text:
            return None
        after = text.split(self._MISTRAL_TOOL_TOKEN, 1)[1].lstrip()

        # Form A: OpenAI-style array of {"name":..,"arguments":..} objects.
        if after.startswith("["):
            for blob in self._iter_json_objects(after):
                try:
                    obj = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                tc = self._coerce_tool_call(obj)
                if tc:
                    return tc
            return None

        # Form B: <tool_name>{<json args>}.
        m = re.match(r"([A-Za-z_]\w*)\s*(\{.*)", after, re.DOTALL)
        if not m:
            return None
        blob = next(self._iter_json_objects(m.group(2)), None)
        if blob is None:
            return None
        try:
            args = json.loads(blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        return self._coerce_tool_call({"tool": m.group(1), "args": args})

    def _parse_tool_call_from_text(self, text: str) -> Optional[ToolCall]:
        """Parse a tool call from model text output (prompt-based).

        Robust across output styles — bare JSON (Qwen), markdown-fenced JSON
        (Phi-4), JSON preceded by reasoning tags (DeepSeek-R1), and JSON
        embedded in prose.  Returns None only when no valid tool call can be
        recovered, which the caller turns into an explicit correction.
        """
        if not text:
            return None

        # Fast path: output that STARTS with a JSON object — Qwen's clean case,
        # and models that LEAD with the tool call then ramble.  DeepSeek-R1 in
        # particular emits the call, then a hallucinated "[Tool Result]" and more
        # text, ALL before its </think>.  Take the FIRST balanced object so (a)
        # trailing content doesn't break json.loads and (b) reasoning-stripping
        # below can't discard a leading tool call.  Done on RAW text first so a
        # value that legitimately contains "<think>" is never mangled.
        stripped_raw = text.strip()
        if stripped_raw.startswith("{"):
            tc = self._first_object_toolcall(stripped_raw)
            if tc:
                return tc

        # Messy output: remove reasoning wrappers, then extract robustly.
        text = self._strip_reasoning(text).strip()
        if text.startswith("{"):
            tc = self._first_object_toolcall(text)
            if tc:
                return tc

        # Strip a surrounding markdown code fence if present (closed OR not).
        candidate = text
        fence = re.match(r"^```(?:json|tool_call)?\s*(.*?)\s*(?:```)?\s*$",
                         text, re.DOTALL)
        if fence and fence.group(1).lstrip().startswith("{"):
            candidate = fence.group(1)

        # Anchored pass FIRST: scan from each '{"tool"/"action"/"name":' so a
        # ```python block or prose with quotes BEFORE the JSON can't derail the
        # string tracking and swallow the real tool call (a real Phi-4/DeepSeek
        # failure mode — they illustrate a script in a fence, then emit the call).
        for blob in self._iter_tool_anchored(candidate):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            tc = self._coerce_tool_call(obj)
            if tc:
                return tc

        # Then every balanced {...} block (handles tool-key-not-first etc.).
        for blob in self._iter_json_objects(candidate):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            tc = self._coerce_tool_call(obj)
            if tc:
                return tc

        # Last resort A: anchored salvage.  Phi-4 frequently emits MULTIPLE
        # tool objects each missing its final brace, e.g.
        #   {"tool":"think",...}}   (one brace short)  +  prose  +
        #   {"tool":"submit",...}   (one brace short)
        # Scan from each '{"tool"/"action"/"name":' signature and, if the object
        # never closes (or closes mid-string-free), re-balance by appending the
        # missing brace(s) and parse.  Recovers the FIRST such call.  Skipped if
        # the object ends mid-string (truncated content → unsafe to act on).
        tc = self._salvage_anchored(candidate)
        if tc:
            return tc

        # Last resort B: salvage a single trailing tool-call object cut off
        # before its closing brace(s).
        tc = self._salvage_truncated(candidate)
        if tc:
            return tc

        # Last resort C: Mistral native [TOOL_CALLS] format (Devstral /
        # Mistral-Small).  Gated on the Mistral-specific special token, so
        # output from any non-Mistral model never reaches this branch.
        return self._parse_mistral_tool_calls(stripped_raw)

    def _salvage_anchored(self, text: str) -> Optional[ToolCall]:
        """Recover the first tool object at a '{"tool":' anchor when it is one
        or more closing braces short — e.g. Phi-4 emits
        {"tool":"think","args":{...}}  but writes only ONE trailing brace, then
        prose, then another such object.  For each '}' after the anchor we cut
        the candidate object there, append exactly the missing brace(s), and try
        to parse — taking the first that yields a tool call.  Never balances
        across a still-open string (truncated content → unsafe)."""
        for m in self._TOOL_ANCHOR.finditer(text):
            depth = 0
            in_str = False
            esc = False
            for i in range(m.start(), len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break  # object closed properly — earlier passes got it
                    if depth > 0:
                        # candidate object end, one+ brace(s) short → re-balance
                        sub = text[m.start():i + 1]
                        try:
                            obj = json.loads(sub + ("}" * depth))
                        except json.JSONDecodeError:
                            continue
                        tc = self._coerce_tool_call(obj)
                        if tc:
                            return tc
        return None

    def _salvage_truncated(self, text: str) -> Optional[ToolCall]:
        """Recover a trailing {...} object missing only its closing brace(s)."""
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
        if depth > 0 and not in_str and start >= 0:
            try:
                obj = json.loads(text[start:] + ("}" * depth))
            except json.JSONDecodeError:
                return None
            return self._coerce_tool_call(obj)
        return None
