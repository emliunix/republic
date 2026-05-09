"""OpenAI chat-completions shape parsing."""

from __future__ import annotations

from typing import Any

from republic.clients.parsing.common import expand_tool_calls, field
from republic.clients.parsing.types import BaseTransportParser


class CompletionTransportParser(BaseTransportParser):
    def is_non_stream_response(self, response: Any) -> bool:
        return isinstance(response, str) or field(response, "choices") is not None

    def extract_chunk_tool_call_deltas(self, chunk: Any) -> list[Any]:
        choices = field(chunk, "choices")
        if not choices:
            return []
        delta = field(choices[0], "delta")
        if delta is None:
            return []
        return field(delta, "tool_calls") or []

    def extract_chunk_text(self, chunk: Any) -> tuple[str | None, str | None]:
        choices = field(chunk, "choices")
        if not choices:
            return None, None
        delta = field(choices[0], "delta")
        if delta is None:
            return None, None
        content = field(delta, "content", "") or ""
        reasoning = field(delta, "reasoning_content", "") or None
        return content, reasoning

    def extract_text(self, response: Any) -> tuple[str, str | None]:
        if isinstance(response, str):
            return response, None

        choices = field(response, "choices")
        if not choices:
            return "", None
        message = field(choices[0], "message")
        if message is None:
            return "", None
        content = field(message, "content", "") or ""
        reasoning = field(message, "reasoning_content") or None
        return content, reasoning

    def extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        choices = field(response, "choices")
        if not choices:
            return []
        message = field(choices[0], "message")
        if message is None:
            return []
        tool_calls = field(message, "tool_calls") or []
        calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = field(tool_call, "function")
            if function is None:
                continue
            entry: dict[str, Any] = {
                "function": {
                    "name": field(function, "name"),
                    "arguments": field(function, "arguments"),
                }
            }
            call_id = field(tool_call, "id")
            if call_id:
                entry["id"] = call_id
            call_type = field(tool_call, "type")
            if call_type:
                entry["type"] = call_type
            calls.append(entry)
        return expand_tool_calls(calls)

    def extract_reasoning(self, response: Any) -> dict | None:
        return None

    def extract_chunk_reasoning(self, chunk: Any) -> str:
        return ""

    def extract_usage(self, response: Any) -> dict[str, Any] | None:
        usage = field(response, "usage")
        if usage is None:
            return None
        if isinstance(usage, dict):
            payload = dict(usage)
        elif hasattr(usage, "model_dump"):
            payload = usage.model_dump()
        else:
            return None
        normalized: dict[str, Any] = {}
        if "input_tokens" in payload:
            normalized["input_tokens"] = payload["input_tokens"]
        elif "prompt_tokens" in payload:
            normalized["input_tokens"] = payload["prompt_tokens"]
        if "output_tokens" in payload:
            normalized["output_tokens"] = payload["output_tokens"]
        elif "completion_tokens" in payload:
            normalized["output_tokens"] = payload["completion_tokens"]
        if "total_tokens" in payload:
            normalized["total_tokens"] = payload["total_tokens"]
        if "requests" in payload:
            normalized["requests"] = payload["requests"]
        return normalized or None
