"""Tests for ChatClient v2 API (chat/stream)."""

from __future__ import annotations

import pytest

from republic.clients.chat import ChatClient
from republic.core.results import (
    ErrorEvent,
    FinalEvent,
    LLMResult,
    PreparedChat,
    TextEvent,
)

from .fakes import make_chunk, make_response, make_tool_call


class FakeLLMCore:
    """Minimal fake core for ChatClient tests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: list = []

    def wrap_error(self, exc: Exception, provider: str, model: str):
        from republic.core.errors import ErrorKind, RepublicError
        return RepublicError(ErrorKind.TEMPORARY, str(exc))

    async def run_chat_async(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("No more responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        on_response = kwargs.get("on_response")
        if on_response is not None:
            return on_response(response, "test_provider", "test_model", 0)
        return response


@pytest.mark.asyncio
async def test_chat_returns_text_result() -> None:
    core = FakeLLMCore()
    core.responses.append(make_response(text="hello"))
    client = ChatClient(core)

    prepared = PreparedChat(model="gpt-4", provider="openai")
    messages = [{"role": "user", "content": "hi"}]

    result = await client.chat(prepared, messages)

    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.ok is True
    assert result.has_tool_calls is False


@pytest.mark.asyncio
async def test_chat_returns_tool_calls() -> None:
    core = FakeLLMCore()
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    client = ChatClient(core)

    prepared = PreparedChat(model="gpt-4", provider="openai")
    messages = [{"role": "user", "content": "call echo"}]

    result = await client.chat(prepared, messages)

    assert isinstance(result, LLMResult)
    assert result.has_tool_calls is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "echo"


@pytest.mark.asyncio
async def test_chat_returns_error_on_exception() -> None:
    core = FakeLLMCore()
    core.responses.append(RuntimeError("network error"))
    client = ChatClient(core)

    prepared = PreparedChat(model="gpt-4", provider="openai")
    messages = [{"role": "user", "content": "hi"}]

    result = await client.chat(prepared, messages)

    assert isinstance(result, LLMResult)
    assert result.error is not None
    assert "network error" in result.error.message


@pytest.mark.asyncio
async def test_stream_yields_text_events_and_final() -> None:
    core = FakeLLMCore()

    async def fake_stream():
        yield make_chunk(text="hello ")
        yield make_chunk(text="world", usage={"total_tokens": 2})

    core.responses.append(fake_stream())
    client = ChatClient(core)

    prepared = PreparedChat(model="gpt-4", provider="openai")
    messages = [{"role": "user", "content": "hi"}]

    stream = await client.stream(prepared, messages)
    events = []
    async for event in stream:
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextEvent)]
    final_events = [e for e in events if isinstance(e, FinalEvent)]

    assert len(text_events) == 2
    assert text_events[0].content == "hello "
    assert text_events[1].content == "world"
    assert len(final_events) == 1
    assert final_events[0].result.text == "hello world"
    assert final_events[0].result.usage == {"total_tokens": 2}


@pytest.mark.asyncio
async def test_stream_returns_error_event_on_failure() -> None:
    core = FakeLLMCore()

    async def broken_stream():
        raise RuntimeError("stream broken")
        yield  # type: ignore[unreachable]

    core.responses.append(broken_stream())
    client = ChatClient(core)

    prepared = PreparedChat(model="gpt-4", provider="openai")
    messages = [{"role": "user", "content": "hi"}]

    stream = await client.stream(prepared, messages)
    events = []
    async for event in stream:
        events.append(event)

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "stream broken" in error_events[0].error.message


@pytest.mark.asyncio
async def test_chat_passes_tools_to_core() -> None:
    core = FakeLLMCore()
    core.responses.append(make_response(text="ok"))
    client = ChatClient(core)

    prepared = PreparedChat(
        model="gpt-4",
        provider="openai",
        tools=[{"type": "function", "function": {"name": "echo"}}],
    )
    messages = [{"role": "user", "content": "hi"}]

    await client.chat(prepared, messages)

    assert len(core.calls) == 1
    assert core.calls[0]["tools_payload"] is not None
    assert len(core.calls[0]["tools_payload"]) == 1
