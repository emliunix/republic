"""Tests for TapeSession v2 API."""

from __future__ import annotations

from typing import Any

import pytest

from republic.clients.chat import ChatClient
from republic.core.results import (
    ErrorEvent,
    FinalEvent,
    Finished,
    LLMResult,
    PreparedChat,
    TextEvent,
    ToolCallNeeded,
)
from republic.tape.entries import TapeEntry
from republic.tape.manager import AsyncTapeManager
from republic.core.errors import ErrorKind, RepublicError
from republic.tape.session import TapeSession, _ensure_text_prompts
from republic.tape.store import InMemoryTapeStore

from .test_chat_client_v2 import FakeLLMCore
from .fakes import make_chunk, make_response, make_tool_call


def test_ensure_text_prompts_returns_string_unchanged() -> None:
    result = _ensure_text_prompts("hello")
    assert result == "hello"


def test_ensure_text_prompts_extracts_text_content() -> None:
    result = _ensure_text_prompts([{"type": "text", "content": "hello"}])
    assert result == "hello"


def test_ensure_text_prompts_returns_empty_string_for_text_without_content() -> None:
    result = _ensure_text_prompts([{"type": "text"}])
    assert result == ""


def test_ensure_text_prompts_raises_for_unsupported_dict() -> None:
    with pytest.raises(RepublicError, match="Expected str") as exc_info:
        _ensure_text_prompts([{"type": "image", "url": "http://example.com"}])
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def test_ensure_text_prompts_raises_for_unsupported_type() -> None:
    with pytest.raises(RepublicError, match="Expected str") as exc_info:
        _ensure_text_prompts(123)
    assert exc_info.value.kind == ErrorKind.INVALID_INPUT


def _make_session(store: InMemoryTapeStore, context: Any = None) -> TapeSession:
    from republic.tape.context import TapeContext
    if context is None:
        context = TapeContext()
    return TapeSession("test_tape", store, context)


@pytest.mark.asyncio
async def test_prepare_records_user_message() -> None:
    store = InMemoryTapeStore()
    session = _make_session(store)

    prepared = await session.prepare(["hello"], "openai", "gpt-4", system_prompt="You are helpful")

    assert isinstance(prepared, PreparedChat)
    entries = store.read("test_tape") or []
    assert len(entries) == 1  # only event; user message deferred in prepared.entries


# =============================================================================
# STATE TRANSITION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_run_error_transition() -> None:
    """prepared → error: run() raises RepublicError and records error event."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    core.responses.append(RuntimeError("network error"))
    client = ChatClient(core)

    prepared = await session.prepare(["hello"], "openai", "gpt-4")

    with pytest.raises(RepublicError, match="network error"):
        await session.run(client, prepared)

    entries = store.read("test_tape") or []
    run_events = [e for e in entries if e.kind == "event" and e.payload["name"] == "run"]
    assert len(run_events) == 1
    assert run_events[0].payload["data"]["status"] == "error"
    assert "network error" in run_events[0].payload["data"]["error"]


@pytest.mark.asyncio
async def test_add_tool_error_transition() -> None:
    """toolcallneeded → error: add_tool_error records error without assistant message."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    client = ChatClient(core)

    prepared = await session.prepare(["call echo"], "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, ToolCallNeeded)

    entries_before = store.read("test_tape") or []
    assert len(entries_before) == 1  # only user message

    await session.add_tool_error(result, RuntimeError("tool failed"))

    entries_after = store.read("test_tape") or []
    run_events = [e for e in entries_after if e.kind == "event" and e.payload["name"] == "run"]
    assert len(run_events) == 1
    assert run_events[0].payload["data"]["status"] == "error"
    assert "tool failed" in run_events[0].payload["data"]["error"]


@pytest.mark.asyncio
async def test_stream_to_finished() -> None:
    """prepared → finish: stream() yields TextEvent + FinalEvent(Finished)."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()

    async def fake_stream():
        yield make_chunk(text="hi ")
        yield make_chunk(text="there", usage={"total_tokens": 2})

    core.responses.append(fake_stream())
    client = ChatClient(core)

    prepared = await session.prepare(["hello"], "openai", "gpt-4")
    stream = await session.stream(client, prepared)

    events = []
    async for event in stream:
        events.append(event)

    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert final_events[0].result.result.text == "hi there"

    entries = store.read("test_tape") or []
    assert len(entries) == 3  # user + assistant + run event
    assert entries[1].kind == "message"
    assert entries[1].payload["role"] == "assistant"
    assert entries[2].kind == "event"
    assert entries[2].payload["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_stream_to_tool_call_needed() -> None:
    """prepared → toolcallneeded: stream() yields FinalEvent(ToolCallNeeded)."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()

    async def fake_stream():
        yield make_chunk(tool_calls=[make_tool_call("echo", '{"text":"hi"}')])

    core.responses.append(fake_stream())
    client = ChatClient(core)

    prepared = await session.prepare(["call echo"], "openai", "gpt-4")
    stream = await session.stream(client, prepared)

    events = []
    async for event in stream:
        events.append(event)

    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert isinstance(final_events[0].result, ToolCallNeeded)
    assert final_events[0].result.tool_calls[0]["function"]["name"] == "echo"

    entries = store.read("test_tape") or []
    assert len(entries) == 1  # only user message, assistant deferred


@pytest.mark.asyncio
async def test_stream_to_error() -> None:
    """prepared → error: stream() yields ErrorEvent and records error."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()

    async def broken_stream():
        raise RuntimeError("stream broken")
        yield  # type: ignore[unreachable]

    core.responses.append(broken_stream())
    client = ChatClient(core)

    prepared = await session.prepare(["hello"], "openai", "gpt-4")
    stream = await session.stream(client, prepared)

    events = []
    async for event in stream:
        events.append(event)

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "stream broken" in error_events[0].error.message

    entries = store.read("test_tape") or []
    run_events = [e for e in entries if e.kind == "event" and e.payload["name"] == "run"]
    assert len(run_events) == 1
    assert run_events[0].payload["data"]["status"] == "error"
    assert "stream broken" in run_events[0].payload["data"]["error"]


@pytest.mark.asyncio
async def test_error_recovery_to_prepared() -> None:
    """error → idle → prepared: after run() error, prepare() works again."""
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    core.responses.append(RuntimeError("first error"))
    client = ChatClient(core)

    prepared1 = await session.prepare(["hello"], "openai", "gpt-4")

    with pytest.raises(RepublicError, match="first error"):
        await session.run(client, prepared1)

    # Recovery: prepare a new turn
    prepared2 = await session.prepare(["world"], "openai", "gpt-4")
    assert prepared2.run_id != prepared1.run_id
    assert len(prepared2.entries) == 1
    assert prepared2.entries[0].payload["content"] == "world"


@pytest.mark.asyncio
async def test_run_records_assistant_message() -> None:
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    core.responses.append(make_response(text="hi there"))
    client = ChatClient(core)

    prepared = await session.prepare(["hello"], "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, Finished)
    assert result.result.text == "hi there"

    entries = store.read("test_tape") or []
    assert len(entries) == 3  # user + assistant + run event
    assert entries[1].kind == "message"
    assert entries[1].payload["role"] == "assistant"
    assert entries[1].payload["content"] == "hi there"
    assert entries[2].kind == "event"
    assert entries[2].payload["name"] == "run"
    assert entries[2].payload["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_returns_tool_call_needed() -> None:
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    from tests.fakes import make_tool_call
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    client = ChatClient(core)

    prepared = await session.prepare(["call echo"], "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, ToolCallNeeded)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "echo"

    entries = store.read("test_tape") or []
    # With deferred recording, only prepare() entries are written until add_tool_results
    assert len(entries) == 1  # user message from prepare()


@pytest.mark.asyncio
async def test_add_tool_results_records_tool_result() -> None:
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = _make_session(store, TapeContext(anchor=None))

    core = FakeLLMCore()
    from tests.fakes import make_tool_call
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    core.responses.append(make_response(text="done"))
    client = ChatClient(core)

    prepared = await session.prepare(["call echo"], "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, ToolCallNeeded)

    next_prepared = await session.add_tool_results(
        result,
        [{"tool_call_id": "call_1", "content": "HI"}],
    )

    assert isinstance(next_prepared, PreparedChat)

    entries = store.read("test_tape") or []
    # Now all entries are recorded: user + assistant + tool_call + tool_result + run event
    assert len(entries) == 5
    tool_result_entries = [e for e in entries if e.kind == "tool_result"]
    assert len(tool_result_entries) == 1

    # Can continue the conversation
    final = await session.run(client, next_prepared)
    assert isinstance(final, Finished)
    assert final.result.text == "done"


@pytest.mark.asyncio
async def test_handoff_appends_anchor() -> None:
    store = InMemoryTapeStore()
    session = _make_session(store)

    async with session:
        session.handoff("checkpoint_1", anchor_state={"step": 1})

    entries = store.read("test_tape") or []
    assert len(entries) == 2  # anchor + handoff event
    assert entries[0].kind == "anchor"
    assert entries[0].payload["name"] == "checkpoint_1"


@pytest.mark.asyncio
async def test_append_event_records_framework_event() -> None:
    store = InMemoryTapeStore()
    session = _make_session(store)

    prepared = await session.prepare(["hello"], "openai", "gpt-4")
    entry = await session.append_event("loop.step", {"iteration": 1})

    assert entry.kind == "event"
    assert entry.payload["name"] == "loop.step"
    assert entry.payload["data"]["iteration"] == 1

    entries = store.read("test_tape") or []
    assert len(entries) == 1  # only event; user message deferred in prepared.entries
