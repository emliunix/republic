"""Tests for TapeSession v2 API."""

from __future__ import annotations

import pytest

from republic.clients.chat import ChatClient
from republic.core.results import (
    Finished,
    LLMResult,
    PreparedChat,
    ToolCallNeeded,
)
from republic.tape.entries import TapeEntry
from republic.tape.session import TapeSession
from republic.tape.store import InMemoryTapeStore

from .test_chat_client_v2 import FakeLLMCore
from .fakes import make_response


@pytest.fixture
async def session() -> TapeSession:
    store = InMemoryTapeStore()
    return TapeSession("test_tape", store)


@pytest.mark.asyncio
async def test_prepare_records_user_message() -> None:
    store = InMemoryTapeStore()
    session = TapeSession("test_tape", store)

    prepared = await session.prepare("hello", "openai", "gpt-4", system_prompt="You are helpful")

    assert isinstance(prepared, PreparedChat)
    entries = store.read("test_tape") or []
    assert len(entries) == 2  # system + user
    assert entries[0].kind == "system"
    assert entries[0].payload["content"] == "You are helpful"
    assert entries[1].kind == "message"
    assert entries[1].payload["role"] == "user"
    assert entries[1].payload["content"] == "hello"


@pytest.mark.asyncio
async def test_run_records_assistant_message() -> None:
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = TapeSession("test_tape", store, context=TapeContext(anchor=None))

    core = FakeLLMCore()
    core.responses.append(make_response(text="hi there"))
    client = ChatClient(core)

    prepared = await session.prepare("hello", "openai", "gpt-4")
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
    session = TapeSession("test_tape", store, context=TapeContext(anchor=None))

    core = FakeLLMCore()
    from tests.fakes import make_tool_call
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    client = ChatClient(core)

    prepared = await session.prepare("call echo", "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, ToolCallNeeded)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["function"]["name"] == "echo"

    entries = store.read("test_tape") or []
    # user + assistant(with tool_calls) + tool_call(dual save) + run event
    assert len(entries) == 4
    assert entries[1].kind == "message"
    assert entries[1].payload.get("tool_calls") is not None
    assert entries[2].kind == "tool_call"


@pytest.mark.asyncio
async def test_add_tool_results_records_tool_result() -> None:
    from republic.tape.context import TapeContext
    store = InMemoryTapeStore()
    session = TapeSession("test_tape", store, context=TapeContext(anchor=None))

    core = FakeLLMCore()
    from tests.fakes import make_tool_call
    core.responses.append(make_response(tool_calls=[make_tool_call("echo", '{"text":"hi"}')]))
    core.responses.append(make_response(text="done"))
    client = ChatClient(core)

    prepared = await session.prepare("call echo", "openai", "gpt-4")
    result = await session.run(client, prepared)

    assert isinstance(result, ToolCallNeeded)

    next_prepared = await session.add_tool_results(
        result,
        [{"tool_call_id": "call_1", "content": "HI"}],
    )

    assert isinstance(next_prepared, PreparedChat)

    entries = store.read("test_tape") or []
    tool_result_entries = [e for e in entries if e.kind == "tool_result"]
    assert len(tool_result_entries) == 1

    # Can continue the conversation
    final = await session.run(client, next_prepared)
    assert isinstance(final, Finished)
    assert final.result.text == "done"


@pytest.mark.asyncio
async def test_handoff_appends_anchor() -> None:
    store = InMemoryTapeStore()
    session = TapeSession("test_tape", store)

    await session.handoff("checkpoint_1", state={"step": 1})

    entries = store.read("test_tape") or []
    assert len(entries) == 2  # anchor + handoff event
    assert entries[0].kind == "anchor"
    assert entries[0].payload["name"] == "checkpoint_1"


@pytest.mark.asyncio
async def test_append_event_records_framework_event() -> None:
    store = InMemoryTapeStore()
    session = TapeSession("test_tape", store)

    prepared = await session.prepare("hello", "openai", "gpt-4")
    entry = await session.append_event(prepared, "loop.step", {"iteration": 1})

    assert entry.kind == "event"
    assert entry.payload["name"] == "loop.step"
    assert entry.payload["data"]["iteration"] == 1

    entries = store.read("test_tape") or []
    assert len(entries) == 2
