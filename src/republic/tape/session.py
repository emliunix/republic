"""Tape session view helpers for Republic."""

from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
from typing import TYPE_CHECKING, Any, Generic, TypeVar
import uuid

from republic.core.errors import ErrorKind, RepublicError
from republic.core.results import (
    AsyncStreamEvents,
    ErrorEvent,
    FinalEvent,
    Finished,
    LLMResult,
    PreparedChat,
    StreamEvent,
    TextEvent,
    ToolCallNeeded,
    TurnResult,
)
from republic.tape.context import TapeContext
from republic.tape.entries import TapeEntry
from republic.tape.manager import AsyncTapeManager
from republic.tape.store import AsyncTapeStore
from republic.tools.schema import ToolInput

if TYPE_CHECKING:
    from republic.clients.chat import ChatClient


class TapeSession:
    """
    Manages a single LLM call lifecycle with tape. Owns serialization of all conversation turns.

    The session enforces a calling sequence of:
    - prepare() to construct a PreparedChat
    - run() or stream() to execute the PreparedChat and get results
    - add_tool_results() if needed to add tool results and get a new PreparedChat for the next turn

    Note:
    - append_entry() takes effect immediately
    - stream() is atomic w.r.t tape in that the accumulated entries will writes in batch on success or a error entry on error
    """
    _run_entry: TapeEntry | None

    def __init__(
        self,
        name: str,
        store: AsyncTapeStore,
        context: TapeContext | None = None,
    ) -> None:
        self._name = name
        self._store = store
        self._context = context or TapeContext()
        self._manager = AsyncTapeManager(store=store, default_context=context)

    @contextlib.asynccontextmanager
    @staticmethod
    async def create(name: str, store: AsyncTapeStore, context: TapeContext | None = None) -> AsyncIterator[TapeSession]:
        """Just to denote the session semantics"""
        session = TapeSession(name, store, context)
        try:
            yield session
        finally:
            pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> TapeContext:
        return self._context

    async def prepare(
        self,
        prompt: str,
        provider: str,
        model: str,
        *,
        system_prompt: str | None = None,
        tools: ToolInput = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> PreparedChat:
        from republic.core.results import get_tool_schemas

        run_id = uuid.uuid4().hex
        meta = {"run_id": run_id}

        if system_prompt:
            await self._append_entry(TapeEntry.system(system_prompt, **meta))
        await self._append_entry(TapeEntry.message({"role": "user", "content": prompt}, **meta))

        return PreparedChat(
            tools=get_tool_schemas(tools),
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            kwargs=kwargs,
            run_id=run_id,
        )

    async def run(
        self,
        chat: ChatClient,
        prepared: PreparedChat,
    ) -> TurnResult:
        messages = await self._manager.read_messages(self._name, context=self._context)
        try:
            result = await chat.chat(prepared, messages)
        except RepublicError as e:
            await self._record_result_error(prepared, e)
            raise

        await self._record_result(prepared, result)

        if result.tool_calls:
            return ToolCallNeeded(
                tool_calls=result.tool_calls,
                result=result,
                _prepared=prepared,
            )

        return Finished(result=result)

    async def stream(
        self,
        chat: ChatClient,
        prepared: PreparedChat,
    ) -> AsyncStreamEvents[TurnResult]:
        messages = await self._manager.read_messages(self._name, context=self._context)
        stream = await chat.stream(prepared, messages)

        async def _wrapper() -> AsyncIterator[StreamEvent[TurnResult]]:
            iterable = AsyncIteratorWrapper(stream.__aiter__())
            async for event in iterable:
                match event:
                    case TextEvent():
                        yield event
                    case FinalEvent(result) if result.tool_calls:
                        yield FinalEvent(result=ToolCallNeeded(
                            tool_calls=result.tool_calls,
                            result=result,
                            _prepared=prepared))
                        await self._record_result(prepared, result)
                        break
                    case FinalEvent(result):
                        yield FinalEvent(result=Finished(result=result))
                        await self._record_result(prepared, result)
                        break
                    case ErrorEvent():
                        yield event
                        await self._record_result_error(prepared, event.error)
                        break
            async for event in iterable:
                yield ErrorEvent(error=RepublicError(ErrorKind.UNKNOWN, "Received events after stream completion"))

        return AsyncStreamEvents(_wrapper())

    async def add_tool_results(
        self,
        needed: ToolCallNeeded,
        results: list[Any],
    ) -> PreparedChat:
        meta = {"run_id": needed._prepared.run_id}
        if len(results) != len(needed.tool_calls):
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"Expected {len(needed.tool_calls)} tool results, got {len(results)}"
            )
        await self._append_entry(TapeEntry.tool_result(results, **meta))
        return needed._prepared

    async def handoff(
        self,
        name: str,
        *,
        state: dict[str, Any] | None = None,
        **meta: Any,
    ) -> list[TapeEntry]:
        return await self._manager.handoff(self._name, name, state=state, **meta)

    async def append_event(
        self,
        prepared: PreparedChat,
        name: str,
        data: dict[str, Any] | None = None,
    ) -> TapeEntry:
        meta = {"run_id": prepared.run_id}
        entry = TapeEntry.event(name, data, **meta)
        await self._manager.append_entry(self._name, entry)
        return entry

    async def _record_result(
        self,
        prepared: PreparedChat,
        result: LLMResult,
    ) -> None:
        meta = {"run_id": prepared.run_id}

        assistant_payload: dict[str, Any] = {"role": "assistant"}
        if result.text:
            assistant_payload["content"] = result.text
        if result.reasoning:
            assistant_payload["reasoning_content"] = result.reasoning
        if result.tool_calls:
            assistant_payload["tool_calls"] = result.tool_calls

        await self._append_entry(TapeEntry.message(assistant_payload, **meta))

        if result.tool_calls:
            await self._append_entry(TapeEntry.tool_call(result.tool_calls, **meta))

        data: dict[str, Any] = { "status": "ok" }
        if result.usage:
            data["usage"] = result.usage
        data["provider"] = prepared.provider
        data["model"] = prepared.model

        await self._append_entry(TapeEntry.event("run", data, **meta))

    async def _record_result_error(
        self,
        prepared: PreparedChat,
        error: Exception,
    ) -> None:
        meta = {"run_id": prepared.run_id}

        data: dict[str, Any] = {
            "status": "error",
            "error": str(error),
            "provider": prepared.provider,
            "model": prepared.model,
        }

        await self._append_entry(TapeEntry.event("run", data, **meta))

    async def _append_entry(self, entry: TapeEntry) -> None:
        await self._manager.append_entry(self._name, entry)


T = TypeVar("T")


class AsyncIteratorWrapper(Generic[T]):
    def __init__(self, async_iterator: AsyncIterator[T]) -> None:
        self._async_iterator = async_iterator

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator