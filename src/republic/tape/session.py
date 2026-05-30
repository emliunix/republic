"""
Tape session view helpers for Republic.

```mermaid
stateDiagram-v2
    [*] --> idle : create session
    idle --> prepared : prepare()

    prepared --> finish : run() / stream()<br/>success, no tool_calls
    prepared --> toolcallneeded : run() / stream()<br/>success, with tool_calls
    prepared --> error : run() / stream()<br/>exception / ErrorEvent

    toolcallneeded --> prepared : add_tool_results()
    toolcallneeded --> error : add_tool_error()

    finish --> idle : turn complete
    error --> idle : error handled

    idle --> [*] : session close
```
"""

from __future__ import annotations

import uuid
import contextlib

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from republic.core.errors import ErrorKind, RepublicError
from republic.core.results import (
    AsyncStreamEvents,
    ErrorEvent,
    FinalEvent,
    Finished,
    LLMResult,
    PreparedChat,
    Prompt,
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

from republic.clients.chat import ChatClient
from republic.utils import ensure_drained


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
    _system_prompt: str | None
    _deferred_entries: list[TapeEntry]

    def __init__(
        self,
        name: str,
        store: AsyncTapeStore,
        context: TapeContext,
    ) -> None:
        self._name = name
        self._store = store
        self._manager = AsyncTapeManager(store=store)
        self._context = context
        self._system_prompt = None
        self._deferred_entries = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> TapeContext:
        return self._context

    async def prepare(
        self,
        provider: str,
        model: str,
        *,
        system_prompt: str | None = None,
        tools: ToolInput = None,
        max_tokens: int | None = None,
        reasoning_effort: Any | None = None,
        **kwargs: Any,
    ) -> PreparedChat:
        from republic.core.results import get_tool_schemas

        run_id = uuid.uuid4().hex
        metas = {"run_id": run_id}

        if system_prompt:
            await self._append_entry(TapeEntry.system(system_prompt, **metas))

        return PreparedChat(
            tools=get_tool_schemas(tools),
            model=model,
            provider=provider,
            entries=[],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            kwargs=kwargs,
            run_id=run_id,
        )

    async def _apply_prepared(self, prepared: PreparedChat):
        for entry in prepared.entries:
            await self._append_entry(entry)
        prepared.entries.clear()

    async def run(
        self,
        chat: ChatClient,
        prepared: PreparedChat,
    ) -> TurnResult:
        await self._apply_prepared(prepared)
        messages = await self._manager.read_messages(self._name, context=self._context)
        if prepared.system_prompt:
            messages = [{"role": "system", "content": prepared.system_prompt}, *messages]
        try:
            result = await chat.chat(prepared, messages)
        except RepublicError as e:
            await self._record_result_error(prepared, e)
            raise

        if result.tool_calls:
            return ToolCallNeeded(
                tool_calls=result.tool_calls,
                result=result,
            )

        await self._record_result(result)
        return Finished(result=result)

    async def stream(
        self,
        chat: ChatClient,
        prepared: PreparedChat,
    ) -> AsyncStreamEvents[TurnResult]:
        async def _wrapper() -> AsyncIterator[StreamEvent[TurnResult]]:
            await self._apply_prepared(prepared)
            messages = await self._manager.read_messages(self._name, context=self._context)
            if prepared.system_prompt:
                messages = [{"role": "system", "content": prepared.system_prompt}, *messages]
            stream = await chat.stream(prepared, messages)
            async with ensure_drained(stream) as iterable:
                async for event in iterable:
                    match event:
                        case TextEvent():
                            yield event
                        case FinalEvent(result) if result.tool_calls:
                            # NOTE: no _record_result, defer to add_tool_results
                            yield FinalEvent(result=ToolCallNeeded(
                                tool_calls=result.tool_calls,
                                result=result,
                            ))
                            break
                        case FinalEvent(result):
                            await self._record_result(result)
                            yield FinalEvent(result=Finished(result=result))
                            break
                        case ErrorEvent():
                            await self._record_result_error(prepared, event.error)
                            yield event
                            break
                async for event in iterable:
                    yield ErrorEvent(error=RepublicError(ErrorKind.UNKNOWN, "Received events after stream completion"))

        return AsyncStreamEvents(_wrapper())

    async def add_tool_results(
        self,
        needed: ToolCallNeeded,
        results: list[Any],
    ) -> PreparedChat:
        if len(results) != len(needed.tool_calls):
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                f"Expected {len(needed.tool_calls)} tool results, got {len(results)}"
            )
        await self._record_result(needed.result, tool_result=results)
        request = needed.result.request
        request.entries.extend(self._deferred_entries)
        self._deferred_entries.clear()
        return request

    async def add_tool_error(
        self,
        needed: ToolCallNeeded,
        error: Exception,
    ) -> None:
        await self._record_result_error(needed.result.request, error)

    def append_entry(self, entry: TapeEntry) -> None:
        self._deferred_entries.append(entry)
    
    async def append_event(
        self,
        name: str,
        data: dict[str, Any] | None = None,
        **meta: Any,
    ) -> TapeEntry:
        """
        Contrary to append_entry, append_event is immediate. 
        The rationale is events are like logs, and we don't need to take care of integrity
        """
        entry = TapeEntry.event(name=name, data=data, **meta)
        await self._append_entry(entry)
        return entry

    async def _record_result(
        self,
        result: LLMResult,
        tool_result: list[Any] | None = None,
    ) -> None:
        """
        Record a complete LLM call:
        - assistant with tool_calls
        - commentary tool_call entry
        - tool_result entries if any

        The invariant is if assistant entry has tool_calls, the tool_calls and tool_result are saved together
        """
        meta = {"run_id": result.request.run_id}

        assistant_payload: dict[str, Any] = {"role": "assistant"}
        if result.text:
            assistant_payload["content"] = result.text
        if result.reasoning:
            assistant_payload["reasoning_content"] = result.reasoning
        if result.tool_calls:
            assistant_payload["tool_calls"] = result.tool_calls

        if result.tool_calls:
            # check it early so that we don't record partial results
            if tool_result is None:
                raise ValueError("tool_result must be provided when result has tool_calls")

        await self._append_entry(TapeEntry.message(assistant_payload, **meta))

        if result.tool_calls:
            await self._append_entry(TapeEntry.tool_call(result.tool_calls, **meta))
        if tool_result is not None:
            await self._append_entry(TapeEntry.tool_result(tool_result, **meta))

        data: dict[str, Any] = { "status": "ok" }
        if result.usage:
            data["usage"] = result.usage
        data["provider"] = result.request.provider
        data["model"] = result.request.model

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

    async def __aenter__(self) -> TapeSession:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not exc_val:
            for entry in self._deferred_entries:
                await self._append_entry(entry)
            self._deferred_entries.clear()


def _ensure_text_prompts(prompt: Prompt) -> str:
    match prompt:
        case str():
            return prompt
        case [dict() as pt] if pt.get("type") == "text":
            return pt.get("content", "")
        case _:
            raise RepublicError(ErrorKind.INVALID_INPUT, f"Unsupported prompt: {prompt}. Expected str.")


def prompt_entry(prompt: Prompt, **metas: Any) -> TapeEntry:
    return TapeEntry.message({"role": "user", "content": _ensure_text_prompts(prompt)}, **metas)
