"""Tape manager helpers for Republic."""

from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
import inspect
from typing import Any, cast

from republic.core.results import RepublicError
from republic.tape.context import TapeContext, build_messages
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.session import TapeSession
from republic.tape.store import (
    AsyncTapeStore,
    InMemoryTapeStore,
)


class AsyncTapeManager:
    """Async tape manager for async chat and tool-call paths."""

    def __init__(
        self,
        *,
        store: AsyncTapeStore | None = None,
        default_context: TapeContext | None = None,
    ) -> None:
        if store is None:
            inmem_store = InMemoryTapeStore()
            self._tape_store = inmem_store
        else:
            self._tape_store = store
        self._global_context = default_context or TapeContext()

    @property
    def default_context(self) -> TapeContext:
        return self._global_context

    @default_context.setter
    def default_context(self, value: TapeContext) -> None:
        self._global_context = value

    async def list_tapes(self) -> list[str]:
        return await self._tape_store.list_tapes()

    async def read_messages(self, tape: str, *, context: TapeContext | None = None) -> list[dict[str, Any]]:
        active_context = context or self._global_context
        query = TapeQuery(tape=tape)
        query = active_context.build_query(query)
        entries = await self._tape_store.fetch_all(query)
        messages = build_messages(entries, active_context)
        if inspect.isawaitable(messages):
            messages = await messages
        return messages

    async def append_entry(self, tape: str, entry: TapeEntry) -> None:
        await self._tape_store.append(tape, entry)

    async def reset_tape(self, tape: str) -> None:
        await self._tape_store.reset(tape)

    async def handoff(
        self,
        tape: str,
        name: str,
        *,
        state: dict[str, Any] | None = None,
        **meta: Any,
    ) -> list[TapeEntry]:
        entry = TapeEntry.anchor(name, state=state, **meta)
        event = TapeEntry.event("handoff", {"name": name, "state": state or {}}, **meta)
        await self._tape_store.append(tape, entry)
        await self._tape_store.append(tape, event)
        return [entry, event]
    
    @contextlib.asynccontextmanager
    async def session(
        self, tape_name: str,
        context: TapeContext | None = None,
    ) -> AsyncIterator[TapeSession]:
        try:
            yield TapeSession(tape_name, self._tape_store, context or self.default_context)
        finally:
            pass
