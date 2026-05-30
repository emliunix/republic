"""Tape manager helpers for Republic."""

from __future__ import annotations

import inspect
from typing import Any

from republic.tape.context import TapeContext, build_messages
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery
from republic.tape.store import (
    AsyncTapeStore,
)

class AsyncTapeManager:
    """Async tape manager for async chat and tool-call paths."""

    def __init__(
        self,
        *,
        store: AsyncTapeStore,
    ) -> None:
        self._tape_store = store

    async def list_tapes(self) -> list[str]:
        return await self._tape_store.list_tapes()

    async def read_messages(self, tape: str, *, context: TapeContext) -> list[dict[str, Any]]:
        active_context = context
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
