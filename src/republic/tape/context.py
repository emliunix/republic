"""Context building for tape entries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
import itertools
from typing import Any, TypeAlias

from republic.core.errors import ErrorKind, RepublicError
from republic.tape.entries import TapeEntry
from republic.tape.query import TapeQuery


class _LastAnchor:
    def __repr__(self) -> str:
        return "LAST_ANCHOR"


LAST_ANCHOR = _LastAnchor()
AnchorSelector: TypeAlias = str | None | _LastAnchor
SelectedMessages: TypeAlias = list[dict[str, Any]] | Coroutine[Any, Any, list[dict[str, Any]]]
ContextSelector: TypeAlias = Callable[[Iterable[TapeEntry], "TapeContext"], SelectedMessages]


@dataclass(frozen=True)
class TapeContext:
    """Rules for selecting tape entries into a prompt context.

    anchor: LAST_ANCHOR for the most recent anchor, None for the full tape, or an anchor name.
    select: Optional selector called after anchor slicing that returns messages.
    state: Optional state dictionary to be passed along with the context.
    """

    anchor: AnchorSelector = LAST_ANCHOR
    select: ContextSelector | None = None
    state: dict[str, Any] = field(default_factory=dict)

    def build_query(self, query: TapeQuery) -> TapeQuery:
        if self.anchor is None:
            return query
        if isinstance(self.anchor, _LastAnchor):
            return query.last_anchor()
        return query.after_anchor(self.anchor)


def build_messages(entries: Iterable[TapeEntry], context: TapeContext) -> SelectedMessages:
    if context.select is not None:
        return context.select(entries, context)
    return _default_messages(entries)


def _default_messages(entries: Iterable[TapeEntry]) -> list[dict[str, Any]]:
    """Build OpenAI-format messages from tape entries.

    Three-pass algorithm:
    1. Build: Merge tool_call entries into preceding assistant messages
    2. Find last user: Locate the boundary between history and active context
    3. Prune: Strip reasoning from historical assistant messages
    """
    messages = _build_full_messages(entries)

    # Pass 2: Find last user message — everything before it is historical context
    # that can have reasoning stripped (R2). Everything after must preserve it (I1).
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        len(messages),
    )

    # Pass 3: Strip reasoning from non tool call assistant messages
    for is_user, grp in itertools.groupby(messages[:last_user], lambda msg: msg.get("role") == "user"):
        # for a complete assistant turn, if there's a tool call, we keep reasoning, otherwise strip it.
        # the assistant turn is messages between 2 user messages, or that stops after the last user message
        if not is_user and all(msg.get("role") != "tool" for msg in grp):
            for msg in grp:
                if msg.get("role") == "assistant":
                    msg.pop("reasoning_content", None)

    return messages


def _build_full_messages(entries: Iterable[TapeEntry]) -> list[dict[str, Any]]:
    """Pass 1: Build all messages with reasoning preserved.

    Supports dual-save format:
    - Assistant messages may already contain tool_calls in their payload (new format)
    - Falls back to merging separate tool_call entries (old format, backward compat)
    """
    messages: list[dict[str, Any]] = []
    entry_list = list(entries)
    last_tool_calls = None

    for i, entry in enumerate(entry_list):
        if entry.kind == "message":
            payload = dict(entry.payload) if isinstance(entry.payload, dict) else {}
            messages.append(payload)

        elif entry.kind == "tool_call":
            last_tool_calls = _assert_list(entry.payload.get("calls", []))
        elif entry.kind == "tool_result":
            results = entry.payload.get("results", []) if isinstance(entry.payload, dict) else []
            if last_tool_calls is None:
                raise RepublicError(ErrorKind.INVALID_INPUT, "Found tool_result without preceding tool_call.")
            if len(results) != len(last_tool_calls):
                raise RepublicError(ErrorKind.INVALID_INPUT, "tool_result calls must match preceding tool_call")
            for call, result in zip(last_tool_calls, results):
                tool_message = {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": call.get("id"),
                }
                messages.append(tool_message)
                last_tool_calls = None

    return messages


def _assert_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise RepublicError(ErrorKind.INVALID_INPUT, f"Expected a list, got {type(value).__name__}")
    return value