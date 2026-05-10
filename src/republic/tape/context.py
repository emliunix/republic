"""Context building for tape entries."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
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


class ReasoningStrategy(StrEnum):
    # preserves all reasoning content in assistant messages
    FULL = "full"
    # removes all reasoning content from assistant messages
    PRUNE = "prune"
    # keeps only the last turn's reasoning content
    LAST_TURN_ONLY = "last_turn_only"
    # keeps only reasoning content associated with tool calls
    TOOLCALLS_ONLY = "tool_calls_only"


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
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.PRUNE

    def build_query(self, query: TapeQuery) -> TapeQuery:
        if self.anchor is None:
            return query
        if isinstance(self.anchor, _LastAnchor):
            return query.last_anchor()
        return query.after_anchor(self.anchor)


def build_messages(entries: Iterable[TapeEntry], context: TapeContext) -> SelectedMessages:
    if context.select is not None:
        return context.select(entries, context)
    return _default_messages(entries, context.reasoning_strategy)


def _default_messages(entries: Iterable[TapeEntry], reasoning_strategy: ReasoningStrategy) -> list[dict[str, Any]]:
    """Build OpenAI-format messages from tape entries.

    Three-pass algorithm:
    1. Build: Merge tool_call entries into preceding assistant messages
    2. Find last user: Locate the boundary between history and active context
    3. Prune: Strip reasoning from historical assistant messages
    """
    messages = _build_full_messages(entries)

    match reasoning_strategy:
        case ReasoningStrategy.FULL:
            pass
        case ReasoningStrategy.PRUNE:
            for msg in messages:
                if msg.get("role") == "assistant":
                    msg.pop("reasoning_content", None)
        case ReasoningStrategy.LAST_TURN_ONLY:
            _prune_all_but_last_assistant(messages)
        case ReasoningStrategy.TOOLCALLS_ONLY:
            _prune_non_toolcalls(messages)

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


def _prune_all_but_last_assistant(messages: list[dict[str, Any]]) -> None:
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        len(messages),
    )
    for i in range(last_user):
        if messages[i].get("role") == "assistant":
            messages[i].pop("reasoning_content", None)
    

def _prune_non_toolcalls(messages: list[dict[str, Any]]) -> None:
    for is_user, grp in itertools.groupby(messages, lambda msg: msg.get("role") == "user"):
        group = list(grp)
        if not is_user and all(msg.get("role") != "tool" for msg in group):
            for msg in group:
                if msg.get("role") == "assistant":
                    msg.pop("reasoning_content", None)