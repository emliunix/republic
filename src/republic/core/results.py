"""Structured results and errors for Republic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from republic.core.errors import RepublicError

T = TypeVar("T")


@dataclass
class PreparedChat:
    """Execution configuration for one LLM API call.

    Does NOT contain messages — messages are always retrieved from tape
    by TapeSession.run(). This enforces the single abstraction: the tape
    is the source of truth for conversation state.

    Users with fixed messages should use InMemoryTapeStore.

    Note: stream vs non-stream is encoded by which method is called (chat() vs stream()),
    not by a field in PreparedChat.

    Not frozen: kwargs is a mutable dict (common practice for **kwargs capture).
    Treated as immutable in practice; mutations are not supported.
    """

    model: str
    provider: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int | None = None
    reasoning_effort: Any | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex)

    @property
    def core_kwargs(self) -> dict[str, Any]:
        return {
            "tools_payload": list(self.tools) or None,
            "model": self.model,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "kwargs": self.kwargs,
        }


@dataclass(frozen=True)
class LLMResult:
    """Complete outcome of a single LLM turn.

    This is the internal representation used by ChatClient.
    TapeSession.run() converts this to TurnResult (Finished | ToolCallNeeded)
    based on whether tool_calls are present.
    """

    request: PreparedChat
    text: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None
    usage: dict[str, Any] | None = None
    metadata_only: bool = False


@dataclass(frozen=True)
class TextEvent:
    content: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class FinalEvent(Generic[T]):
    result: T


@dataclass(frozen=True)
class ErrorEvent:
    error: RepublicError


# Union type for streaming events
type StreamEvent[T] = TextEvent | FinalEvent[T] | ErrorEvent


class AsyncStreamEvents(Generic[T]):
    """Wrapper over async generator that allows setup work in coroutine body.

    chat() returns LLMResult directly.
    stream() returns AsyncStreamEvents[LLMResult] which yields TextEvent | FinalEvent[LLMResult] | ErrorEvent.
    """

    def __init__(self, iterator: AsyncIterator[StreamEvent[T]]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> AsyncIterator[StreamEvent[T]]:
        return self._iterator


# =============================================================================
# TURN RESULT TYPES (orchestration layer)
# =============================================================================

@dataclass(frozen=True)
class Finished:
    """LLM turn completed with final response."""

    result: LLMResult


@dataclass(frozen=True)
class ToolCallNeeded:
    """LLM turn requires tool execution before continuing.

    Carries the tool calls extracted from the LLM result for the caller to execute.
    The session extracts _prepared internally to construct the next PreparedChat.
    Callers must NOT access _prepared directly — the only valid operation is
    passing this object to session.add_tool_results().
    """

    tool_calls: list[dict[str, Any]]
    result: LLMResult
    _prepared: PreparedChat


type TurnResult = Finished | ToolCallNeeded


# =============================================================================
# TOOL SCHEMA HELPER
# =============================================================================

def get_tool_schemas(tools: Any) -> list[dict[str, Any]]:
    """Extract JSON schemas from ToolInput for LLM API payload.

    Separates schema extraction from runnable tool execution.
    ChatClient receives schemas only; the agent loop receives both schemas and runnable tools.
    """
    from republic.tools.schema import normalize_tools

    toolset = normalize_tools(tools)
    return toolset.payload or []
