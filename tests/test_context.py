"""Tests for reasoning pruning helpers in republic.tape.context."""

from __future__ import annotations

import pytest

from republic.tape.context import _prune_all_but_last_assistant, _prune_non_toolcalls, _build_full_messages
from republic.tape.entries import TapeEntry


class TestPruneAllButLastAssistant:
    """_prune_all_but_last_assistant keeps reasoning only on the last assistant turn."""

    def test_empty_list(self) -> None:
        messages: list[dict] = []
        _prune_all_but_last_assistant(messages)
        assert messages == []

    def test_single_user_no_pruning(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        _prune_all_but_last_assistant(messages)
        assert messages == [{"role": "user", "content": "hi"}]

    def test_keeps_reasoning_after_last_user(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
        ]
        _prune_all_but_last_assistant(messages)
        assert "reasoning_content" not in messages[1]  # before last user → removed
        assert messages[3]["reasoning_content"] == "r2"  # after last user → kept

    def test_removes_reasoning_before_last_user(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
        ]
        _prune_all_but_last_assistant(messages)
        assert "reasoning_content" not in messages[1]  # before last user → removed
        assert messages[3]["reasoning_content"] == "r2"  # after last user → kept

    def test_multiple_assistants_before_last_user(self) -> None:
        messages = [
            {"role": "assistant", "content": "pre", "reasoning_content": "pre-r"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a3", "reasoning_content": "r3"},
        ]
        _prune_all_but_last_assistant(messages)
        assert "reasoning_content" not in messages[0]
        assert "reasoning_content" not in messages[2]
        assert "reasoning_content" not in messages[3]
        assert messages[5]["reasoning_content"] == "r3"

    def test_no_reasoning_key_idempotent(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        _prune_all_but_last_assistant(messages)
        assert messages == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

    def test_tool_messages_ignored(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
        ]
        _prune_all_but_last_assistant(messages)
        assert "reasoning_content" not in messages[1]
        assert messages[4]["reasoning_content"] == "r2"


class TestPruneNonToolcalls:
    """_prune_non_toolcalls keeps reasoning only on assistant turns that involved tool calls."""

    def test_empty_list(self) -> None:
        messages: list[dict] = []
        _prune_non_toolcalls(messages)
        assert messages == []

    def test_single_user_no_pruning(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        _prune_non_toolcalls(messages)
        assert messages == [{"role": "user", "content": "hi"}]

    def test_removes_reasoning_when_no_tool_calls(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        ]
        _prune_non_toolcalls(messages)
        assert "reasoning_content" not in messages[1]

    def test_keeps_reasoning_when_tool_calls_present(self) -> None:
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
        ]
        _prune_non_toolcalls(messages)
        assert messages[1]["reasoning_content"] == "r1"

    def test_mixed_turns(self) -> None:
        """Turn 1: no tool calls → reasoning stripped.
        Turn 2: has tool calls → reasoning kept.
        Turn 3: no tool calls → reasoning stripped.
        """
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3", "reasoning_content": "r3"},
        ]
        _prune_non_toolcalls(messages)
        assert "reasoning_content" not in messages[1]  # turn 1, no tools
        assert messages[3]["reasoning_content"] == "r2"  # turn 2, has tools
        assert "reasoning_content" not in messages[6]  # turn 3, no tools

    def test_multiple_assistants_in_same_turn(self) -> None:
        """Multiple consecutive assistants without tools → all stripped."""
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
        ]
        _prune_non_toolcalls(messages)
        assert "reasoning_content" not in messages[1]
        assert "reasoning_content" not in messages[2]

    def test_no_reasoning_key_idempotent(self) -> None:
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        _prune_non_toolcalls(messages)
        assert messages == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]


class TestBuildFullMessages:
    """_build_full_messages excludes system entries — they are for logging only."""

    def test_system_entries_excluded(self) -> None:
        """System tape entries should NOT appear in API messages."""
        entries = [
            TapeEntry.system("You are a helpful assistant"),
            TapeEntry.message({"role": "user", "content": "hello"}),
        ]
        messages = _build_full_messages(entries)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_old_system_entries_without_role_excluded(self) -> None:
        """Old system entries lacking 'role' field must not cause API errors."""
        entries = [
            TapeEntry(id=0, kind="system", payload={"content": "old system"}),
            TapeEntry.message({"role": "user", "content": "hi"}),
        ]
        messages = _build_full_messages(entries)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_message_entries_included(self) -> None:
        """Regular message entries are included in API messages."""
        entries = [
            TapeEntry.message({"role": "user", "content": "q1"}),
            TapeEntry.message({"role": "assistant", "content": "a1"}),
        ]
        messages = _build_full_messages(entries)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
