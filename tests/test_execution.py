from __future__ import annotations

import json
from typing import Any

import pytest

from republic.core.execution import LLMCore


def _make_core(client_args: dict[str, Any] | None = None) -> LLMCore:
    return LLMCore(
        provider="openai",
        model="gpt-4",
        fallback_models=[],
        max_retries=1,
        api_key=None,
        api_key_resolver=None,
        api_base=None,
        client_args=client_args or {},
        api_format="completion",
        verbose=0,
    )


class TestResolveClientArgs:
    def test_global_args_returned_when_no_provider_keys(self) -> None:
        core = _make_core(client_args={"extra_headers": {"X-Title": "App"}})
        assert core._resolve_client_args("zai") == {"extra_headers": {"X-Title": "App"}}
        assert core._resolve_client_args("openai") == {"extra_headers": {"X-Title": "App"}}

    def test_provider_specific_args(self) -> None:
        core = _make_core(client_args={
            "default": {"extra_headers": {"X-Title": "App"}},
            "zai": {"thinking": {"type": "enabled"}},
        })
        assert core._resolve_client_args("zai") == {"thinking": {"type": "enabled"}}
        assert core._resolve_client_args("openai") == {"extra_headers": {"X-Title": "App"}}

    def test_provider_specific_requires_default_key(self) -> None:
        # Without a "default" key, the dict is treated as global args.
        core = _make_core(client_args={
            "zai": {"thinking": {"type": "enabled"}},
        })
        assert core._resolve_client_args("zai") == {"zai": {"thinking": {"type": "enabled"}}}
        assert core._resolve_client_args("openai") == {"zai": {"thinking": {"type": "enabled"}}}

    def test_empty_args(self) -> None:
        core = _make_core(client_args={})
        assert core._resolve_client_args("zai") == {}

    def test_none_args(self) -> None:
        core = _make_core(client_args=None)
        assert core._resolve_client_args("zai") == {}

    def test_cache_key_differs_by_provider(self) -> None:
        core = _make_core(client_args={
            "default": {"timeout": 30},
            "zai": {"thinking": {"type": "enabled"}},
        })
        zai_key = core._freeze_cache_key("zai", None, None)
        openai_key = core._freeze_cache_key("openai", None, None)
        assert zai_key != openai_key

        zai_payload = json.loads(zai_key)
        openai_payload = json.loads(openai_key)
        assert zai_payload["client_args"] == {"thinking": {"type": "enabled"}}
        assert openai_payload["client_args"] == {"timeout": 30}
