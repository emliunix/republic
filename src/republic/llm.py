"""Republic LLM facade."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

from republic.__about__ import DEFAULT_MODEL
from republic.auth import APIKeyResolver
from republic.clients._internal import InternalOps
from republic.clients.chat import ChatClient
from republic.clients.embedding import EmbeddingClient
from republic.core.errors import ErrorKind, RepublicError
from republic.core.execution import LLMCore
from republic.tape import (
    AsyncTapeManager,
    AsyncTapeStore,
    InMemoryTapeStore,
    TapeContext,
)
from republic.tools.executor import ToolExecutor


class LLM:
    """Developer-first LLM client powered by any-llm.

    DEPRECATED: This facade forces coupling of ChatClient, ToolExecutor, and TapeManager.
    Use ChatClient + TapeSession + AgentRunner explicitly instead.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        api_key: str | dict[str, str] | None = None,
        api_key_resolver: APIKeyResolver | None = None,
        api_base: str | dict[str, str] | None = None,
        client_args: dict[str, Any] | None = None,
        api_format: Literal["completion", "responses", "messages"] = "completion",
        verbose: int = 0,
        tape_store: AsyncTapeStore | None = None,
        context: TapeContext | None = None,
        error_classifier: Callable[[Exception], ErrorKind | None] | None = None,
    ) -> None:
        warnings.warn(
            "LLM facade is deprecated. Use ChatClient + TapeSession instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if verbose not in (0, 1, 2):
            raise RepublicError(ErrorKind.INVALID_INPUT, "verbose must be 0, 1, or 2")
        if max_retries < 0:
            raise RepublicError(ErrorKind.INVALID_INPUT, "max_retries must be >= 0")
        if api_format not in {"completion", "responses", "messages"}:
            raise RepublicError(
                ErrorKind.INVALID_INPUT,
                "api_format must be 'completion', 'responses', or 'messages'",
            )

        if not model:
            model = DEFAULT_MODEL
            warnings.warn(f"No model was provided, defaulting to {model}", UserWarning, stacklevel=2)

        resolved_provider, resolved_model = LLMCore.resolve_model_provider(model, provider)

        self._core = LLMCore(
            provider=resolved_provider,
            model=resolved_model,
            fallback_models=fallback_models or [],
            max_retries=max_retries,
            api_key=api_key,
            api_key_resolver=api_key_resolver,
            api_base=api_base,
            client_args=client_args or {},
            api_format=api_format,
            verbose=verbose,
            error_classifier=error_classifier,
        )
        if tape_store is None:
            async_tape_store = InMemoryTapeStore()
        else:
            async_tape_store = tape_store

        self._async_tape = AsyncTapeManager(store=async_tape_store, default_context=context)
        self._chat_client = ChatClient(self._core)
        self.embeddings = EmbeddingClient(self._core)
        self.tools = ToolExecutor()
        self._internal = InternalOps(self._core)

    @property
    def model(self) -> str:
        return self._core.model

    @property
    def provider(self) -> str:
        return self._core.provider

    @property
    def fallback_models(self) -> list[str]:
        return self._core.fallback_models

    @property
    def context(self) -> TapeContext:
        return self._async_tape.default_context

    @context.setter
    def context(self, value: TapeContext) -> None:
        self._async_tape.default_context = value

    async def embed_async(
        self,
        inputs: str | list[str],
        *,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ):
        return await self.embeddings.embed_async(inputs, model=model, provider=provider, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<LLM provider={self._core.provider} model={self._core.model} "
            f"fallback_models={self._core.fallback_models} max_retries={self._core.max_retries}>"
        )
