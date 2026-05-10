from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
import contextlib
from typing import Generic, TypeVar


T = TypeVar("T")


@contextlib.asynccontextmanager
async def ensure_drained(stream: AsyncIterable[T]) -> AsyncGenerator[AsyncIterable[T], None]:
    """Ensure that an async generator is fully drained."""
    wrapped = AsyncIteratorWrapper(stream.__aiter__())
    try:
        result = wrapped
        yield result
    finally:
        async for _ in wrapped:
            pass


class AsyncIteratorWrapper(Generic[T]):
    def __init__(self, async_iterator: AsyncIterator[T]) -> None:
        self._async_iterator = async_iterator

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator
