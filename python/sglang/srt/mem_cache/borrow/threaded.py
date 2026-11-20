"""Host a cache service on its own thread, as an external service would run.

Calls run one at a time, in the caller's order. Calls whose answer the caller
needs block on it; the others (publishing, releasing, batch completion) return
at once. The first failure stops the service, and every later call raises it.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.borrow.api import (
    AcquireBatch,
    AcquireReq,
    CacheKey,
    CacheStats,
    Fence,
    KVCacheService,
    QueryResult,
    StateAnchor,
)

logger = logging.getLogger(__name__)


class CacheServiceError(RuntimeError):
    pass


class ThreadedService(KVCacheService):
    def __init__(
        self,
        factory: Callable[[], KVCacheService],
        *,
        cuda_device: Optional[int] = None,
        name: str = "kv-cache-service",
    ):
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._error: Optional[BaseException] = None
        self._closed = False
        started: Future = Future()
        self._thread = threading.Thread(
            target=self._serve,
            kwargs=dict(factory=factory, cuda_device=cuda_device, started=started),
            name=name,
            daemon=True,
        )
        self._thread.start()
        started.result()

    def _serve(
        self,
        *,
        factory: Callable[[], KVCacheService],
        cuda_device: Optional[int],
        started: Future,
    ) -> None:
        try:
            if cuda_device is not None:
                torch.cuda.set_device(cuda_device)
            impl = factory()
        except BaseException as e:
            started.set_exception(e)
            return
        started.set_result(None)
        while True:
            item = self._queue.get()
            if item is None:
                return
            method, args, kwargs, reply = item
            if self._error is not None:
                if reply is not None:
                    reply.set_exception(self._failure())
                continue
            try:
                result = getattr(impl, method)(*args, **kwargs)
            except BaseException as e:
                logger.exception("cache service failed in %s", method)
                self._error = e
                if reply is not None:
                    reply.set_exception(self._failure())
                continue
            if reply is not None:
                reply.set_result(result)

    def _failure(self) -> CacheServiceError:
        err = CacheServiceError(f"cache service failed: {self._error!r}")
        err.__cause__ = self._error
        return err

    def _check(self) -> None:
        if self._error is not None:
            raise self._failure()
        if self._closed:
            raise CacheServiceError("cache service is closed")

    def _call(self, method: str, *args, **kwargs) -> Any:
        self._check()
        reply: Future = Future()
        self._queue.put((method, args, kwargs, reply))
        return reply.result()

    def _post(self, method: str, *args, **kwargs) -> None:
        self._check()
        self._queue.put((method, args, kwargs, None))

    # ===== Verbs =====

    def query(self, keys: Sequence[CacheKey]) -> list[QueryResult]:
        return self._call("query", keys)

    def acquire(self, reqs: Sequence[AcquireReq]) -> AcquireBatch:
        return self._call("acquire", reqs)

    def release_mut(
        self,
        lease: int,
        token_ids: Sequence[int],
        *,
        state: Optional[StateAnchor] = None,
    ) -> None:
        self._post("release_mut", lease, token_ids, state=state)

    def release(
        self,
        lease: int,
        *,
        from_len: int = 0,
        publish_token_ids: Optional[Sequence[int]] = None,
        state: Optional[StateAnchor] = None,
    ) -> None:
        self._post(
            "release",
            lease,
            from_len=from_len,
            publish_token_ids=publish_token_ids,
            state=state,
        )

    def poll(self) -> list:
        return self._call("poll")

    def begin_batch(self, batch_id: int, leases: Sequence[int]) -> Optional[Fence]:
        return self._call("begin_batch", batch_id, leases)

    def end_batch(self, batch_id: int, done_event: Any) -> list:
        self._post("end_batch", batch_id, done_event)
        return []

    # ===== Control plane =====

    def layer_hook(self) -> Any:
        return self._call("layer_hook")

    def stats(self) -> CacheStats:
        return self._call("stats")

    def flush(self) -> bool:
        return self._call("flush")

    def sync(self) -> None:
        """Wait until every earlier call has run (and raise its failure)."""
        self._call("stats")

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._error is None:
                self._call("close")
        finally:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
