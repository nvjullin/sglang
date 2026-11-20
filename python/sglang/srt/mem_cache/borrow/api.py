"""The runtime <-> cache contract: borrows over a request's token positions.

A cache service owns every KV slot. A request holds one lease, which borrows:

- ``&`` (shared): the cached prefix ``[0, shared_len)``; read-only, pinned.
- ``&mut`` (exclusive): ``[shared_len, shared_len + mut_len)``; written only by
  this request's forwards.

Everything not borrowed belongs to the service, which may evict, move between
tiers or reuse it. Slot ids returned here index the device KV pools; the runtime
writes them into its page table (``req_to_token``) and never frees them itself.

Lockstep: a runtime that replicates its scheduler per rank must issue the same
calls, in the same order, with the same arguments, on every rank that shares a
service replica group; answers are then identical on all of them.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import msgspec
import torch


class CacheKey(msgspec.Struct, frozen=True):
    """Token ids whose KV a lease may share, in a namespace.

    Requests share cached KV only when both the token prefix and the namespace
    (e.g. LoRA id, cache salt, media digest) are equal.
    """

    token_ids: Sequence[int]
    namespace: tuple = ()


class LeaseAttrs(msgspec.Struct, frozen=True):
    """Inputs to the service's headroom and revocation choices."""

    priority: int = 0
    prompt_len: int = 0
    max_new_tokens: int = 0


class AcquireReq(msgspec.Struct, frozen=True):
    """Borrow positions ``[0, end)``: open a lease (``key`` given) or grow one.

    On open the service shares what it can of ``key`` and grants ``&mut`` on
    the rest. The span added by this call is at most ``max_mut_len`` and at
    least ``min_mut_len`` (``None``: all of what ``end`` and ``max_mut_len``
    allow). A span the service cuts short ends on a page boundary that is also
    a multiple of ``cut_align``. Otherwise the request is refused.
    """

    end: int
    lease: Optional[int] = None
    key: Optional[CacheKey] = None
    max_mut_len: Optional[int] = None
    min_mut_len: Optional[int] = None
    cut_align: Optional[int] = None
    attrs: LeaseAttrs = LeaseAttrs()


class AcquireStatus(enum.IntEnum):
    GRANTED = 0
    # Nothing changed; the lease (if any) keeps what it held.
    REFUSED = 1
    # The lease is gone and its slots are reclaimed; the runtime must not
    # schedule it again (also reported as a ``Revoked`` event).
    REVOKED = 2


class AcquireResult(msgspec.Struct, frozen=True):
    status: AcquireStatus
    lease: Optional[int] = None
    shared_len: int = 0
    # The lease now borrows [0, total_len).
    total_len: int = 0
    # Device slot ids of [0, shared_len); set on the grant that opens a lease.
    shared_slots: Optional[torch.Tensor] = None
    # Device slot ids of the positions this call added, in position order.
    mut_slots: Optional[torch.Tensor] = None
    # Of shared_len, tokens brought up from a lower tier for this lease.
    loaded_len: int = 0


class Ready(msgspec.Struct, frozen=True):
    lease: int


class Revoked(msgspec.Struct, frozen=True):
    lease: int


class Failed(msgspec.Struct, frozen=True):
    lease: int
    reason: str = ""


class AcquireBatch(msgspec.Struct, frozen=True):
    results: list
    events: list = []


class QueryResult(msgspec.Struct, frozen=True):
    # Tokens a lease opened now would share without any transfer.
    ready_len: int
    # Tokens cached in any tier (>= ready_len).
    cached_len: int


class Fence(msgspec.Struct, frozen=True):
    """In-stream transfers a batch's forward must wait for, layer by layer.

    ``consumer_index`` selects the transfers on the layer hook the service
    registered with the device KV pools; ``final_event`` completes after the
    last layer (for forwards that cannot wait per layer, e.g. graph replay).
    """

    consumer_index: int
    final_event: Any = None


class CacheStats(msgspec.Struct, frozen=True):
    device_slots: int
    device_free_slots: int
    device_evictable_slots: int
    host_slots: int = 0
    host_free_slots: int = 0
    open_leases: int = 0
    counters: dict = {}


class KVCacheService(ABC):
    """Borrow API: 4 verbs, a poll, 2 batch fences, events, control plane."""

    @abstractmethod
    def query(self, keys: Sequence[CacheKey]) -> list[QueryResult]:
        """Advisory sizes for ranking; no borrow is taken. A query is a hint
        that the keys will be acquired soon (the service may keep them warm)."""

    @abstractmethod
    def acquire(self, reqs: Sequence[AcquireReq]) -> AcquireBatch:
        """Open or grow leases, in order. Growth that does not fit may revoke
        other leases; their results come back as events."""

    @abstractmethod
    def release_mut(self, lease: int, token_ids: Sequence[int]) -> None:
        """Publish ``[0, len(token_ids))``: downgrade ``&mut`` to ``&``.

        ``token_ids`` are the tokens at those positions. Valid only after the
        ``end_batch`` of every batch that wrote them.
        """

    @abstractmethod
    def release(
        self,
        lease: int,
        *,
        from_len: int = 0,
        publish_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Publish ``publish_token_ids`` if given, then drop every borrow at
        positions ``>= from_len``. ``from_len == 0`` ends the lease."""

    @abstractmethod
    def poll(self) -> list:
        """Events due at this lockstep point."""

    @abstractmethod
    def begin_batch(self, batch_id: int, leases: Sequence[int]) -> Optional[Fence]:
        """Start the in-stream transfers the next forward depends on."""

    @abstractmethod
    def end_batch(self, batch_id: int, done_event: Any) -> list:
        """``done_event`` completes after the forward of ``batch_id`` on the
        forward stream. Slots dropped during the batch are reusable after it."""

    # ===== Control plane =====

    @abstractmethod
    def layer_hook(self) -> Any:
        """Object the device KV pools call per layer (``wait_until(layer)``) and
        the model worker arms per batch (``set_consumer(index)``)."""

    @abstractmethod
    def stats(self) -> CacheStats: ...

    @abstractmethod
    def flush(self) -> bool:
        """Drop all cached, unborrowed KV. False if leases are open."""

    @abstractmethod
    def close(self) -> None: ...
