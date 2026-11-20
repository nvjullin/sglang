"""What the cache service needs from a data plane (device <-> host copies)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from sglang.srt.mem_cache.borrow.api import Fence


class TransferOp(ABC):
    """A copy the service may have to wait for before reusing its pages."""

    launched: bool

    @abstractmethod
    def wait(self) -> None:
        """Block the calling thread until the copy completed. Launched ops only."""


class DataPlane(ABC):
    """Host tier storage plus device <-> host copies, all on service streams."""

    page_size: int
    host_num_pages: int

    @abstractmethod
    def host_num_pages_free(self) -> int: ...

    @abstractmethod
    def alloc_host(self, num_pages: int) -> Optional[list[int]]: ...

    @abstractmethod
    def free_host(self, pages: Sequence[int]) -> None: ...

    @abstractmethod
    def backup(
        self, device_pages: Sequence[int], host_pages: Sequence[int], *, after: Any
    ) -> TransferOp:
        """Launch device -> host now, after the event ``after`` (may be None)."""

    @abstractmethod
    def queue_load(
        self, host_pages: Sequence[int], device_pages: Sequence[int]
    ) -> TransferOp:
        """Queue host -> device; it launches with the next ``launch_loads``."""

    @abstractmethod
    def launch_loads(self, *, after: Any) -> Optional[Fence]:
        """Launch every queued load after the event ``after``; None if none queued.

        Every per-layer completion is enqueued before this returns.
        """

    @abstractmethod
    def layer_hook(self) -> Any: ...

    @abstractmethod
    def close(self) -> None: ...


class StatePlane(ABC):
    """Recurrent-state slot copies, all on one service stream.

    Every op launches at once, after the event ``after`` (the latest
    ``end_batch`` done event, so forwards that wrote or read the slots come
    first); ops run in launch order. A forward that starts after an op waits on
    ``fence_event``, so a freed slot is reusable as soon as its last op launched.
    """

    host_num_slots: int

    @abstractmethod
    def host_num_free(self) -> int: ...

    @abstractmethod
    def alloc_host(self, num: int) -> Optional[list[int]]: ...

    @abstractmethod
    def free_host(self, slots: Sequence[int]) -> None: ...

    @abstractmethod
    def zero(self, slot: int, *, after: Any) -> None: ...

    @abstractmethod
    def copy(self, src: int, dst: int, *, after: Any) -> None:
        """Device slot -> device slot."""

    @abstractmethod
    def load(self, host_slot: int, slot: int, *, after: Any) -> None:
        """Host slot -> device slot."""

    @abstractmethod
    def backup(self, slot: int, host_slot: int, *, after: Any) -> None:
        """Device slot -> host slot."""

    @abstractmethod
    def fence_event(self, *, also_after: Any) -> Any:
        """An event completing after every op launched so far (and after
        ``also_after`` if given); None when nothing launched since the last
        fence and ``also_after`` is None."""

    @abstractmethod
    def close(self) -> None: ...
