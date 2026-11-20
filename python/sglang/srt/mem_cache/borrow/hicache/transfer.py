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
