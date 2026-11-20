"""Recurrent-state slot copies for the HiCache service, on one service stream.

Built for a static ``MambaPool`` (conv + temporal state per slot) and an
optional ``MambaPoolHost`` host tier holding one state per host slot.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.transfer import StatePlane


class GpuStatePlane(StatePlane):
    def __init__(
        self,
        *,
        device_pool: Any,
        host_pool: Optional[Any],
        io_backend: str,
        device: torch.device,
    ):
        self.pool = device_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.host_num_slots = 0 if host_pool is None else host_pool.size
        self._host = PageAllocator(num_pages=self.host_num_slots)
        self._waited: Any = None
        self._unfenced = False

    def host_num_free(self) -> int:
        return self._host.num_free

    def alloc_host(self, num: int) -> Optional[list[int]]:
        return self._host.alloc(num)

    def free_host(self, slots: Sequence[int]) -> None:
        self._host.free(slots)

    def _ids(self, slots: Sequence[int]) -> torch.Tensor:
        # Pinned + non_blocking: a pageable copy would block this thread until
        # the stream (waiting on a forward) reached it.
        t = torch.tensor(list(slots), dtype=torch.int64).pin_memory()
        return t.to(self.device, non_blocking=True)

    def _launch(self, after: Any, op: Callable[[], None]) -> None:
        with torch.cuda.stream(self.stream):
            if after is not None and after is not self._waited:
                self.stream.wait_event(after)
                self._waited = after
            op()
        self._unfenced = True

    def zero(self, slot: int, *, after: Any) -> None:
        self._launch(after, lambda: self.pool.clear_slots(self._ids([slot])))

    def copy(self, src: int, dst: int, *, after: Any) -> None:
        self._launch(after, lambda: self.pool.copy_from(self._ids([src]), self._ids([dst])))

    def load(self, host_slot: int, slot: int, *, after: Any) -> None:
        def op() -> None:
            host_ids, dev_ids = self._ids([host_slot]), self._ids([slot])
            for layer in range(self.host_pool.num_mamba_layers):
                self.host_pool.load_to_device_per_layer(
                    self.pool, host_ids, dev_ids, layer, self.io_backend
                )

        self._launch(after, op)

    def backup(self, slot: int, host_slot: int, *, after: Any) -> None:
        self._launch(
            after,
            lambda: self.host_pool.backup_from_device_all_layer(
                self.pool, self._ids([host_slot]), self._ids([slot]), self.io_backend
            ),
        )

    def fence_event(self, *, also_after: Any) -> Any:
        if not self._unfenced and also_after is None:
            return None
        with torch.cuda.stream(self.stream):
            if also_after is not None:
                self.stream.wait_event(also_after)
            event = torch.cuda.Event()
            event.record(self.stream)
        self._unfenced = False
        return event

    def close(self) -> None:
        self.stream.synchronize()
        if self.host_pool is not None:
            self.host_pool.destroy()
