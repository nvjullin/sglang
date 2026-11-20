"""Device <-> host KV copies for the HiCache service, on the service's streams.

Built for full-attention MHA / MLA device pools (with optional packed EAGLE
draft pools). Every GPU op runs on ``side_stream`` or the transfer engine's
streams, never on the caller thread's default stream.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional, Sequence

import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter, LayerLoadingEvent
from sglang.srt.mem_cache.borrow.api import Fence
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.transfer import DataPlane, TransferOp
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine

logger = logging.getLogger(__name__)


class _GpuOp(TransferOp):
    __slots__ = ("launched", "event")

    def __init__(self):
        self.launched = False
        self.event = None

    def wait(self) -> None:
        assert self.launched
        self.event.synchronize()


class ServiceLayerCounter(LayerDoneCounter):
    """Per-layer load events in a ring; a slot is reused only after its last
    load finished (blocking), so a late forward never waits on a newer load."""

    def __init__(self, num_layers: int, ring: int = 8):
        super().__init__(num_layers)
        self.num_counters = ring
        self.events = [LayerLoadingEvent(num_layers) for _ in range(ring)]

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        self.events[self.producer_index].finish_event.synchronize()
        return self.producer_index


class _HostMirror:
    """A device pool and the host pool holding its pages at the same ids."""

    __slots__ = ("device_pool", "host_pool", "staged_write")

    def __init__(self, device_pool: Any, host_pool: Any, io_backend: str):
        self.device_pool = device_pool
        self.host_pool = host_pool
        # The staged write-back kernel takes CPU host ids; the others GPU ids.
        self.staged_write = (
            io_backend == "kernel"
            and host_pool.layout in ("page_first", "page_unified")
            and host_pool.can_use_write_back_jit
        )


def _packed_draft_layer(layer_id: int, *, expected: int, host_layer: int) -> Optional[int]:
    return host_layer if layer_id == expected else None


class GpuDataPlane(DataPlane):
    """``packed_draft_pools`` are draft layers stored in the target host pool
    (NextN MTP); ``sidecars`` are (draft device pool, its own host pool) pairs
    whose ids follow the target's (EAGLE draft models)."""

    def __init__(
        self,
        *,
        kv_pool: Any,
        host_pool: Any,
        page_size: int,
        io_backend: str,
        device: torch.device,
        packed_draft_pools: Sequence[Any] = (),
        sidecars: Sequence[tuple[Any, Any]] = (),
    ):
        self.page_size = page_size
        self.kv_pool = kv_pool
        self.host_pool = host_pool
        self.io_backend = io_backend
        self.device = device
        self._mirrors = [_HostMirror(kv_pool, host_pool, io_backend)] + [
            _HostMirror(d, h, io_backend) for d, h in sidecars
        ]
        for m in self._mirrors:
            assert m.host_pool.logical_page_size == page_size
        self.host_num_pages = min(
            m.host_pool.logical_size // page_size for m in self._mirrors
        )
        self._packed_draft_pools = tuple(packed_draft_pools)
        self._host = PageAllocator(num_pages=self.host_num_pages)
        self.engine = L2TransferEngine(io_backend)
        self.side_stream = torch.cuda.Stream(device=device)
        self.counter = ServiceLayerCounter(kv_pool.layer_num)
        self._layer_count = kv_pool.layer_num
        self._queued: list[tuple[list[int], list[int], _GpuOp]] = []

    def _slots(self, pages: Sequence[int], *, on_device: bool) -> torch.Tensor:
        ps = self.page_size
        page_t = torch.tensor(list(pages), dtype=torch.int64)
        slots = (page_t[:, None] * ps + torch.arange(ps)).flatten()
        if not on_device:
            return slots
        return slots.pin_memory().to(self.device, non_blocking=True)

    def _transfers(
        self, host_pages: Sequence[int], device_pages: Sequence[int], *, write: bool
    ) -> list[L2Transfer]:
        device_ids = self._slots(device_pages, on_device=True)
        host_gpu = self._slots(host_pages, on_device=True)
        host_cpu = self._slots(host_pages, on_device=False)
        transfers = [
            L2Transfer(
                host_pool=m.host_pool,
                device_pool=m.device_pool,
                host_indices=host_cpu if write and m.staged_write else host_gpu,
                device_indices=device_ids,
            )
            for m in self._mirrors
        ]
        if not write:
            # A whole-layer backup already covers packed draft layers.
            target_layers = self._layer_count
            for depth, pool in enumerate(self._packed_draft_pools):
                transfers.append(
                    L2Transfer(
                        host_pool=self.host_pool,
                        device_pool=pool,
                        host_indices=host_gpu,
                        device_indices=device_ids,
                        layer_mapper=partial(
                            _packed_draft_layer,
                            expected=min(depth, target_layers - 1),
                            host_layer=target_layers + depth,
                        ),
                        is_draft=True,
                    )
                )
        return transfers

    def host_num_pages_free(self) -> int:
        return self._host.num_free

    def alloc_host(self, num_pages: int) -> Optional[list[int]]:
        return self._host.alloc(num_pages)

    def free_host(self, pages: Sequence[int]) -> None:
        self._host.free(pages)

    def backup(self, device_pages, host_pages, *, after: Any) -> TransferOp:
        op = _GpuOp()
        with torch.cuda.stream(self.side_stream):
            if after is not None:
                self.side_stream.wait_event(after)
            completion = self.engine.submit_device_to_host(
                self._transfers(host_pages, device_pages, write=True)
            )
        op.launched, op.event = True, completion.finish_event
        return op

    def queue_load(self, host_pages, device_pages) -> TransferOp:
        op = _GpuOp()
        self._queued.append((list(host_pages), list(device_pages), op))
        return op

    def launch_loads(self, *, after: Any) -> Optional[Fence]:
        if not self._queued:
            return None
        host_pages = [p for h, _, _ in self._queued for p in h]
        device_pages = [p for _, d, _ in self._queued for p in d]
        with torch.cuda.stream(self.side_stream):
            if after is not None:
                self.side_stream.wait_event(after)
            # Index copies go first: the transfer waits only on the start event.
            transfers = self._transfers(host_pages, device_pages, write=False)
            index = self.counter.update_producer()
            layer_events = self.counter.events[index]
            layer_events.start_event.record()
            self.engine.submit_host_to_device(
                transfers,
                start_event=layer_events.start_event,
                on_layer_done=layer_events.complete,
                transfer_layer_id_max=self._layer_count,
            )
        for _, _, op in self._queued:
            op.launched, op.event = True, layer_events.finish_event
        self._queued = []
        return Fence(consumer_index=index, final_event=layer_events.finish_event)

    def layer_hook(self) -> Any:
        return self.counter

    def close(self) -> None:
        self.side_stream.synchronize()
        self.engine.device_to_host_stream.synchronize()
        self.engine.host_to_device_stream.synchronize()
        # Unpins the host buffers; a later pool mapped at the same address
        # cannot register it otherwise.
        for m in self._mirrors:
            m.host_pool.destroy()
