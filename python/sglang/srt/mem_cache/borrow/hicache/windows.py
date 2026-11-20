"""Sliding-window pages of the HiCache service: a second device id space.

Every borrowed position has a full KV slot; positions a forward may still read
through its window also have a window slot (same in-page offset, in a window
page). A lease owns the window pages of its private pages and pins the cached
window pages of its shared prefix's trailing window; the runtime drops window
borrows behind the window with ``release(lease, window_below=p)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.radix_index import Node, PrefixIndex

if TYPE_CHECKING:
    from sglang.srt.mem_cache.borrow.hicache.service import Lease


class WindowConfig(msgspec.Struct, frozen=True):
    # Window pages [1, num_pages]; page 0 holds the padding slot.
    num_pages: int
    # A forward reads positions >= its first position - window_tokens.
    window_tokens: int
    # Positions past the window a lease may hold before the runtime releases
    # them (its retirement interval); sizes the decode reservation.
    retire_interval: int = 128


class WindowPages:
    def __init__(self, *, config: WindowConfig, index: PrefixIndex, page_size: int):
        self.cfg = config
        self.index = index
        self.ps = page_size
        self.alloc = PageAllocator(num_pages=config.num_pages, first_page=1)

    # ===== Capacity =====

    def live(self, lease: Lease) -> int:
        return sum(1 for w in lease.wpages if w is not None)

    def lease_cap(self) -> int:
        """Window pages a decoding lease holds at most between retirements."""
        span = self.cfg.window_tokens + self.cfg.retire_interval
        return -(-span // self.ps) + 1

    def reserve(self, lease: Lease) -> int:
        return max(0, self.lease_cap() - self.live(lease))

    def headroom(self, reserved: int) -> int:
        return self.alloc.num_free + self.index.evictable_window_pages - reserved

    def take(self, num: int) -> Optional[list[int]]:
        """Allocate window pages, tombstoning unpinned cached ones if needed."""
        short = num - self.alloc.num_free
        if short > 0:
            self.alloc.free(self.index.evict_windows(short))
        return self.alloc.alloc(num)

    # ===== Shared prefixes =====

    def first_page(self, boundary: int) -> int:
        """First page a forward starting at ``boundary`` reads through the window."""
        return max(0, boundary - self.cfg.window_tokens) // self.ps

    def shared_ok(self, nodes: Sequence[Node], k: int, n_dev: int) -> bool:
        """Whether a prefix of ``nodes[:k]`` has window KV for its trailing
        window: device window pages on the first ``n_dev`` nodes, host window
        pages on the ones to be loaded."""
        boundary = sum(len(n.tokens) for n in nodes[:k])
        first, page = self.first_page(boundary), 0
        for i, node in enumerate(nodes[:k]):
            num = len(node.tokens) // self.ps
            if page + num > first:
                pages = node.window_pages if i < n_dev else node.host_window_pages
                lo = max(0, first - page)
                if pages is None or any(w is None for w in pages[lo:num]):
                    return False
            page += num
        return True

    def _device_windows(self, node: Node) -> list:
        # None: every page of the node is a tombstone.
        if node.window_pages is None:
            return [None] * (len(node.tokens) // self.ps)
        return node.window_pages

    def pin_window_of(self, nodes: Sequence[Node], boundary: int, start_page: int) -> None:
        """Pin the trailing-window pages of device-resident ``nodes``, which
        start at page ``start_page`` of a prefix ending at ``boundary``. Done
        before allocating, so that allocation cannot tombstone them."""
        first, page, pins = self.first_page(boundary), start_page, []
        for node in nodes:
            for w in self._device_windows(node):
                if page >= first and w is not None:
                    pins.append(w)
                page += 1
        self.index.pin_windows(pins)

    def set_shared(self, lease: Lease, nodes: Sequence[Node], boundary: int) -> None:
        """Record the pinned trailing window of the shared prefix ``nodes``."""
        first = self.first_page(boundary)
        pages = [w for n in nodes for w in self._device_windows(n)]
        lease.wpages = [w if i >= first else None for i, w in enumerate(pages)]
        lease.wprivate = [False] * len(pages)
        lease.window_floor = first

    # ===== Lease borrows =====

    def add_private(self, lease: Lease, pages: list[int]) -> None:
        lease.wpages.extend(pages)
        lease.wprivate.extend([True] * len(pages))

    def adopt(self, lease: Lease, adopted: Sequence[bool]) -> None:
        """Pages the index adopted on publish took their window pages too; the
        lease keeps reading them, so it pins them instead of owning them."""
        pins = []
        for i, took in enumerate(adopted):
            w = lease.wpages[i]
            if took and w is not None and lease.wprivate[i]:
                lease.wprivate[i] = False
                pins.append(w)
        self.index.pin_windows(pins)

    def _drop(self, lease: Lease, lo: int, hi: int) -> None:
        private, pinned = [], []
        for i in range(lo, hi):
            w = lease.wpages[i]
            if w is None:
                continue
            (private if lease.wprivate[i] else pinned).append(w)
            lease.wpages[i] = None
        self.alloc.free(private)
        self.index.unpin_windows(pinned)

    def release_below(self, lease: Lease, position: int) -> None:
        upto = min(position // self.ps, len(lease.wpages))
        if upto > lease.window_floor:
            self._drop(lease, lease.window_floor, upto)
            lease.window_floor = upto

    def truncate(self, lease: Lease, keep_pages: int) -> None:
        self._drop(lease, keep_pages, len(lease.wpages))
        del lease.wpages[keep_pages:], lease.wprivate[keep_pages:]

    def end(self, lease: Lease) -> None:
        self._drop(lease, 0, len(lease.wpages))

    def slots(self, wpages: Sequence[Optional[int]], start: int, end: int) -> torch.Tensor:
        """Window slot ids of positions [start, end); 0 where none is borrowed."""
        ps = self.ps
        if end <= start:
            return torch.empty((0,), dtype=torch.int64)
        first, last = start // ps, (end - 1) // ps
        page_t = torch.tensor(
            [-1 if w is None else w for w in wpages[first : last + 1]], dtype=torch.int64
        )
        slots = (page_t[:, None] * ps + torch.arange(ps)).flatten()
        slots[(page_t < 0).repeat_interleave(ps)] = 0
        off = first * ps
        return slots[start - off : end - off]
