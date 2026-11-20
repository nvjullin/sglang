"""HiCache behind the borrow API: device tier + host tier, write-through.

Every decision is a deterministic function of the call sequence: completion of
a copy is only observed by blocking on it, so replicas fed the same calls stay
identical without collectives.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

import msgspec
import torch

from sglang.srt.mem_cache.borrow.api import (
    AcquireBatch,
    AcquireReq,
    AcquireResult,
    AcquireStatus,
    CacheKey,
    CacheStats,
    Fence,
    KVCacheService,
    LeaseAttrs,
    QueryResult,
    Revoked,
)
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.radix_index import Node, PrefixIndex
from sglang.srt.mem_cache.borrow.hicache.transfer import DataPlane

logger = logging.getLogger(__name__)


class ServiceConfig(msgspec.Struct, frozen=True):
    page_size: int
    # Device slots [page_size, (num_device_pages + 1) * page_size); page 0 holds
    # the padding slot.
    num_device_pages: int
    init_new_token_ratio: float = 0.7
    min_new_token_ratio: float = 0.1
    new_token_ratio_decay: float = 0.001


class Lease:
    __slots__ = (
        "id",
        "namespace",
        "node",
        "shared_len",
        "total_len",
        "pages",
        "private",
        "attrs",
    )

    def __init__(self, *, id: int, namespace: tuple, node: Node, attrs: LeaseAttrs):
        self.id = id
        self.namespace = namespace
        # Index node at the end of the shared prefix; holds one lock.
        self.node = node
        self.shared_len = 0
        # Positions [0, total_len) are borrowed: & below shared_len, &mut above.
        self.total_len = 0
        # Device page per page of positions; private[i] means the lease frees it.
        self.pages: list[int] = []
        self.private: list[bool] = []
        self.attrs = attrs

    def decode_phase(self) -> bool:
        return self.total_len >= self.attrs.prompt_len


class HiCacheService(KVCacheService):
    def __init__(self, *, config: ServiceConfig, data_plane: Optional[DataPlane]):
        self.cfg = config
        self.ps = config.page_size
        self.index = PrefixIndex(page_size=self.ps)
        self.device = PageAllocator(num_pages=config.num_device_pages, first_page=1)
        self.dp = data_plane
        self.leases: dict[int, Lease] = {}
        self._next_lease = 1
        self._latest_done: Any = None
        self._events: list = []
        self.new_token_ratio = config.init_new_token_ratio
        self.counters = dict(
            device_hit_tokens=0,
            loaded_tokens=0,
            refused=0,
            revoked=0,
            evicted_device_pages=0,
            evicted_host_pages=0,
            backed_up_pages=0,
        )

    # ===== Helpers =====

    def _pages_for(self, tokens: int) -> int:
        return -(-tokens // self.ps)

    def _slots(self, pages: Sequence[int], start: int, end: int) -> torch.Tensor:
        """Device slot ids of positions [start, end) given one page per page."""
        if end <= start:
            return torch.empty((0,), dtype=torch.int64)
        first, last = start // self.ps, (end - 1) // self.ps
        page_t = torch.tensor(pages[first : last + 1], dtype=torch.int64)
        slots = (page_t[:, None] * self.ps + torch.arange(self.ps)).flatten()
        off = first * self.ps
        return slots[start - off : end - off]

    def _reserve_pages(self, lease: Lease) -> int:
        decoded = max(0, lease.total_len - lease.attrs.prompt_len)
        remaining = max(0, lease.attrs.max_new_tokens - decoded)
        return self._pages_for(math.ceil(remaining * self.new_token_ratio))

    def _reserved(self, exclude: Optional[Lease] = None) -> int:
        return sum(self._reserve_pages(l) for l in self.leases.values() if l is not exclude)

    def _headroom(self, reserved: int) -> int:
        return self.device.num_free + self.index.evictable_device_pages - reserved

    def _take(self, num_pages: int) -> Optional[list[int]]:
        """Allocate device pages, evicting unborrowed cache if needed."""
        short = num_pages - self.device.num_free
        if short > 0:
            freed = self.index.evict_device(short)
            self.counters["evicted_device_pages"] += len(freed)
            self.device.free(freed)
        return self.device.alloc(num_pages)

    def _take_host(self, num_pages: int) -> Optional[list[int]]:
        if num_pages > self.dp.host_num_pages:
            return None
        short = num_pages - self.dp.host_num_pages_free()
        if short > 0:
            freed = self.index.evict_host(short)
            self.counters["evicted_host_pages"] += len(freed)
            self.dp.free_host(freed)
        return self.dp.alloc_host(num_pages)

    # ===== Verbs =====

    def query(self, keys: Sequence[CacheKey]) -> list[QueryResult]:
        out = []
        for key in keys:
            ready, cached = self.index.peek(key.namespace, key.token_ids, touch=True)
            out.append(QueryResult(ready_len=ready, cached_len=cached))
        return out

    def acquire(self, reqs: Sequence[AcquireReq]) -> AcquireBatch:
        results: list[Optional[AcquireResult]] = [None] * len(reqs)
        must_grow = []
        for i, r in enumerate(reqs):
            if r.lease is None:
                results[i] = self._open(r)
            elif self.leases[r.lease].decode_phase():
                must_grow.append(i)
            else:
                results[i] = self._grow_flexible(r)
        if must_grow:
            self._grow_all(reqs, must_grow, results)
        events, self._events = self._events, []
        return AcquireBatch(results=results, events=events)

    def release_mut(self, lease: int, token_ids: Sequence[int]) -> None:
        self._publish(self.leases[lease], token_ids)

    def release(
        self,
        lease: int,
        *,
        from_len: int = 0,
        publish_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        l = self.leases[lease]
        if publish_token_ids is not None:
            self._publish(l, publish_token_ids)
        if from_len == 0:
            self._end(l)
            return
        assert l.shared_len <= from_len <= l.total_len, (from_len, l.shared_len, l.total_len)
        keep = self._pages_for(from_len)
        self.device.free([p for p, own in zip(l.pages[keep:], l.private[keep:]) if own])
        del l.pages[keep:], l.private[keep:]
        l.total_len = from_len

    def poll(self) -> list:
        events, self._events = self._events, []
        return events

    def begin_batch(self, batch_id: int, leases: Sequence[int]) -> Optional[Fence]:
        if self.dp is None:
            return None
        return self.dp.launch_loads(after=self._latest_done)

    def end_batch(self, batch_id: int, done_event: Any) -> list:
        self._latest_done = done_event
        return []

    # ===== Open / grow =====

    def _open(self, r: AcquireReq) -> AcquireResult:
        ns = r.key.namespace
        path = self.index.match(ns, r.key.token_ids)
        n_dev = sum(1 for n in path.nodes if n.device_pages is not None)
        dev_nodes, host_nodes = path.nodes[:n_dev], path.nodes[n_dev:]
        anchor = dev_nodes[-1] if dev_nodes else self.index.root
        self.index.lock(anchor)
        if self.dp is None:
            host_nodes = []
        plan = self._plan_open(r, host_nodes, path.device_len)
        if plan is None:
            self.index.unlock(anchor)
            self.counters["refused"] += 1
            return AcquireResult(status=AcquireStatus.REFUSED)
        loaded_nodes, end = plan
        loaded = self._load(loaded_nodes)
        if loaded_nodes:
            self.index.lock(loaded_nodes[-1])
            self.index.unlock(anchor)
            anchor = loaded_nodes[-1]
        shared_len = path.device_len + loaded
        mut_pages = self._take(self._pages_for(end) - shared_len // self.ps)
        assert mut_pages is not None, "headroom check admitted an unallocatable lease"
        lease = Lease(id=self._next_lease, namespace=ns, node=anchor, attrs=r.attrs)
        self._next_lease += 1
        for n in dev_nodes + loaded_nodes:
            lease.pages.extend(n.device_pages)
        lease.private = [False] * len(lease.pages) + [True] * len(mut_pages)
        lease.pages.extend(mut_pages)
        lease.shared_len = shared_len
        lease.total_len = end
        self.leases[lease.id] = lease
        self.counters["device_hit_tokens"] += path.device_len
        self.counters["loaded_tokens"] += loaded
        return AcquireResult(
            status=AcquireStatus.GRANTED,
            lease=lease.id,
            shared_len=shared_len,
            total_len=end,
            shared_slots=self._slots(lease.pages, 0, shared_len),
            mut_slots=self._slots(lease.pages, shared_len, end),
            loaded_len=loaded,
        )

    def _grant_end(self, r: AcquireReq, start: int, room_pages: int) -> Optional[int]:
        """End of the span granted from position ``start`` with ``room_pages``
        new pages (beyond the pages covering ``start``), or None to refuse."""
        if r.end <= start:
            return start
        align = self.ps if r.cut_align is None else math.lcm(self.ps, r.cut_align)
        cap_end = (self._pages_for(start) + max(0, room_pages)) * self.ps
        limit = r.end if r.max_mut_len is None else min(r.end, start + r.max_mut_len)
        if limit < r.end:
            limit = limit // align * align
        end = limit if limit <= cap_end else cap_end // align * align
        floor = limit if r.min_mut_len is None else min(start + r.min_mut_len, limit)
        if end <= start or end < floor:
            return None
        return end

    def _plan_open(
        self, r: AcquireReq, host_nodes: list[Node], device_len: int
    ) -> Optional[tuple[list[Node], int]]:
        """Pick (host nodes to load, granted end): load if it fits, else recompute."""
        own_reserve = self._pages_for(
            math.ceil(r.attrs.max_new_tokens * self.new_token_ratio)
        )
        room = self._headroom(self._reserved()) - own_reserve
        options = [([], device_len)]
        if host_nodes:
            host_len = sum(len(n.tokens) for n in host_nodes)
            options.insert(0, (host_nodes, device_len + host_len))
        for loaded_nodes, shared in options:
            load_pages = (shared - device_len) // self.ps
            end = self._grant_end(r, shared, room - load_pages)
            if load_pages <= room and end is not None:
                return loaded_nodes, end
        return None

    def _load(self, nodes: list[Node]) -> int:
        """Commit device pages to host-only nodes and queue their copies."""
        if not nodes:
            return 0
        need = sum(len(n.host_pages) for n in nodes)
        pages = self._take(need)
        assert pages is not None, "headroom check admitted an unloadable prefix"
        host_pages = [p for n in nodes for p in n.host_pages]
        op = self.dp.queue_load(host_pages, pages)
        start = 0
        for n in nodes:
            k = len(n.host_pages)
            self.index.commit_load(n, pages[start : start + k], op)
            start += k
        return need * self.ps

    def _grow_pages(self, lease: Lease, end: int) -> int:
        return max(0, self._pages_for(end) - len(lease.pages))

    def _extend(self, lease: Lease, end: int, pages: list[int]) -> AcquireResult:
        start = lease.total_len
        # Growth to an end the lease already covers grants nothing new.
        end = max(end, start)
        lease.pages.extend(pages)
        lease.private.extend([True] * len(pages))
        lease.total_len = end
        return AcquireResult(
            status=AcquireStatus.GRANTED,
            lease=lease.id,
            shared_len=lease.shared_len,
            total_len=end,
            mut_slots=self._slots(lease.pages, start, end),
        )

    def _grow_flexible(self, r: AcquireReq) -> AcquireResult:
        """Prompt-phase growth (next chunk): grant what fits, or refuse."""
        lease = self.leases[r.lease]
        room = self._headroom(self._reserved())
        end = self._grant_end(r, lease.total_len, room)
        if end is None:
            self.counters["refused"] += 1
            return AcquireResult(status=AcquireStatus.REFUSED, lease=lease.id)
        pages = self._take(self._grow_pages(lease, end))
        assert pages is not None
        return self._extend(lease, end, pages)

    def _grow_all(
        self, reqs: Sequence[AcquireReq], idx: list[int], results: list
    ) -> None:
        """Decode-phase growth must succeed: revoke leases until it fits."""
        live = list(idx)
        need = {i: self._grow_pages(self.leases[reqs[i].lease], reqs[i].end) for i in live}
        pages = None
        while live:
            total = sum(need[i] for i in live)
            if total <= self.device.num_free + self.index.evictable_device_pages:
                pages = self._take(total)
                if pages is not None:
                    break
            victim = min(live, key=lambda i: self._victim_key(self.leases[reqs[i].lease]))
            live.remove(victim)
            results[victim] = self._revoke(self.leases[reqs[victim].lease])
        if len(live) < len(idx):
            self.new_token_ratio = self.cfg.init_new_token_ratio
        else:
            self.new_token_ratio = max(
                self.cfg.min_new_token_ratio,
                self.new_token_ratio - self.cfg.new_token_ratio_decay,
            )
        start = 0
        for i in live:
            k = need[i]
            results[i] = self._extend(
                self.leases[reqs[i].lease], reqs[i].end, pages[start : start + k]
            )
            start += k

    @staticmethod
    def _victim_key(lease: Lease) -> tuple:
        # Fewest decoded tokens first, then the longest prompt.
        decoded = lease.total_len - lease.attrs.prompt_len
        return (decoded, -lease.attrs.prompt_len, -lease.id)

    def _revoke(self, lease: Lease) -> AcquireResult:
        self._end(lease)
        self.counters["revoked"] += 1
        self._events.append(Revoked(lease=lease.id))
        return AcquireResult(status=AcquireStatus.REVOKED, lease=lease.id)

    # ===== Publish / end =====

    def _publish(self, lease: Lease, token_ids: Sequence[int]) -> None:
        upto = len(token_ids)
        assert upto <= lease.total_len, (upto, lease.total_len)
        pub = upto // self.ps * self.ps
        if pub <= lease.shared_len:
            return
        out = self.index.insert(lease.namespace, token_ids[:pub], lease.pages[: pub // self.ps])
        for i, adopted in enumerate(out.adopted):
            if adopted:
                assert lease.private[i]
                lease.private[i] = False
        self.index.lock(out.last_node)
        self.index.unlock(lease.node)
        lease.node = out.last_node
        lease.shared_len = pub
        if self.dp is not None and out.new_resident:
            self._write_through(out.last_node)

    def _write_through(self, node: Node) -> None:
        nodes = self.index.unbacked_on_path(node)
        dev, host, backed = [], [], []
        for n in nodes:
            hp = self._take_host(len(n.device_pages))
            if hp is None:
                # Keep host residency prefix-closed: stop at the first miss.
                break
            dev.extend(n.device_pages)
            host.extend(hp)
            backed.append((n, hp))
        if not backed:
            return
        op = self.dp.backup(dev, host, after=self._latest_done)
        for n, hp in backed:
            self.index.set_backup(n, hp, op)
        self.counters["backed_up_pages"] += len(dev)

    def _end(self, lease: Lease) -> None:
        self.index.unlock(lease.node)
        self.device.free([p for p, own in zip(lease.pages, lease.private) if own])
        del self.leases[lease.id]

    # ===== Control plane =====

    def layer_hook(self) -> Any:
        return None if self.dp is None else self.dp.layer_hook()

    def stats(self) -> CacheStats:
        ps = self.ps
        return CacheStats(
            device_slots=self.device.num_pages * ps,
            device_free_slots=self.device.num_free * ps,
            device_evictable_slots=self.index.evictable_device_pages * ps,
            host_slots=0 if self.dp is None else self.dp.host_num_pages * ps,
            host_free_slots=0 if self.dp is None else self.dp.host_num_pages_free() * ps,
            open_leases=len(self.leases),
            counters=dict(self.counters),
        )

    def flush(self) -> bool:
        if self.leases:
            return False
        device, host = self.index.drop_all()
        self.device.free(device)
        if self.dp is not None:
            self.dp.free_host(host)
        return True

    def close(self) -> None:
        if self.dp is not None:
            self.dp.close()
