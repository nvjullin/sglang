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
    StateAnchor,
    StateGrant,
)
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.radix_index import Node, PrefixIndex
from sglang.srt.mem_cache.borrow.hicache.transfer import DataPlane, StatePlane
from sglang.srt.mem_cache.borrow.hicache.windows import WindowConfig, WindowPages

logger = logging.getLogger(__name__)


class StateConfig(msgspec.Struct, frozen=True):
    """Recurrent-state caching (Mamba / GDN / KDA layers)."""

    # Device state slots [first_slot, first_slot + num_slots).
    num_slots: int
    first_slot: int
    # Tracking slots granted per lease next to its working slot.
    tracking_slots: int
    # Branch points land on multiples of this (a multiple of the page size).
    checkpoint_grid: int
    # A running lease's state at the end of its prompt is cached too (a PD
    # prefill worker hands the request off there).
    publish_at_prompt_end: bool = False


class ServiceConfig(msgspec.Struct, frozen=True):
    page_size: int
    # Device slots [page_size, (num_device_pages + 1) * page_size); page 0 holds
    # the padding slot.
    num_device_pages: int
    init_new_token_ratio: float = 0.7
    min_new_token_ratio: float = 0.1
    new_token_ratio_decay: float = 0.001
    state: Optional[StateConfig] = None
    window: Optional[WindowConfig] = None


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
        "state_slots",
        "branch_len",
        "adopted_len",
        "wpages",
        "wprivate",
        "window_floor",
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
        # Working slot, then tracking slots; all freed when the lease ends.
        self.state_slots: list[int] = []
        self.branch_len: Optional[int] = None
        # States at lengths <= adopted_len are cached or were offered already.
        self.adopted_len = 0
        # Window page per page of positions (None: no window borrow);
        # wprivate[i] means the lease frees it, else it pins a cached page.
        self.wpages: list[Optional[int]] = []
        self.wprivate: list[bool] = []
        # Pages below this index have no window borrow left.
        self.window_floor = 0

    def decode_phase(self) -> bool:
        return self.total_len >= self.attrs.prompt_len


class HiCacheService(KVCacheService):
    def __init__(
        self,
        *,
        config: ServiceConfig,
        data_plane: Optional[DataPlane],
        state_plane: Optional[StatePlane] = None,
    ):
        assert (config.state is None) == (state_plane is None)
        assert config.state is None or config.window is None, (
            "recurrent state with a sliding window is not supported"
        )
        if config.state is not None:
            assert config.state.checkpoint_grid % config.page_size == 0
        self.cfg = config
        self.ps = config.page_size
        self.index = PrefixIndex(page_size=self.ps)
        self.device = PageAllocator(num_pages=config.num_device_pages, first_page=1)
        self.dp = data_plane
        self.sp = state_plane
        self.win = (
            None
            if config.window is None
            else WindowPages(config=config.window, index=self.index, page_size=self.ps)
        )
        self.states = (
            None
            if config.state is None
            else PageAllocator(
                num_pages=config.state.num_slots, first_page=config.state.first_slot
            )
        )
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
            state_device_hits=0,
            state_host_loads=0,
            states_published=0,
            states_backed_up=0,
            windows_recovered=0,
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

    def _window_room(self, own_reserve: int = 0) -> int:
        """Window pages grantable now, beyond what decoding leases may need."""
        if self.win is None:
            return 1 << 62
        reserved = sum(self.win.reserve(l) for l in self.leases.values())
        return self.win.headroom(reserved) - own_reserve

    def _take(self, num_pages: int) -> Optional[list[int]]:
        """Allocate device pages, evicting unborrowed cache if needed."""
        short = num_pages - self.device.num_free
        if short > 0:
            ev = self.index.evict_device(short)
            self.counters["evicted_device_pages"] += len(ev.pages)
            self.device.free(ev.pages)
            if ev.states:
                self.states.free(ev.states)
            if ev.windows:
                self.win.alloc.free(ev.windows)
        return self.device.alloc(num_pages)

    def _take_pair(self, num_pages: int) -> tuple[list[int], list[int]]:
        """KV pages and (with a window pool) as many window pages."""
        pages = self._take(num_pages)
        assert pages is not None, "headroom check admitted an unallocatable span"
        if self.win is None:
            return pages, []
        windows = self.win.take(num_pages)
        assert windows is not None, "window headroom check admitted an unallocatable span"
        return pages, windows

    def _take_host(self, num_pages: int) -> Optional[list[int]]:
        if num_pages > self.dp.host_num_pages:
            return None
        short = num_pages - self.dp.host_num_pages_free()
        if short > 0:
            ev = self.index.evict_host(short)
            self.counters["evicted_host_pages"] += len(ev.pages)
            self.dp.free_host(ev.pages)
            if ev.states:
                self.sp.free_host(ev.states)
            if ev.windows:
                self.dp.free_host_windows(ev.windows)
        return self.dp.alloc_host(num_pages)

    def _take_host_windows(self, num: int) -> Optional[list[int]]:
        if num > self.dp.host_num_window_pages:
            return None
        short = num - self.dp.host_window_pages_free()
        if short > 0:
            self.dp.free_host_windows(self.index.evict_host_windows(short))
        return self.dp.alloc_host_windows(num)

    def _take_states(self, num: int, *, keep: Optional[Node]) -> Optional[list[int]]:
        """Allocate device state slots, evicting cached states (not ``keep``'s)."""
        short = num - self.states.num_free
        if short > 0:
            self.states.free(self.index.evict_states(short, keep=keep))
        return self.states.alloc(num)

    def _take_host_state(self) -> Optional[int]:
        if self.sp.host_num_slots == 0:
            return None
        if self.sp.host_num_free() == 0:
            self.sp.free_host(self.index.evict_host_states(1))
        got = self.sp.alloc_host(1)
        return None if got is None else got[0]

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

    def release_mut(
        self,
        lease: int,
        token_ids: Sequence[int],
        *,
        state: Optional[StateAnchor] = None,
    ) -> None:
        self._publish(self.leases[lease], token_ids, state=state, running=True)

    def release(
        self,
        lease: int,
        *,
        from_len: Optional[int] = 0,
        publish_token_ids: Optional[Sequence[int]] = None,
        state: Optional[StateAnchor] = None,
        window_below: Optional[int] = None,
    ) -> None:
        l = self.leases[lease]
        if publish_token_ids is not None:
            self._publish(l, publish_token_ids, state=state, running=from_len != 0)
        if window_below is not None and self.win is not None:
            self.win.release_below(l, window_below)
        if from_len is None:
            return
        if from_len == 0:
            self._end(l)
            return
        assert l.shared_len <= from_len <= l.total_len, (from_len, l.shared_len, l.total_len)
        keep = self._pages_for(from_len)
        if self.win is not None:
            self.win.truncate(l, keep)
        self.device.free([p for p, own in zip(l.pages[keep:], l.private[keep:]) if own])
        del l.pages[keep:], l.private[keep:]
        l.total_len = from_len

    def poll(self) -> list:
        events, self._events = self._events, []
        return events

    def begin_batch(self, batch_id: int, leases: Sequence[int]) -> Optional[Fence]:
        fence = None if self.dp is None else self.dp.launch_loads(after=self._latest_done)
        if self.sp is None:
            return fence
        event = self.sp.fence_event(also_after=None if fence is None else fence.final_event)
        if event is None:
            return fence
        index = -1 if fence is None else fence.consumer_index
        return Fence(consumer_index=index, final_event=event)

    def end_batch(self, batch_id: int, done_event: Any) -> list:
        self._latest_done = done_event
        return []

    # ===== Open / grow =====

    def _open(self, r: AcquireReq) -> AcquireResult:
        ns = r.key.namespace
        path = self.index.match(ns, r.key.token_ids)
        nodes = path.nodes
        n_dev = sum(1 for n in nodes if n.device_pages is not None)
        guard = nodes[n_dev - 1] if n_dev else self.index.root
        self.index.lock(guard)
        plan = None
        if self._state_slots_fit():
            plan = self._plan_open(r, nodes, n_dev, self._open_options(nodes, n_dev))
        if plan is None:
            self.index.unlock(guard)
            self.counters["refused"] += 1
            return AcquireResult(status=AcquireStatus.REFUSED)
        k, end = plan
        shared_len = sum(len(n.tokens) for n in nodes[:k])
        if self.win is not None:
            # Before any allocation, which could tombstone them.
            self.win.pin_window_of(nodes[: min(k, n_dev)], shared_len, 0)
        loaded = self._load(nodes[n_dev:k])
        if self.win is not None and k > n_dev:
            start_page = sum(len(n.tokens) for n in nodes[:n_dev]) // self.ps
            self.win.pin_window_of(nodes[n_dev:k], shared_len, start_page)
        anchor = nodes[k - 1] if k else self.index.root
        self.index.lock(anchor)
        self.index.unlock(guard)
        mut_pages, mut_windows = self._take_pair(
            self._pages_for(end) - shared_len // self.ps
        )
        lease = Lease(id=self._next_lease, namespace=ns, node=anchor, attrs=r.attrs)
        self._next_lease += 1
        for n in nodes[:k]:
            lease.pages.extend(n.device_pages)
        lease.private = [False] * len(lease.pages) + [True] * len(mut_pages)
        lease.pages.extend(mut_pages)
        lease.shared_len = lease.adopted_len = shared_len
        lease.total_len = end
        if self.win is not None:
            self.win.set_shared(lease, nodes[:k], shared_len)
            self.win.add_private(lease, mut_windows)
        state = None
        if self.states is not None:
            state = self._grant_state(lease, path.device_len + path.host_len)
        self.leases[lease.id] = lease
        self.counters["device_hit_tokens"] += min(path.device_len, shared_len)
        self.counters["loaded_tokens"] += loaded
        return AcquireResult(
            status=AcquireStatus.GRANTED,
            lease=lease.id,
            shared_len=shared_len,
            total_len=end,
            shared_slots=self._slots(lease.pages, 0, shared_len),
            mut_slots=self._slots(lease.pages, shared_len, end),
            loaded_len=loaded,
            state=state,
            **self._window_slots(lease, 0, shared_len, end),
        )

    def _window_slots(self, lease: Lease, start: int, mid: int, end: int) -> dict:
        if self.win is None:
            return {}
        return dict(
            shared_window_slots=self.win.slots(lease.wpages, start, mid) if mid > start else None,
            mut_window_slots=self.win.slots(lease.wpages, mid, end),
        )

    def _open_options(self, nodes: list[Node], n_dev: int) -> list[int]:
        """Candidate shared prefixes, best first, as counts of matched nodes:
        load the host-only part, else share the device-resident part. With
        recurrent state a shared prefix must end where a state is cached."""
        top = len(nodes) if self.dp is not None else n_dev
        if self.states is not None:
            with_state = [i + 1 for i in range(top) if self._has_state(nodes[i])]
            top = with_state[-1] if with_state else 0
            on_device = [i for i in with_state if i <= n_dev]
            n_dev_ok = on_device[-1] if on_device else 0
        elif self.win is not None:
            valid = [k for k in range(top, 0, -1) if self.win.shared_ok(nodes, k, n_dev)]
            top = valid[0] if valid else 0
            on_device = [k for k in valid if k <= n_dev]
            n_dev_ok = on_device[0] if on_device else 0
        else:
            n_dev_ok = n_dev
        return [top, n_dev_ok] if top > n_dev_ok else [n_dev_ok]

    @staticmethod
    def _has_state(node: Node) -> bool:
        return node.state is not None or node.host_state is not None

    def _state_slots_fit(self) -> bool:
        if self.states is None:
            return True
        need = 1 + self.cfg.state.tracking_slots
        # The anchor's own state is never evicted for its lease.
        return self.states.num_free + self.index.num_states - 1 >= need

    def _grant_state(self, lease: Lease, full_hit: int) -> StateGrant:
        """Take the lease's state slots and start its working slot at the
        state after ``shared_len`` positions."""
        cfg = self.cfg.state
        anchor = lease.node
        slots = self._take_states(1 + cfg.tracking_slots, keep=anchor)
        assert slots is not None, "state headroom check admitted an unfit lease"
        working, after = slots[0], self._latest_done
        if anchor.state is not None:
            self.sp.copy(anchor.state, working, after=after)
            self.counters["state_device_hits"] += 1
        elif anchor.host_state is not None:
            self.sp.load(anchor.host_state, working, after=after)
            self.counters["state_host_loads"] += 1
        else:
            assert anchor is self.index.root, f"shared prefix ends without a state: {anchor}"
            self.sp.zero(working, after=after)
        lease.state_slots = slots
        aligned = full_hit // cfg.checkpoint_grid * cfg.checkpoint_grid
        lease.branch_len = aligned if aligned > lease.shared_len else None
        return StateGrant(slots=tuple(slots), branch_len=lease.branch_len)

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
        self, r: AcquireReq, nodes: list[Node], n_dev: int, options: list[int]
    ) -> Optional[tuple[int, int]]:
        """Pick (shared node count, granted end): the first option that fits.
        Nodes past ``n_dev`` are host-only and must be loaded."""
        own_reserve = self._pages_for(
            math.ceil(r.attrs.max_new_tokens * self.new_token_ratio)
        )
        room = self._headroom(self._reserved()) - own_reserve
        window_room = self._window_room(
            own_reserve=0 if self.win is None else self.win.lease_cap()
        )
        for k in options:
            load_pages = sum(len(n.host_pages) for n in nodes[n_dev:k])
            load_windows = sum(
                len(self.index.live_windows(n.host_window_pages)) for n in nodes[n_dev:k]
            )
            shared = sum(len(n.tokens) for n in nodes[:k])
            fit = min(room - load_pages, window_room - load_windows)
            end = self._grant_end(r, shared, fit)
            if load_pages <= room and load_windows <= window_room and end is not None:
                return k, end
        return None

    def _load(self, nodes: list[Node]) -> int:
        """Commit device pages to host-only nodes and queue their copies."""
        if not nodes:
            return 0
        need = sum(len(n.host_pages) for n in nodes)
        pages = self._take(need)
        assert pages is not None, "headroom check admitted an unloadable prefix"
        host_pages = [p for n in nodes for p in n.host_pages]
        windows, node_windows = None, [None] * len(nodes)
        if self.win is not None:
            host_w = [w for n in nodes for w in self.index.live_windows(n.host_window_pages)]
            dev_w = self.win.take(len(host_w))
            assert dev_w is not None, "window headroom check admitted an unloadable prefix"
            windows, it = (host_w, dev_w), iter(dev_w)
            node_windows = [
                None
                if n.host_window_pages is None
                else [None if w is None else next(it) for w in n.host_window_pages]
                for n in nodes
            ]
        op = self.dp.queue_load(host_pages, pages, windows=windows)
        start = 0
        for n, wp in zip(nodes, node_windows):
            k = len(n.host_pages)
            self.index.commit_load(n, pages[start : start + k], op, window_pages=wp)
            start += k
        return need * self.ps

    def _grow_pages(self, lease: Lease, end: int) -> int:
        return max(0, self._pages_for(end) - len(lease.pages))

    def _extend(
        self, lease: Lease, end: int, pages: list[int], windows: list[int]
    ) -> AcquireResult:
        start = lease.total_len
        # Growth to an end the lease already covers grants nothing new.
        end = max(end, start)
        lease.pages.extend(pages)
        lease.private.extend([True] * len(pages))
        lease.total_len = end
        if self.win is not None:
            self.win.add_private(lease, windows)
        return AcquireResult(
            status=AcquireStatus.GRANTED,
            lease=lease.id,
            shared_len=lease.shared_len,
            total_len=end,
            mut_slots=self._slots(lease.pages, start, end),
            mut_window_slots=None
            if self.win is None
            else self.win.slots(lease.wpages, start, end),
        )

    def _grow_flexible(self, r: AcquireReq) -> AcquireResult:
        """Prompt-phase growth (next chunk): grant what fits, or refuse."""
        lease = self.leases[r.lease]
        room = min(self._headroom(self._reserved()), self._window_room())
        end = self._grant_end(r, lease.total_len, room)
        if end is None:
            self.counters["refused"] += 1
            return AcquireResult(
                status=AcquireStatus.REFUSED,
                lease=lease.id,
                shared_len=lease.shared_len,
                total_len=lease.total_len,
            )
        pages, windows = self._take_pair(self._grow_pages(lease, end))
        return self._extend(lease, end, pages, windows)

    def _grow_all(
        self, reqs: Sequence[AcquireReq], idx: list[int], results: list
    ) -> None:
        """Decode-phase growth must succeed: revoke leases until it fits."""
        live = list(idx)
        need = {i: self._grow_pages(self.leases[reqs[i].lease], reqs[i].end) for i in live}
        pages = windows = None
        while live:
            total = sum(need[i] for i in live)
            if total <= self.device.num_free + self.index.evictable_device_pages and (
                self.win is None or total <= self.win.headroom(0)
            ):
                pages = self._take(total)
                if pages is not None:
                    windows = [] if self.win is None else self.win.take(total)
                    assert windows is not None
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
                self.leases[reqs[i].lease],
                reqs[i].end,
                pages[start : start + k],
                windows[start : start + k] if self.win is not None else [],
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

    def _publish(
        self,
        lease: Lease,
        token_ids: Sequence[int],
        *,
        state: Optional[StateAnchor],
        running: bool,
    ) -> None:
        self._publish_kv(lease, token_ids)
        if state is not None and self.states is not None:
            self._publish_state(lease, token_ids, state, running=running)

    def _publish_kv(self, lease: Lease, token_ids: Sequence[int]) -> None:
        upto = len(token_ids)
        assert upto <= lease.total_len, (upto, lease.total_len)
        pub = upto // self.ps * self.ps
        if pub <= lease.shared_len:
            return
        n = pub // self.ps
        out = self.index.insert(
            lease.namespace,
            token_ids[:pub],
            lease.pages[:n],
            window_pages=None if self.win is None else lease.wpages[:n],
        )
        for i, adopted in enumerate(out.adopted):
            if adopted:
                assert lease.private[i]
                lease.private[i] = False
        if self.win is not None:
            self.win.adopt(lease, out.adopted)
            if out.replaced:
                self.device.free(out.replaced)
                self.counters["windows_recovered"] += len(out.replaced)
        self.index.lock(out.last_node)
        self.index.unlock(lease.node)
        lease.node = out.last_node
        lease.shared_len = pub
        if self.dp is not None and out.new_resident:
            self._write_through(out.last_node)

    def _publish_state(
        self, lease: Lease, token_ids: Sequence[int], anchor: StateAnchor, *, running: bool
    ) -> None:
        """Cache a copy of the offered state when its position is a page
        boundary inside the published prefix. A running lease offers states as
        it goes; only its branch point (and, on a PD prefill worker, its prompt
        end) is kept, so a path does not fill up with states."""
        cfg = self.cfg.state
        length = anchor.length
        if length <= lease.adopted_len or length % self.ps or length > lease.shared_len:
            return
        if running and length != lease.branch_len:
            at_prompt_end = len(token_ids) >= lease.attrs.prompt_len
            if not (cfg.publish_at_prompt_end and at_prompt_end):
                return
        lease.adopted_len = length
        node = self.index.node_at(lease.namespace, token_ids[:length])
        if self._has_state(node):
            return
        got = self._take_states(1, keep=None)
        if got is None:
            return
        slot, after = got[0], self._latest_done
        self.sp.copy(lease.state_slots[anchor.slot_index], slot, after=after)
        self.index.set_state(node, slot)
        self.counters["states_published"] += 1
        if node.host_pages is None:
            return
        host = self._take_host_state()
        if host is not None:
            self.sp.backup(slot, host, after=after)
            self.index.set_host_state(node, host)
            self.counters["states_backed_up"] += 1

    def _write_through(self, node: Node) -> None:
        nodes = self.index.unbacked_on_path(node)
        dev, host, backed = [], [], []
        dev_w, host_w = [], []
        for n in nodes:
            hp = self._take_host(len(n.device_pages))
            if hp is None:
                # Keep host residency prefix-closed: stop at the first miss.
                break
            dev.extend(n.device_pages)
            host.extend(hp)
            backed.append((n, hp, self._backup_windows(n, dev_w, host_w)))
        if not backed:
            return
        op = self.dp.backup(
            dev, host, after=self._latest_done, windows=(dev_w, host_w) if dev_w else None
        )
        for n, hp, hw in backed:
            self.index.set_backup(n, hp, op, host_window_pages=hw)
        self.counters["backed_up_pages"] += len(dev)

    def _backup_windows(self, node: Node, dev_w: list, host_w: list) -> Optional[list]:
        """Host window pages for ``node``'s live window pages (appended to the
        copy lists); None entries where none was copied."""
        live = self.index.live_windows(node.window_pages)
        if not live:
            return None
        got = self._take_host_windows(len(live))
        if got is None:
            return None
        dev_w.extend(live)
        host_w.extend(got)
        it = iter(got)
        return [None if w is None else next(it) for w in node.window_pages]

    def _end(self, lease: Lease) -> None:
        self.index.unlock(lease.node)
        self.device.free([p for p, own in zip(lease.pages, lease.private) if own])
        if lease.state_slots:
            self.states.free(lease.state_slots)
        if self.win is not None:
            self.win.end(lease)
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
            state_slots=0 if self.states is None else self.states.num_pages,
            state_free_slots=0 if self.states is None else self.states.num_free,
            cached_states=self.index.num_states,
            window_pages=0 if self.win is None else self.win.alloc.num_pages,
            window_free_pages=0 if self.win is None else self.win.alloc.num_free,
            cached_window_pages=self.index.window_pages_cached,
            counters=dict(self.counters),
        )

    def flush(self) -> bool:
        if self.leases:
            return False
        device, host = self.index.drop_all()
        self.device.free(device.pages)
        if self.dp is not None:
            self.dp.free_host(host.pages)
        if self.states is not None:
            self.states.free(device.states)
            if host.states:
                self.sp.free_host(host.states)
        if self.win is not None:
            self.win.alloc.free(device.windows)
            if host.windows:
                self.dp.free_host_windows(host.windows)
        return True

    def close(self) -> None:
        if self.dp is not None:
            self.dp.close()
        if self.sp is not None:
            self.sp.close()
