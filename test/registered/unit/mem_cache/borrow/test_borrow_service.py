"""Contract tests of the borrow API on the HiCache service (CPU, fake data plane)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import random
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fake_runtime import (
    FakeDataPlane,
    FakeReq,
    FakeRuntime,
    FakeStatePlane,
    kv_content,
)

from sglang.srt.mem_cache.borrow.api import AcquireReq, CacheKey, LeaseAttrs
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.service import (
    HiCacheService,
    ServiceConfig,
    StateConfig,
)
from sglang.srt.mem_cache.borrow.hicache.windows import WindowConfig
from sglang.srt.mem_cache.borrow.threaded import CacheServiceError, ThreadedService


def make(page_size=4, device_pages=32, host_pages=64, with_host=True):
    kv = np.full((device_pages + 1) * page_size, -1, dtype=np.int64)
    dp = (
        FakeDataPlane(page_size=page_size, host_num_pages=host_pages, device_kv=kv)
        if with_host
        else None
    )
    svc = HiCacheService(
        config=ServiceConfig(page_size=page_size, num_device_pages=device_pages),
        data_plane=dp,
    )
    return svc, FakeRuntime(svc, kv)


def make_state(
    page_size=4,
    device_pages=32,
    host_pages=64,
    state_slots=12,
    host_states=16,
    grid=8,
    with_host=True,
    prompt_end=False,
):
    kv = np.full((device_pages + 1) * page_size, -1, dtype=np.int64)
    dp = (
        FakeDataPlane(page_size=page_size, host_num_pages=host_pages, device_kv=kv)
        if with_host
        else None
    )
    sp = FakeStatePlane(num_slots=state_slots + 1, host_num_slots=host_states if with_host else 0)
    cfg = ServiceConfig(
        page_size=page_size,
        num_device_pages=device_pages,
        state=StateConfig(
            num_slots=state_slots,
            first_slot=1,
            tracking_slots=2,
            checkpoint_grid=grid,
            publish_at_prompt_end=prompt_end,
        ),
    )
    svc = HiCacheService(config=cfg, data_plane=dp, state_plane=sp)
    return svc, FakeRuntime(svc, kv, state=sp, checkpoint_grid=grid)


def make_window(
    page_size=4,
    device_pages=32,
    window_pages=16,
    host_pages=64,
    host_windows=32,
    window=8,
    with_host=True,
):
    kv = np.full((device_pages + 1) * page_size, -1, dtype=np.int64)
    wkv = np.full((window_pages + 1) * page_size, -1, dtype=np.int64)
    dp = (
        FakeDataPlane(
            page_size=page_size,
            host_num_pages=host_pages,
            device_kv=kv,
            device_window_kv=wkv,
            host_num_window_pages=host_windows,
        )
        if with_host
        else None
    )
    cfg = ServiceConfig(
        page_size=page_size,
        num_device_pages=device_pages,
        window=WindowConfig(num_pages=window_pages, window_tokens=window, retire_interval=4),
    )
    svc = HiCacheService(config=cfg, data_plane=dp)
    rt = FakeRuntime(svc, kv, window_kv=wkv, window_tokens=window, page_size=page_size)
    return svc, rt


def check_window_accounting(svc: HiCacheService) -> None:
    """Every window page is free, cached by the index, or private to one lease;
    pins are exactly the leases' borrows of cached pages."""
    index = svc.index
    private, pinned = [], []
    for l in svc.leases.values():
        for w, own in zip(l.wpages, l.wprivate):
            if w is not None:
                (private if own else pinned).append(w)
    cached = [w for n in index._window_nodes for w in index.live_windows(n.window_pages)]
    assert len(private + cached) == len(set(private + cached)), "a window page has two owners"
    assert svc.win.alloc.num_free + len(private) + len(cached) == svc.cfg.window.num_pages
    assert set(pinned) <= set(cached), "a lease pins an uncached window page"
    counts = {}
    for w in pinned:
        counts[w] = counts.get(w, 0) + 1
    assert counts == index._window_pins


def check_state_accounting(svc: HiCacheService) -> None:
    """Every state slot is free, cached by one node, or held by one lease."""
    held = [s for l in svc.leases.values() for s in l.state_slots]
    cached = [n.state for n in svc.index._state_nodes]
    both = held + cached
    assert len(both) == len(set(both)), "a state slot has two owners"
    assert svc.states.num_free + len(both) == svc.cfg.state.num_slots


def check_accounting(svc: HiCacheService) -> None:
    """Every device page is free, cached in the index, or private to one lease."""
    private = [p for l in svc.leases.values() for p, own in zip(l.pages, l.private) if own]
    assert len(private) == len(set(private)), "a private page is owned twice"
    total = svc.device.num_free + svc.index.device_pages_cached + len(private)
    assert total == svc.cfg.num_device_pages, (total, svc.cfg.num_device_pages)


def check_exclusive_mut(svc: HiCacheService) -> None:
    """&mut positions never alias another lease's borrowed slots."""
    ps = svc.ps
    owner = {}
    for l in svc.leases.values():
        for i in range(l.shared_len // ps, len(l.pages)):
            if l.private[i]:
                assert l.pages[i] not in owner, f"page {l.pages[i]} in two &mut spans"
                owner[l.pages[i]] = l.id


class TestPageAllocator(unittest.TestCase):
    def test_rejects_double_free(self):
        a = PageAllocator(num_pages=4, first_page=1)
        pages = a.alloc(2)
        a.free(pages)
        with self.assertRaises(RuntimeError):
            a.free(pages[:1])
        with self.assertRaises(RuntimeError):
            a.free([0])


class TestSharing(unittest.TestCase):
    def test_second_request_shares_published_prefix_page_floored(self):
        svc, rt = make(page_size=4)
        a = FakeReq(1, list(range(1, 15)), max_new=0)
        self.assertTrue(rt.open(a))
        rt.forward([a])
        rt.finish(a)
        b = FakeReq(2, list(range(1, 11)) + [99, 98, 97], max_new=0)
        self.assertTrue(rt.open(b))
        # 10 common tokens, page floor -> 8 shared.
        self.assertEqual(rt.svc.leases[b.lease].shared_len, 8)
        rt.forward([b])
        check_accounting(svc)

    def test_namespaces_do_not_share(self):
        svc, rt = make()
        a = FakeReq(1, list(range(1, 20)), max_new=0, namespace=("lora-a",))
        rt.open(a)
        rt.forward([a])
        rt.finish(a)
        b = FakeReq(2, list(range(1, 20)), max_new=0, namespace=("lora-b",))
        rt.open(b)
        self.assertEqual(svc.leases[b.lease].shared_len, 0)

    def test_concurrent_publish_keeps_duplicates_private(self):
        """Two leases computing the same prefix: the second publish keeps its
        own pages (no remap) and frees them only at release."""
        svc, rt = make(page_size=2)
        a = FakeReq(1, list(range(1, 12)), max_new=0)
        b = FakeReq(2, list(range(1, 12)), max_new=0)
        rt.open(a)
        rt.open(b)
        rt.forward([a, b])
        rt.publish(a)
        rt.publish(b)
        lb = svc.leases[b.lease]
        self.assertTrue(all(lb.private[: 10 // 2]))
        check_accounting(svc)
        rt.finish(a)
        rt.finish(b)
        check_accounting(svc)
        self.assertEqual(svc.index.device_pages_cached, 5)


class TestTiers(unittest.TestCase):
    def test_evicted_prefix_loads_back_only_after_begin_batch(self):
        svc, rt = make(page_size=4, device_pages=8, host_pages=32)
        a = FakeReq(1, list(range(1, 18)), max_new=0)
        rt.open(a)
        rt.forward([a])
        rt.finish(a)
        # Fill the device with another prefix so a's pages are demoted to host.
        filler = FakeReq(2, list(range(100, 130)), max_new=0)
        self.assertTrue(rt.open(filler))
        rt.forward([filler])
        rt.finish(filler)
        self.assertEqual(svc.query([CacheKey(token_ids=a.prompt)])[0].ready_len, 0)
        b = FakeReq(3, list(a.prompt), max_new=0)
        self.assertTrue(rt.open(b))
        lease = svc.leases[b.lease]
        self.assertEqual(lease.shared_len, 16)
        stale = [rt.kv[s] != kv_content((), b.tokens, p) for p, s in enumerate(b.slots[:16])]
        self.assertTrue(any(stale), "loaded slots must not hold KV before the fence")
        fence = rt.forward([b])
        self.assertIsNotNone(fence)
        check_accounting(svc)

    def test_load_falls_back_to_recompute_when_it_does_not_fit(self):
        svc, rt = make(page_size=1, device_pages=6, host_pages=32)
        a = FakeReq(1, [1, 2, 3, 4, 5], max_new=0)
        rt.open(a)
        rt.forward([a])
        rt.finish(a)
        hold = FakeReq(2, [9, 9, 9, 9, 9, 9], max_new=0)
        rt.open(hold)  # evicts a to host; holds all 6 pages
        rt.forward([hold])
        b = FakeReq(3, [1, 2, 3, 4, 5], max_new=0)
        self.assertFalse(rt.open(b, min_mut_len=1))
        rt.finish(hold)
        self.assertTrue(rt.open(b))
        rt.forward([b])

    def test_device_only_service_works_without_host_tier(self):
        svc, rt = make(with_host=False, device_pages=8)
        a = FakeReq(1, list(range(1, 30)), max_new=0)
        rt.open(a)
        rt.forward([a])
        rt.finish(a)
        b = FakeReq(2, list(range(50, 70)), max_new=0)
        self.assertTrue(rt.open(b))
        self.assertGreater(svc.counters["evicted_device_pages"], 0)
        check_accounting(svc)


class TestGrowth(unittest.TestCase):
    def test_chunks_publish_between_and_cut_on_page_boundary(self):
        svc, rt = make(page_size=4, device_pages=64)
        a = FakeReq(1, list(range(1, 40)), max_new=0)
        self.assertTrue(rt.open(a, max_mut_len=10, min_mut_len=1))
        # A chunk cut by the compute budget also ends on a page boundary.
        self.assertEqual(len(a.slots), 8)
        rt.forward([a])
        rt.publish(a)
        self.assertEqual(svc.leases[a.lease].shared_len, 8)
        while len(a.slots) < len(a.prompt):
            self.assertTrue(rt.grow_chunk(a, max_mut_len=10))
            rt.forward([a])
            rt.publish(a)
        # Another request now shares everything a published.
        b = FakeReq(2, list(a.prompt), max_new=0)
        rt.open(b)
        self.assertEqual(svc.leases[b.lease].shared_len, 36)
        rt.forward([b])

    def test_chunk_cut_by_memory_lands_on_page_boundary(self):
        svc, rt = make(page_size=4, device_pages=5)
        a = FakeReq(1, list(range(1, 40)), max_new=0)
        self.assertTrue(rt.open(a, min_mut_len=1))
        self.assertEqual(len(a.slots), 20)

    def test_decode_shortfall_revokes_fewest_decoded_first(self):
        svc, rt = make(page_size=1, device_pages=24, with_host=False)
        reqs = [FakeReq(i, [i * 10 + j for j in range(6)], max_new=100) for i in range(3)]
        svc.new_token_ratio = 0.0
        for r in reqs:
            self.assertTrue(rt.open(r))
        rt.forward(reqs)
        alive = reqs
        alive = rt.decode(alive)  # each request: 7 tokens -> 21 slots
        rt.forward(alive)
        alive = [r for r in alive if r.rid != 0]
        alive = rt.decode(alive) + [reqs[0]]
        rt.forward(alive)
        # 3 * 8 = 24 slots; one more step cannot fit all three.
        alive = rt.decode(alive)
        self.assertEqual(len(rt.revoked), 1)
        self.assertEqual(rt.revoked[0].rid, 0)
        check_accounting(svc)
        check_exclusive_mut(svc)

    def test_growth_to_a_covered_end_keeps_the_lease(self):
        """Spec decoding reserves ahead: a step whose end the lease already
        covers grants nothing and must not shrink the lease."""
        svc, rt = make(page_size=4, device_pages=32, with_host=False)
        a = FakeReq(1, list(range(1, 8)), max_new=50)
        rt.open(a)
        rt.forward([a])
        lease = svc.leases[a.lease]
        res = svc.acquire([AcquireReq(end=16, lease=a.lease, attrs=rt.attrs(a))]).results[0]
        self.assertEqual((res.total_len, len(res.mut_slots)), (16, 9))
        res = svc.acquire([AcquireReq(end=12, lease=a.lease, attrs=rt.attrs(a))]).results[0]
        self.assertEqual((res.total_len, len(res.mut_slots)), (16, 0))
        self.assertEqual((lease.total_len, len(lease.pages)), (16, 4))

    def test_memory_cut_honors_cut_align(self):
        svc, rt = make(page_size=4, device_pages=7, with_host=False)
        key = CacheKey(token_ids=list(range(1, 40)))
        res = svc.acquire(
            [AcquireReq(end=40, key=key, min_mut_len=1, cut_align=8, attrs=LeaseAttrs(prompt_len=40))]
        ).results[0]
        # 7 pages hold 28 tokens; the cut lands on the 8-token grid.
        self.assertEqual(res.total_len, 24)

    def test_spec_rollback_frees_rejected_tail(self):
        svc, rt = make(page_size=2, device_pages=32, with_host=False)
        a = FakeReq(1, list(range(1, 8)), max_new=50)
        rt.open(a)
        rt.forward([a])
        rt.decode([a], k=5)  # 12 positions -> 6 pages
        self.assertEqual(len(svc.leases[a.lease].pages), 6)
        svc.release(a.lease, from_len=9)
        self.assertEqual(len(svc.leases[a.lease].pages), 5)
        check_accounting(svc)


class TestLockstep(unittest.TestCase):
    def _workload(self, svc_rt, seed):
        svc, rt = svc_rt
        rng = random.Random(seed)
        log = []
        running = []
        prefixes = [[rng.randrange(1, 9) for _ in range(rng.randrange(4, 20))] for _ in range(4)]
        for step in range(300):
            if rng.random() < 0.5 and len(running) < 6:
                prompt = rng.choice(prefixes) + [rng.randrange(1, 9) for _ in range(rng.randrange(1, 12))]
                r = FakeReq(step, prompt, max_new=rng.randrange(1, 12))
                ok = rt.open(r, min_mut_len=1, max_mut_len=rng.choice([None, 8]))
                log.append(("open", step, ok, r.slots[:]))
                if ok:
                    running.append(r)
            if running:
                rt.forward(running)
                for r in running:
                    if rng.random() < 0.3:
                        rt.publish(r)
                done = [r for r in running if len(r.slots) >= len(r.prompt) and len(r.out) >= r.max_new]
                for r in done:
                    rt.finish(r, publish=rng.random() < 0.8)
                    running.remove(r)
                prefill = [r for r in running if len(r.slots) < len(r.prompt)]
                for r in prefill:
                    rt.grow_chunk(r, max_mut_len=8)
                # A chunk granted just now runs its forward before any decode.
                decoding = [
                    r for r in running if r not in prefill and len(r.slots) >= len(r.prompt)
                ]
                if decoding:
                    alive = rt.decode(decoding)
                    running = [r for r in running if r in alive or r in prefill]
                log.append(("step", step, [(r.rid, tuple(r.slots)) for r in running]))
            check_accounting(svc)
            check_exclusive_mut(svc)
        return log, svc.stats()

    def test_replicas_fed_same_calls_agree(self):
        """Two service replicas make identical decisions from identical calls
        (no collectives), including evictions, loads and revocations."""
        log1, st1 = self._workload(make(page_size=2, device_pages=40, host_pages=50), seed=7)
        log2, st2 = self._workload(make(page_size=2, device_pages=40, host_pages=50), seed=7)
        self.assertEqual(log1, log2)
        self.assertEqual(st1, st2)
        self.assertGreater(st1.counters["loaded_tokens"], 0)
        self.assertGreater(st1.counters["evicted_host_pages"], 0)

    def test_random_workloads_keep_kv_correct(self):
        for seed in range(6):
            for ps in (1, 4):
                self._workload(make(page_size=ps, device_pages=60 // ps + 8, host_pages=80 // ps), seed=seed)


class TestRecurrentState(unittest.TestCase):
    """Leases of models with recurrent layers: the working slot must start at
    the state after shared_len, which therefore ends where a state is cached."""

    def _run(self, rt, req, *, chunk=None):
        self.assertTrue(rt.open(req, max_mut_len=chunk))
        rt.forward([req])
        while len(req.slots) < len(req.prompt):
            self.assertTrue(rt.grow_chunk(req, max_mut_len=chunk))
            rt.forward([req])

    def test_shared_prefix_is_cut_to_the_deepest_cached_state(self):
        """KV is cached for 36 tokens but the only state is at 24; a hit shares
        24 positions, and the branch point is the last grid multiple of the KV hit."""
        svc, rt = make_state(page_size=4, grid=8)
        prompt = list(range(1, 38))
        a = FakeReq(1, prompt, max_new=0)
        self._run(rt, a)
        # One forward over 37 tokens tracked the state at 32; offer 24 instead.
        rt.svc.release(a.lease, publish_token_ids=prompt[:36], state=None)
        b = FakeReq(2, prompt[:24] + [99] * 4, max_new=0)
        self._run(rt, b)
        rt.finish(b)
        c = FakeReq(3, prompt + [5], max_new=0)
        self.assertTrue(rt.open(c))
        self.assertEqual(c.computed, 24)
        self.assertEqual(c.branch_len, 32)
        rt.forward([c])
        self.assertEqual(svc.counters["state_device_hits"], 1)
        check_state_accounting(svc)

    def test_running_lease_keeps_only_its_branch_point(self):
        """A running publish caches a state only at the lease's branch point;
        a finished lease's state is cached."""
        svc, rt = make_state(page_size=4, grid=8)
        base = list(range(1, 30))
        a = FakeReq(1, base, max_new=0)
        self._run(rt, a)
        rt.svc.release(a.lease, publish_token_ids=base[:28], state=None)
        b = FakeReq(2, base[:28] + list(range(40, 60)), max_new=0)
        self.assertTrue(rt.open(b, max_mut_len=4))
        self.assertEqual((b.computed, b.branch_len), (0, 24))
        rt.forward([b])
        while len(b.slots) < len(b.prompt):
            rt.publish(b)
            rt.grow_chunk(b, max_mut_len=4)
            rt.forward([b])
        self.assertEqual(svc.index.num_states, 1)
        self.assertEqual(svc.counters["states_published"], 1)
        rt.finish(b)
        self.assertEqual(svc.index.num_states, 2)
        c = FakeReq(3, base[:24] + [7] * 8, max_new=0)
        self.assertTrue(rt.open(c))
        self.assertEqual(c.computed, 24)
        rt.forward([c])

    def test_prompt_end_publish_on_pd_prefill(self):
        svc, rt = make_state(page_size=4, grid=8, prompt_end=True)
        a = FakeReq(1, list(range(1, 34)), max_new=0)
        self._run(rt, a)
        rt.publish(a)
        self.assertEqual(svc.index.num_states, 1)

    def test_state_round_trips_through_the_host_tier(self):
        """Device eviction frees a node's KV pages and state slot; a later hit
        loads both from host and the working slot starts at the cached state."""
        svc, rt = make_state(page_size=4, device_pages=12, grid=8)
        p1 = list(range(1, 34))
        a = FakeReq(1, p1, max_new=0)
        self._run(rt, a)
        rt.finish(a)
        for i in range(3):
            other = FakeReq(10 + i, [50 + i] * 33, max_new=0)
            self._run(rt, other)
            rt.finish(other)
        c = FakeReq(2, p1 + [3], max_new=0)
        self.assertTrue(rt.open(c))
        self.assertEqual(c.computed, 32)
        self.assertGreater(svc.counters["loaded_tokens"], 0)
        self.assertEqual(svc.counters["state_host_loads"], 1)
        rt.forward([c])
        check_state_accounting(svc)

    def test_state_slot_shortage_refuses_and_evicts_cached_states(self):
        """Leases need 3 state slots each. Cached states are evicted for new
        leases, LRU first, but a hit never evicts the state it starts from; when
        leases hold every slot the next open is refused."""
        svc, rt = make_state(page_size=4, device_pages=64, state_slots=7, grid=8)
        for i in (1, 3):
            r = FakeReq(i, [i] * 17, max_new=0)
            self._run(rt, r)
            rt.finish(r)
        b = FakeReq(2, [2] * 17, max_new=0)
        self._run(rt, b)
        # Free: 7 - 3 (b) - 2 cached states = 2 < 3. The LRU state is the one
        # this hit starts from, so the other one goes.
        hit = FakeReq(5, [1] * 17 + [2], max_new=0)
        self.assertTrue(rt.open(hit))
        self.assertEqual(hit.computed, 16)
        self.assertIsNotNone(svc.leases[hit.lease].node.state)
        self.assertEqual(svc.index.num_states, 1)
        self.assertFalse(rt.open(FakeReq(9, [9] * 17, max_new=0)))
        rt.forward([hit])
        check_state_accounting(svc)

    def test_random_workloads_keep_state_correct(self):
        """Mixed chunked prefill, decode, publishes, evictions, loads and
        revocations: every forward finds its working state correct."""
        lock = TestLockstep()
        for seed in range(6):
            for ps in (1, 4):
                svc_rt = make_state(
                    page_size=ps,
                    device_pages=60 // ps + 8,
                    host_pages=80 // ps,
                    state_slots=20,
                    host_states=6,
                    grid=4 if ps == 1 else 8,
                )
                lock._workload(svc_rt, seed=seed)
                check_state_accounting(svc_rt[0])
        st = svc_rt[0].stats().counters
        self.assertGreater(st["state_host_loads"] + st["state_device_hits"], 0)


class TestSlidingWindow(unittest.TestCase):
    """Window KV lives in its own pages; a forward reads the window KV of the
    window_tokens positions before it, so a shared prefix must end where that
    trailing window is still cached."""

    def _run(self, rt, req):
        self.assertTrue(rt.open(req))
        rt.forward([req])

    def test_hit_needs_live_trailing_window(self):
        """Windows released before the publish become tombstones: a prefix
        ending inside them is not shared, one whose trailing window survives is."""
        svc, rt = make_window(page_size=4, window=8)
        a = FakeReq(1, list(range(1, 42)), max_new=0)
        self._run(rt, a)
        rt.retire_windows(a)
        rt.finish(a)
        b = FakeReq(2, a.prompt[:24] + [99] * 5, max_new=0)
        self._run(rt, b)
        self.assertEqual(b.computed, 29)
        self.assertEqual(svc.counters["device_hit_tokens"], 0)
        rt.finish(b, publish=False)
        c = FakeReq(3, a.prompt[:40] + [7], max_new=0)
        self.assertTrue(rt.open(c))
        self.assertEqual(c.computed, 40)
        rt.forward([c])
        check_window_accounting(svc)

    def test_released_windows_admit_more_leases(self):
        """Window pages, not KV pages, bound admission here; releasing them
        behind the window lets the next lease in."""
        svc, rt = make_window(page_size=4, device_pages=64, window_pages=12, window=4)
        a = FakeReq(1, list(range(1, 34)), max_new=0)
        self._run(rt, a)
        b = FakeReq(2, list(range(50, 70)), max_new=0)
        self.assertFalse(rt.open(b))
        rt.retire_windows(a)
        self.assertTrue(rt.open(b))
        rt.forward([b])
        check_window_accounting(svc)

    def test_window_round_trips_through_the_host_tier(self):
        svc, rt = make_window(page_size=4, device_pages=14, window_pages=24, window=8)
        a = FakeReq(1, list(range(1, 34)), max_new=0)
        self._run(rt, a)
        rt.finish(a)
        for i in range(3):
            other = FakeReq(10 + i, [60 + i] * 33, max_new=0)
            self._run(rt, other)
            rt.finish(other)
        c = FakeReq(2, a.prompt[:32] + [5], max_new=0)
        self.assertTrue(rt.open(c))
        self.assertEqual(c.computed, 32)
        self.assertGreater(svc.counters["loaded_tokens"], 0)
        rt.forward([c])
        check_window_accounting(svc)

    def _workload(self, svc_rt, seed):
        svc, rt = svc_rt
        rng = random.Random(seed)
        log, running = [], []
        prefixes = [[rng.randrange(1, 9) for _ in range(rng.randrange(4, 24))] for _ in range(4)]
        for step in range(300):
            if rng.random() < 0.5 and len(running) < 6:
                prompt = rng.choice(prefixes) + [rng.randrange(1, 9) for _ in range(rng.randrange(1, 12))]
                r = FakeReq(step, prompt, max_new=rng.randrange(1, 16))
                ok = rt.open(r, min_mut_len=1, max_mut_len=rng.choice([None, 8]))
                log.append(("open", step, ok, r.slots[:], r.wslots[:]))
                if ok:
                    running.append(r)
            if running:
                rt.forward(running)
                for r in running:
                    if rng.random() < 0.3:
                        rt.publish(r)
                    if rng.random() < 0.5:
                        rt.retire_windows(r)
                done = [r for r in running if len(r.slots) >= len(r.prompt) and len(r.out) >= r.max_new]
                for r in done:
                    rt.finish(r, publish=rng.random() < 0.8)
                    running.remove(r)
                prefill = [r for r in running if len(r.slots) < len(r.prompt)]
                for r in prefill:
                    rt.grow_chunk(r, max_mut_len=8)
                # A chunk granted just now runs its forward before any decode.
                decoding = [
                    r for r in running if r not in prefill and len(r.slots) >= len(r.prompt)
                ]
                if decoding:
                    alive = rt.decode(decoding)
                    running = [r for r in running if r in alive or r in prefill]
                log.append(("step", step, [(r.rid, tuple(r.slots), tuple(r.wslots)) for r in running]))
            check_accounting(svc)
            check_exclusive_mut(svc)
            check_window_accounting(svc)
        return log, svc.stats()

    def test_random_workloads_keep_window_kv_correct(self):
        """Chunked prefill, decode, retirement, publishes, tombstoning,
        evictions, loads and revocations: every forward reads correct window KV."""
        for seed in range(6):
            for ps in (1, 4):
                self._workload(
                    make_window(
                        page_size=ps,
                        device_pages=60 // ps + 8,
                        window_pages=40 // ps + 6,
                        host_pages=80 // ps,
                        host_windows=40 // ps,
                        window=4,
                    ),
                    seed=seed,
                )

    def test_replicas_fed_same_calls_agree(self):
        log1, st1 = self._workload(make_window(page_size=2, device_pages=40, window_pages=26), seed=3)
        log2, st2 = self._workload(make_window(page_size=2, device_pages=40, window_pages=26), seed=3)
        self.assertEqual(log1, log2)
        self.assertEqual(st1, st2)
        self.assertGreater(st1.counters["loaded_tokens"], 0)


class TestThreaded(unittest.TestCase):
    def test_runs_on_its_own_thread_in_caller_order(self):
        seen = []

        class Probe(HiCacheService):
            def release_mut(self, lease, token_ids, *, state=None):
                seen.append(("release_mut", threading.current_thread().name))
                super().release_mut(lease, token_ids, state=state)

        kv = np.full(33 * 4, -1, dtype=np.int64)
        svc = ThreadedService(
            lambda: Probe(config=ServiceConfig(page_size=4, num_device_pages=32), data_plane=None),
            name="cache-test",
        )
        rt = FakeRuntime(svc, kv)
        a = FakeReq(1, list(range(1, 14)), max_new=0)
        rt.open(a)
        rt.forward([a])
        rt.publish(a)
        # The publish is fire-and-forget; the next open must still observe it.
        b = FakeReq(2, list(range(1, 14)), max_new=0)
        rt.open(b)
        self.assertEqual(seen, [("release_mut", "cache-test")])
        self.assertEqual(rt.svc.query([CacheKey(token_ids=a.prompt)])[0].ready_len, 12)
        svc.close()

    def test_async_failure_surfaces_on_next_call(self):
        svc = ThreadedService(
            lambda: HiCacheService(
                config=ServiceConfig(page_size=1, num_device_pages=8), data_plane=None
            )
        )
        svc.release(12345)  # unknown lease, fails on the service thread
        with self.assertRaises(CacheServiceError):
            svc.poll()
        with self.assertRaises(CacheServiceError):
            svc.release(1)
        svc.close()


if __name__ == "__main__":
    unittest.main()
