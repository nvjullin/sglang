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

from fake_runtime import FakeDataPlane, FakeReq, FakeRuntime, kv_content

from sglang.srt.mem_cache.borrow.api import AcquireReq, CacheKey, LeaseAttrs
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.service import HiCacheService, ServiceConfig
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
                decoding = [r for r in running if len(r.slots) >= len(r.prompt)]
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


class TestThreaded(unittest.TestCase):
    def test_runs_on_its_own_thread_in_caller_order(self):
        seen = []

        class Probe(HiCacheService):
            def release_mut(self, lease, token_ids):
                seen.append(("release_mut", threading.current_thread().name))
                super().release_mut(lease, token_ids)

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
