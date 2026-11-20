"""Device <-> host round trip of the borrow-API HiCache data plane (GPU)."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")

import threading
import unittest

import torch

from sglang.srt.configs.inkling import (
    InklingConvCacheParams,
    InklingConvStateShape,
    InklingStateDType,
)
from sglang.srt.mem_cache.borrow.hicache.gpu_data_plane import GpuDataPlane, HostMirror
from sglang.srt.mem_cache.borrow.hicache.gpu_state_plane import GpuStatePlane
from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


def _slots(pages, ps):
    return (torch.tensor(pages)[:, None] * ps + torch.arange(ps)).flatten().cuda()


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGpuDataPlane(unittest.TestCase):
    def _make(self, ps=4, layers=3):
        kv = MHATokenToKVPool(
            size=64 * ps,
            page_size=ps,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=layers,
            device="cuda",
            enable_memory_saver=False,
        )
        info = kv.get_device_pool_infos()[0].buffer_info
        host = MHATokenToKVPoolHost(
            info,
            2.0,
            0,
            ps,
            "page_first",
            device_capacity_tokens=kv.size,
            start_layer=kv.start_layer,
            end_layer=kv.end_layer,
            write_back_staging=kv.hicache_write_back_staging,
        )
        dp = GpuDataPlane(
            mirrors=[HostMirror(device_pool=kv, host_pool=host)],
            page_size=ps,
            io_backend="kernel",
            device=torch.device("cuda"),
        )
        self.addCleanup(dp.close)
        return kv, dp

    def _layers(self, kv, slots):
        return [
            (kv.k_buffer[l][slots].clone(), kv.v_buffer[l][slots].clone())
            for l in range(kv.layer_num)
        ]

    def test_backup_then_load_restores_every_layer(self):
        """KV backed up from device pages and loaded into other pages arrives
        intact in every layer, ordered after the event the copies wait on."""
        ps = 4
        kv, dp = self._make(ps=ps)
        src = _slots([1, 2, 3], ps)
        for l in range(kv.layer_num):
            kv.k_buffer[l][src] = torch.randn_like(kv.k_buffer[l][src])
            kv.v_buffer[l][src] = torch.randn_like(kv.v_buffer[l][src])
        written = torch.cuda.Event()
        written.record()
        expect = self._layers(kv, src)

        host_pages = dp.alloc_host(3)
        op = dp.backup([1, 2, 3], host_pages, after=written)
        op.wait()
        for l in range(kv.layer_num):
            kv.k_buffer[l][src] = 0
            kv.v_buffer[l][src] = 0
        cleared = torch.cuda.Event()
        cleared.record()

        load = dp.queue_load(host_pages, [7, 8, 9])
        self.assertFalse(load.launched)
        fence = dp.launch_loads(after=cleared)
        self.assertIsNotNone(fence)
        torch.cuda.current_stream().wait_event(fence.final_event)
        got = self._layers(kv, _slots([7, 8, 9], ps))
        for (ek, ev), (gk, gv) in zip(expect, got):
            torch.testing.assert_close(gk, ek, rtol=0, atol=0)
            torch.testing.assert_close(gv, ev, rtol=0, atol=0)
        self.assertIsNone(dp.launch_loads(after=None))

    def test_load_launched_from_another_thread_orders_the_forward(self):
        """A load launched on another thread is visible to a stream that waits
        on the returned fence (the service thread / scheduler thread split)."""
        ps = 2
        kv, dp = self._make(ps=ps, layers=2)
        src = _slots([1], ps)
        kv.k_buffer[0][src] = 3.0
        written = torch.cuda.Event()
        written.record()
        host_pages = dp.alloc_host(1)
        dp.backup([1], host_pages, after=written).wait()
        dp.queue_load(host_pages, [5])
        out = {}

        def service_thread():
            torch.cuda.set_device(0)
            out["fence"] = dp.launch_loads(after=None)

        t = threading.Thread(target=service_thread)
        t.start()
        t.join()
        torch.cuda.current_stream().wait_event(out["fence"].final_event)
        self.assertTrue(torch.all(kv.k_buffer[0][_slots([5], ps)] == 3.0))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGpuStatePlane(unittest.TestCase):
    def _make(self, layers=(0, 2, 5)):
        pool = MambaPool(
            size=8,
            spec_state_size=0,
            cache_params=InklingConvCacheParams(
                shape=InklingConvStateShape(conv=[(4, 8)], temporal=(2, 16, 16)),
                layers=list(layers),
                dtype=InklingStateDType(conv=torch.bfloat16, temporal=torch.float32),
            ),
            mamba_layer_ids=list(layers),
            device="cuda",
        )
        (info,) = pool.get_device_pool_infos()
        host = MambaPoolHost(
            info.buffer_info, 2.0, 0, device_capacity=pool.size, layout="page_first"
        )
        sp = GpuStatePlane(
            device_pool=pool, host_pool=host, io_backend="kernel", device=torch.device("cuda")
        )
        self.addCleanup(sp.close)
        return pool, sp

    @staticmethod
    def _slot(pool, slot):
        cache = pool.mamba_cache
        return [c[:, slot].clone() for c in cache.conv] + [cache.temporal[:, slot].clone()]

    @staticmethod
    def _fill(pool, slot, value):
        cache = pool.mamba_cache
        for c in cache.conv:
            c[:, slot] = torch.randn_like(c[:, slot]) + value
        cache.temporal[:, slot] = torch.randn_like(cache.temporal[:, slot]) + value

    def _assert_same(self, a, b):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_round_trip_ordered_after_the_writer(self):
        """State written on another stream after a long kernel is what the
        plane copies, backs up and loads: every op waits on ``after``, and the
        fence event covers every op launched before it."""
        pool, sp = self._make()
        writer = torch.cuda.Stream()
        with torch.cuda.stream(writer):
            torch.cuda._sleep(200_000_000)
            self._fill(pool, 1, 5.0)
            written = torch.cuda.Event()
            written.record(writer)
        sp.copy(1, 2, after=written)
        (host,) = sp.alloc_host(1)
        sp.backup(1, host, after=written)
        sp.zero(1, after=written)
        sp.load(host, 4, after=written)
        fence = sp.fence_event(also_after=None)
        torch.cuda.current_stream().wait_event(fence)
        writer.synchronize()
        expect = [x.clone() for x in self._slot(pool, 2)]
        self.assertGreater(float(expect[-1].abs().sum()), 0.0)
        self._assert_same(self._slot(pool, 4), expect)
        for x in self._slot(pool, 1):
            self.assertEqual(float(x.abs().sum()), 0.0)
        self.assertIsNone(sp.fence_event(also_after=None))


if __name__ == "__main__":
    unittest.main()
