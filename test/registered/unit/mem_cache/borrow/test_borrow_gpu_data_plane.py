"""Device <-> host round trip of the borrow-API HiCache data plane (GPU)."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")

import threading
import unittest

import torch

from sglang.srt.mem_cache.borrow.hicache.gpu_data_plane import GpuDataPlane
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
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
            kv_pool=kv,
            host_pool=host,
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


if __name__ == "__main__":
    unittest.main()
