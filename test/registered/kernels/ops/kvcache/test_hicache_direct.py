"""Bit-exactness of the page_first_direct HiCache kernels behind ``direct_kernel``.

Device layers are separate allocations reached through a pointer table, and the
host pool is a cudaHostRegister-ed [page, layer, page_bytes] buffer, as in
MLATokenToKVPoolHost / DSAIndexerPoolHost. Every case compares whole buffers, so
it also checks that bytes outside the transferred pages are untouched.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache_direct import (
    WRITE_BACK_LINES,
    WRITE_BACK_PAGE_FUSED,
    WRITE_BACK_PAGE_LAYER,
    HiCacheDirectTiling,
    hicache_direct_tiling,
    transfer_hicache_direct_load,
    transfer_hicache_direct_write_back,
)
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="HiCache direct_kernel transfers are CUDA-only.",
)

DEVICE = "cuda"
PAGE_SIZE = 64
# GLM-5.2 fp8 MLA KV page (raw 576 B rows) and DSA indexer page (128 B key + 4 B scale).
KV_PAGE_BYTES = PAGE_SIZE * 576
INDEXER_PAGE_BYTES = PAGE_SIZE * 132
# Host layers per rank: a PP4 stage (19/20), a PP4 last stage with the packed MTP
# layer (21), and DEP4 with the packed MTP layer (79).
LAYER_COUNTS = [19, 21, 79]


def _fused_threads(page_bytes: int) -> int:
    vec_bytes = 32 if torch.cuda.get_device_capability()[0] >= 10 else 16
    for threads in range(1024, 31, -8):
        if page_bytes % (threads * vec_bytes) == 0:
            return threads
    raise AssertionError(f"no CTA size tiles {page_bytes} B pages")


def _write_back_tilings(page_bytes: int) -> list[HiCacheDirectTiling]:
    default = hicache_direct_tiling(page_bytes)
    tilings = {
        default,
        default._replace(write_back_variant=WRITE_BACK_LINES, write_back_param=4),
        default._replace(
            write_back_variant=WRITE_BACK_PAGE_FUSED,
            write_back_param=_fused_threads(page_bytes),
        ),
    }
    if page_bytes % (256 * 16) == 0:
        tilings.add(
            default._replace(
                write_back_variant=WRITE_BACK_PAGE_LAYER, write_back_param=256
            )
        )
    return sorted(tilings)


def _registered_host(num_pages: int, num_layers: int, page_bytes: int) -> torch.Tensor:
    host = alloc_with_host_register(
        (num_pages, num_layers, page_bytes),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
        allocator=HostTensorAllocator(),
    )
    host.copy_(torch.randint(0, 256, host.shape, dtype=torch.uint8))
    return host


def _token_indices(pages: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(PAGE_SIZE, dtype=torch.int64, device=pages.device)
    return (pages[:, None] * PAGE_SIZE + offsets).reshape(-1)


class _Pools:
    def __init__(self, *, num_layers, num_pages, page_bytes, seed):
        gen = torch.Generator().manual_seed(seed)
        self.page_bytes = page_bytes
        num_device_pages = 2 * num_pages + 3
        num_host_pages = 2 * num_pages + 5
        # Separate allocations, like the per-layer device KV and indexer buffers.
        self.device_layers = [
            torch.randint(0, 256, (num_device_pages, page_bytes), dtype=torch.uint8).to(
                DEVICE
            )
            for _ in range(num_layers)
        ]
        self.layer_ptrs = torch.tensor(
            [t.data_ptr() for t in self.device_layers],
            dtype=torch.uint64,
            device=DEVICE,
        )
        self.host = _registered_host(num_host_pages, num_layers, page_bytes)
        # Device page 0 is the padding page.
        self.device_pages = (
            torch.randperm(num_device_pages - 1, generator=gen)[:num_pages] + 1
        )
        self.host_pages = torch.randperm(num_host_pages, generator=gen)[:num_pages]
        self.device_indices = _token_indices(self.device_pages).to(DEVICE)
        self.host_indices = _token_indices(self.host_pages).to(DEVICE)

    def release(self):
        _cuda_host_unregister(self.host)


@pytest.fixture
def make_pools():
    created = []

    def make(**kwargs):
        pools = _Pools(**kwargs)
        created.append(pools)
        return pools

    yield make
    torch.cuda.synchronize()
    for pools in created:
        pools.release()


@pytest.mark.parametrize("num_layers", LAYER_COUNTS)
@pytest.mark.parametrize("num_pages", [1, 17])
@pytest.mark.parametrize("page_bytes", [KV_PAGE_BYTES, INDEXER_PAGE_BYTES])
def test_write_back(make_pools, page_bytes, num_pages, num_layers):
    pools = make_pools(
        num_layers=num_layers,
        num_pages=num_pages,
        page_bytes=page_bytes,
        seed=num_layers,
    )
    for tiling in _write_back_tilings(page_bytes):
        pools.host.copy_(torch.randint(0, 256, pools.host.shape, dtype=torch.uint8))
        expected = pools.host.clone()
        for layer, device_layer in enumerate(pools.device_layers):
            expected[pools.host_pages, layer] = device_layer[
                pools.device_pages.to(DEVICE)
            ].cpu()

        transfer_hicache_direct_write_back(
            host_pool=pools.host,
            device_layer_ptrs=pools.layer_ptrs,
            host_indices=pools.host_indices,
            device_indices=pools.device_indices,
            page_size=PAGE_SIZE,
            page_bytes=page_bytes,
            tiling=tiling,
        )
        torch.cuda.synchronize()
        assert torch.equal(pools.host, expected), tiling


@pytest.mark.parametrize("num_layers", LAYER_COUNTS)
@pytest.mark.parametrize("num_pages", [1, 17])
@pytest.mark.parametrize("page_bytes", [KV_PAGE_BYTES, INDEXER_PAGE_BYTES])
def test_load(make_pools, page_bytes, num_pages, num_layers):
    pools = make_pools(
        num_layers=num_layers,
        num_pages=num_pages,
        page_bytes=page_bytes,
        seed=num_layers,
    )
    for layer in (0, num_layers - 1):
        device_layer = pools.device_layers[layer]
        expected = device_layer.clone()
        expected[pools.device_pages.to(DEVICE)] = pools.host[
            pools.host_pages, layer
        ].to(DEVICE)

        transfer_hicache_direct_load(
            device_layer=device_layer,
            host_pool=pools.host,
            host_indices=pools.host_indices,
            device_indices=pools.device_indices,
            host_layer_id=layer,
            page_size=PAGE_SIZE,
            page_bytes=page_bytes,
        )
        torch.cuda.synchronize()
        assert torch.equal(device_layer, expected)


# 656 B rows are the fp8 FlashMLA layout; 64 x 528 B pages fit neither page
# tiling, so they exercise the line write-back and the generic load chunking.
@pytest.mark.parametrize("page_bytes", [PAGE_SIZE * 656, PAGE_SIZE * 528])
def test_untuned_page_sizes_round_trip(make_pools, page_bytes):
    assert hicache_direct_tiling(page_bytes) is not None
    pools = make_pools(num_layers=3, num_pages=5, page_bytes=page_bytes, seed=0)
    originals = [layer.clone() for layer in pools.device_layers]
    transfer_hicache_direct_write_back(
        host_pool=pools.host,
        device_layer_ptrs=pools.layer_ptrs,
        host_indices=pools.host_indices,
        device_indices=pools.device_indices,
        page_size=PAGE_SIZE,
        page_bytes=page_bytes,
    )
    for layer in pools.device_layers:
        layer.zero_()
    for layer_id, layer in enumerate(pools.device_layers):
        transfer_hicache_direct_load(
            device_layer=layer,
            host_pool=pools.host,
            host_indices=pools.host_indices,
            device_indices=pools.device_indices,
            host_layer_id=layer_id,
            page_size=PAGE_SIZE,
            page_bytes=page_bytes,
        )
    torch.cuda.synchronize()
    device_pages = pools.device_pages.to(DEVICE)
    for layer, original in zip(pools.device_layers, originals):
        assert torch.equal(layer[device_pages], original[device_pages])


def test_rejects_host_layout_mismatch(make_pools):
    """A host pool whose page size or layer count disagrees with the call must not be written."""
    pools = make_pools(num_layers=3, num_pages=2, page_bytes=KV_PAGE_BYTES, seed=0)
    snapshot = pools.host.clone()
    with pytest.raises(Exception, match="host page"):
        transfer_hicache_direct_write_back(
            host_pool=pools.host.view(pools.host.shape[0], 6, KV_PAGE_BYTES // 2),
            device_layer_ptrs=pools.layer_ptrs,
            host_indices=pools.host_indices,
            device_indices=pools.device_indices,
            page_size=PAGE_SIZE,
            page_bytes=KV_PAGE_BYTES,
        )
    with pytest.raises(Exception):
        transfer_hicache_direct_write_back(
            host_pool=pools.host,
            device_layer_ptrs=pools.layer_ptrs[:2],
            host_indices=pools.host_indices,
            device_indices=pools.device_indices,
            page_size=PAGE_SIZE,
            page_bytes=KV_PAGE_BYTES,
        )
    torch.cuda.synchronize()
    assert torch.equal(pools.host, snapshot)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
