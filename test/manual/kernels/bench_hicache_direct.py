"""Compare HiCache `direct` and `direct_kernel` L2 transfers on GLM-5.2 shapes.

Drives MLATokenToKVPoolHost + DSAIndexerPoolHost the way HiCacheController does:
indices are moved per pool as `move_indices` would, then the KV pool and the
indexer pool transfer in order. Reports wall time to completion (what the
scheduler waits for on write-back) and GPU time between events.

    python test/manual/kernels/bench_hicache_direct.py --layers 78 79 --tokens 1024 16384
"""

import argparse
import statistics
import time

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

PAGE_SIZE = 64


def _move_indices(io_backend, host_indices, device_indices):
    if io_backend == "direct_kernel":
        return host_indices.to("cuda", non_blocking=True), device_indices
    return host_indices, device_indices.cpu()


def _build(layers: int, device_tokens: int, host_tokens: int, draft: bool):
    def device_pool(layer_num):
        return DSATokenToKVPool(
            size=device_tokens,
            page_size=PAGE_SIZE,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
        )

    target = device_pool(layers - 1 if draft else layers)
    drafts = (device_pool(1),) if draft else ()
    kv_host = MLATokenToKVPoolHost(
        buffer_info=target.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=host_tokens / device_tokens,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="page_first_direct",
        pin_memory=True,
        device="cpu",
        override_kv_cache_dim=target.kv_cache_dim,
        draft_buffer_infos=tuple(
            d.get_device_pool_infos()[0].buffer_info for d in drafts
        ),
        device_capacity_tokens=target.size,
        start_layer=target.start_layer,
        end_layer=target.end_layer,
        target_layer_num=target.layer_num,
    )
    indexer_host = DSAIndexerPoolHost(
        decl=make_dsa_indexer_pool_decl(target),
        anchor_host=kv_host,
        packed_draft_device_pools=drafts,
    )
    return target, drafts, kv_host, indexer_host


def _time(fn, iters, l2_flush):
    walls, gpus = [], []
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for i in range(iters + 3):
        # Evicted pages were written long before their write-back, so read them L2-cold.
        l2_flush.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        if i >= 3:
            walls.append((time.perf_counter() - t0) * 1e3)
            gpus.append(start.elapsed_time(end))
    return statistics.median(walls), statistics.median(gpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[19, 20, 21, 78, 79])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 8192, 16384])
    parser.add_argument("--device-tokens", type=int, default=128 * 1024)
    parser.add_argument("--host-tokens", type=int, default=256 * 1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--per-pool", action="store_true", help="also time KV and indexer pools alone"
    )
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(0)
    l2_flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    side_stream = torch.cuda.Stream()
    print(
        f"{'layers':>6} {'tokens':>6} {'op':>10} {'backend':>13} {'wall_ms':>9} {'gpu_ms':>9}"
    )
    for layers in args.layers:
        # A packed MTP layer rides on DEP4 ranks and the last PP4 stage.
        draft = layers in (21, 79)
        target, drafts, kv_host, indexer_host = _build(
            layers, args.device_tokens, args.host_tokens, draft
        )
        for pool in (kv_host, indexer_host):
            pool.prepare_io_backend("direct_kernel")
        transfer_layers = target.layer_num
        for tokens in args.tokens:
            pages = tokens // PAGE_SIZE
            device_pages = (
                torch.randperm(args.device_tokens // PAGE_SIZE - 1, generator=gen)[
                    :pages
                ]
                + 1
            )
            host_pages = torch.randperm(kv_host.page_num, generator=gen)[:pages]
            offsets = torch.arange(PAGE_SIZE)
            device_indices = (
                (device_pages[:, None] * PAGE_SIZE + offsets).reshape(-1).cuda()
            )
            host_indices = (host_pages[:, None] * PAGE_SIZE + offsets).reshape(-1)

            for backend in ("direct", "direct_kernel"):

                def write_back(pools=(kv_host, indexer_host)):
                    for pool in pools:
                        h, d = _move_indices(backend, host_indices, device_indices)
                        pool.backup_from_device_all_layer(target, h, d, backend)

                def load_back(pools=(kv_host, indexer_host)):
                    moved = [
                        _move_indices(backend, host_indices, device_indices)
                        for _ in pools
                    ]
                    for layer_id in range(transfer_layers):
                        for pool, (h, d) in zip(pools, moved):
                            pool.load_to_device_per_layer(
                                target, h, d, layer_id, backend
                            )
                    for depth, draft_pool in enumerate(drafts):
                        for pool, (h, d) in zip(pools, moved):
                            pool.load_to_device_per_layer(
                                draft_pool,
                                h,
                                d,
                                transfer_layers + depth,
                                backend,
                                is_draft=True,
                            )

                cases = [("write_back", write_back), ("load_back", load_back)]
                if args.per_pool:
                    for tag, pool in (("kv", kv_host), ("idx", indexer_host)):
                        cases.append((f"wb_{tag}", lambda p=pool: write_back((p,))))
                        cases.append((f"lb_{tag}", lambda p=pool: load_back((p,))))
                for name, fn in cases:
                    # L2TransferEngine runs transfers on side streams.
                    with torch.cuda.stream(side_stream):
                        wall, gpu = _time(fn, args.iters, l2_flush)
                    print(
                        f"{layers:>6} {tokens:>6} {name:>10} {backend:>13} {wall:>9.3f} {gpu:>9.3f}",
                        flush=True,
                    )
        kv_host.destroy()
        _cuda_host_unregister(indexer_host.index_k_with_scale_buffer)
        target = drafts = kv_host = indexer_host = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
