"""Start the HiCache service on its own thread and hand the scheduler its facade."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from sglang.srt.managers.borrow_cache.client import BorrowCacheClient
from sglang.srt.managers.borrow_cache.facade import BorrowTreeFacade
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.borrow.hicache.service import HiCacheService, ServiceConfig
from sglang.srt.mem_cache.borrow.threaded import ThreadedService
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode, HiCacheDraftPlan

logger = logging.getLogger(__name__)


def _unsupported(
    *,
    model_config: Any,
    req_to_token_pool: Any,
    allocator: Any,
    spec_algorithm: Any,
    draft_mode: HiCacheDraftMode,
) -> list[str]:
    memory, parallel = get_memory(), get_parallel()
    checks = {
        "KV pools other than full-attention MHA/MLA": type(allocator.get_kvcache())
        not in (MHATokenToKVPool, MLATokenToKVPool),
        "recurrent state (Mamba / linear attention)": allocator.state is not None,
        "request tables other than ReqToTokenPool": type(req_to_token_pool)
        is not ReqToTokenPool,
        "pipeline parallelism": parallel.pp_size > 1,
        "decode context parallelism": parallel.dcp_enabled,
        "attention context parallelism": parallel.attn_cp_size > 1,
        "PD decode workers": get_disagg().disaggregation_mode == "decode",
        "encoder-decoder models": model_config.is_encoder_decoder,
        "unified memory": memory.enable_unified_memory,
        "session radix cache": memory.enable_session_radix_cache,
        "HiCache storage (L3)": bool(memory.hicache_storage_backend),
        "--hicache-write-policy other than write_through": memory.hicache_write_policy
        != "write_through",
        "--hicache-host-memory-mode buffer_only": memory.hicache_host_memory_mode
        == "buffer_only",
        "--radix-cache-backend": bool(memory.radix_cache_backend),
        "the external cache linker": memory.enable_unified_cache_external_linker,
        "speculative decoding other than EAGLE / EAGLE3": not (
            spec_algorithm.is_none()
            or (spec_algorithm.is_eagle() and not spec_algorithm.is_frozen_kv_mtp())
        ),
        "draft KV fused into target page envelopes": draft_mode
        == HiCacheDraftMode.FUSED,
    }
    return [name for name, bad in checks.items() if bad]


def build_borrow_tree_facade(
    *,
    tp_worker: Any,
    model_config: Any,
    req_to_token_pool: Any,
    token_to_kv_pool_allocator: Any,
    page_size: int,
    enable_hierarchical_cache: bool,
    spec_algorithm: Any,
    draft_plan: Optional[HiCacheDraftPlan],
) -> BorrowTreeFacade:
    kv_pool = token_to_kv_pool_allocator.get_kvcache()
    draft_plan = draft_plan or HiCacheDraftPlan()
    if bad := _unsupported(
        model_config=model_config,
        req_to_token_pool=req_to_token_pool,
        allocator=token_to_kv_pool_allocator,
        spec_algorithm=spec_algorithm,
        draft_mode=draft_plan.mode,
    ):
        raise ValueError(
            "SGLANG_ENABLE_DECOUPLED_HICACHE does not support: " + ", ".join(bad)
        )
    packed = tuple(
        draft_plan.device_pools if draft_plan.mode == HiCacheDraftMode.PACKED else ()
    )
    sidecar_pools = tuple(
        draft_plan.device_pools if draft_plan.mode == HiCacheDraftMode.SIDECAR else ()
    )
    host_pool, sidecars = None, ()
    if enable_hierarchical_cache:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_kv_host_pool,
        )

        host_pool = build_kv_host_pool(
            kv_pool=kv_pool, page_size=page_size, mtp_draft_device_pools=packed
        )
        sidecars = tuple(
            (pool, build_kv_host_pool(kv_pool=pool, page_size=page_size, pool_label="draft"))
            for pool in sidecar_pools
        )
    ratio = NewTokenRatioTracker.from_config()
    config = ServiceConfig(
        page_size=page_size,
        num_device_pages=token_to_kv_pool_allocator.size // page_size,
        init_new_token_ratio=ratio.init,
        min_new_token_ratio=ratio.min,
        new_token_ratio_decay=ratio.decay,
    )
    device = torch.device(kv_pool.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    io_backend = get_memory().hicache_io_backend

    def make_service() -> HiCacheService:
        data_plane = None
        if host_pool is not None:
            from sglang.srt.mem_cache.borrow.hicache.gpu_data_plane import (
                GpuDataPlane,
            )

            data_plane = GpuDataPlane(
                kv_pool=kv_pool,
                host_pool=host_pool,
                page_size=page_size,
                io_backend=io_backend,
                device=device,
                packed_draft_pools=packed,
                sidecars=sidecars,
            )
        return HiCacheService(config=config, data_plane=data_plane)

    service = ThreadedService(
        make_service,
        cuda_device=device.index if device.type == "cuda" else None,
        name=f"hicache-service-tp{get_parallel().tp_rank}",
    )
    logger.info(
        "Decoupled HiCache service started: %d device pages of %d tokens, host tier %s",
        config.num_device_pages,
        page_size,
        "on" if host_pool is not None else "off",
    )
    return BorrowTreeFacade(
        client=BorrowCacheClient(service=service, page_size=page_size),
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
    )
