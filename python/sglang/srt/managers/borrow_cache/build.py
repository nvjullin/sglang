"""Start the HiCache service on its own thread and hand the scheduler its facade."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import msgspec
import torch

from sglang.srt.managers.borrow_cache.client import BorrowCacheClient
from sglang.srt.managers.borrow_cache.facade import BorrowTreeFacade
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.borrow.hicache.gpu_data_plane import GpuDataPlane, HostMirror
from sglang.srt.mem_cache.borrow.hicache.gpu_state_plane import GpuStatePlane
from sglang.srt.mem_cache.borrow.hicache.service import (
    HiCacheService,
    ServiceConfig,
    StateConfig,
)
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _make_layer_mapper,
    _state_host_layout,
    build_host_pool_group,
    build_kv_host_pool,
    build_mamba_host_pool,
)
from sglang.srt.mem_cache.borrow.hicache.windows import WindowConfig
from sglang.srt.mem_cache.borrow.threaded import ThreadedService
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import host_only_mamba_checkpoint
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    mamba_checkpoint_grid,
)
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
    state = allocator.state
    table = HybridReqToTokenPool if state is not None else ReqToTokenPool
    kv_pool = allocator.get_kvcache()
    checks = {
        "KV pools other than MHA/MLA/DSA (alone, in a Mamba hybrid pool, or an "
        "MHA full + sliding-window pair)": not _supported_kv_pool(kv_pool),
        "sliding-window token allocators other than SWATokenToKVPoolAllocator (paged)": isinstance(
            kv_pool, BaseSWAKVPool
        )
        and type(allocator) is not SWATokenToKVPoolAllocator,
        "encoder SWA bounded replay": get_exec().features.enable_encoder_swa_bounded_replay,
        "decoder SWA bounded replay outside DeepSeek V4 pools": bool(
            get_exec().features.enable_decoder_swa_bounded_replay
        )
        and type(kv_pool) is not DeepSeekV4TokenToKVPool,
        "the HiCache host tier with DeepSeek V4 pools": type(kv_pool)
        is DeepSeekV4TokenToKVPool
        and memory.enable_hierarchical_cache,
        "speculative decoding with a sliding-window pool": isinstance(kv_pool, BaseSWAKVPool)
        and not spec_algorithm.is_none(),
        "request tables other than ReqToTokenPool (HybridReqToTokenPool with "
        "recurrent state)": type(req_to_token_pool) is not table,
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
    if state is not None:
        checks.update(_unsupported_state(state, spec_algorithm=spec_algorithm))
    return [name for name, bad in checks.items() if bad]


def _unsupported_state(state: Any, *, spec_algorithm: Any) -> dict[str, bool]:
    mamba, memory = get_exec().mamba, get_memory()
    pool = state.pool
    return {
        "recurrent state pools other than a static MambaPool": type(pool)
        is not MambaPool,
        "--mamba-radix-cache-strategy other than extra_buffer": not mamba.enable_mamba_extra_buffer,
        "lazy Mamba tracking": mamba.enable_mamba_extra_buffer_lazy,
        "--hybrid-cache-checkpoints-level above 2": mamba.hybrid_cache_checkpoints_level
        > 2,
        "int8 Mamba checkpoints": state.ckpt_pool is not None,
        "host-only Mamba checkpoints": host_only_mamba_checkpoint(),
        "ReplaySSM state rings": pool.replayssm_write_pos is not None,
        "model-owned recurrent state": state.own_state is not None,
        "per-slot side states (short conv / N-gram)": bool(pool._slot_siblings)
        or state.short_conv_pool.enabled
        or state.ngram_pool.enabled,
        "speculative decoding with recurrent state": not spec_algorithm.is_none(),
        "--hicache-size with recurrent state": memory.hicache_size > 0,
    }


def _state_config(state: Any, *, req_to_token_pool: Any, page_size: int) -> StateConfig:
    return StateConfig(
        num_slots=state.slots.size,
        # Slot 0 is the padding rows' write target.
        first_slot=1,
        tracking_slots=req_to_token_pool.mamba_initial_tracking_slots,
        checkpoint_grid=mamba_checkpoint_grid(page_size),
        publish_at_prompt_end=get_disagg().disaggregation_mode == "prefill",
    )


def _state_host_pool(mamba_pool: Any) -> Any:
    (info,) = mamba_pool.get_device_pool_infos()
    info.buffer_info.validate(layer_ids=info.layer_ids)
    # host_size 0: sized by --hicache-ratio.
    return build_mamba_host_pool(
        mamba_pool, info, host_size=0, layout=_state_host_layout()
    )


def _supported_kv_pool(kv_pool: Any) -> bool:
    if type(kv_pool) is DeepSeekV4TokenToKVPool:
        # Device only: its compressed KV and indexer rows are full_loc // ratio
        # sidecars of the full pages; its window KV is the window pool.
        return True
    if type(kv_pool) is SWAKVPool:
        return all(
            type(p) is MHATokenToKVPool for p in (kv_pool.full_kv_pool, kv_pool.swa_kv_pool)
        )
    return type(_full_kv_pool(kv_pool)) in (
        MHATokenToKVPool,
        MLATokenToKVPool,
        DSATokenToKVPool,
    )


def _full_kv_pool(kv_pool: Any) -> Any:
    if isinstance(kv_pool, (HybridLinearKVPool, SWAKVPool)):
        return kv_pool.full_kv_pool
    return kv_pool


def _swa_mirrors(*, kv_pool: SWAKVPool, page_size: int) -> list[HostMirror]:
    """Host pools for the full and the sliding-window pool; the window pool's
    pages are their own id space."""
    start = kv_pool.start_layer
    _, transfer_layer_id_max = _kv_transfer_layers(kv_pool)
    mirrors = []
    for pool, is_swa, label in (
        (kv_pool.full_kv_pool, False, "kv"),
        (kv_pool.swa_kv_pool, True, "swa"),
    ):
        mapping = {
            layer - start: index
            for layer, (index, swa) in kv_pool.layers_mapping.items()
            if swa == is_swa
        }
        mirrors.append(
            HostMirror(
                device_pool=pool,
                host_pool=build_kv_host_pool(kv_pool=pool, page_size=page_size, pool_label=label),
                layer_mapper=_make_layer_mapper(mapping, transfer_layer_id_max),
                space="window" if is_swa else "kv",
            )
        )
    return mirrors


def _window_config(allocator: Any, *, page_size: int, window: int) -> WindowConfig:
    return WindowConfig(
        num_pages=allocator.size_swa // page_size,
        window_tokens=window,
        retire_interval=max(1, envs.SGLANG_SWA_EVICTION_INTERVAL.get()),
    )


def _pool_table_binder(allocator: Any, *, page_size: int) -> Callable[[], None]:
    """Scheduler-side table setup the service relies on; re-run after a pool clear.

    The scheduler's own state slot allocator is drained: the service owns every
    state slot, so a stray allocation fails instead of aliasing one.
    """

    def bind() -> None:
        if allocator.state is not None:
            slots = allocator.state.slots
            if slots.available_size():
                assert slots.alloc(slots.available_size()) is not None

    return bind


def _kv_transfer_layers(kv_pool: Any) -> tuple[dict[int, int], int]:
    """(transfer layer id -> KV pool layer, transfer id bound). A hybrid
    pool's attention layers are a subset of the stage's layer ids."""
    if isinstance(kv_pool, HybridLinearKVPool):
        mapping = {
            layer - kv_pool.start_layer: index
            for layer, index in kv_pool.full_attention_layer_id_mapping.items()
        }
        return mapping, max(mapping) + 1
    if isinstance(kv_pool, SWAKVPool):
        return {}, max(kv_pool.layers_mapping) - kv_pool.start_layer + 1
    return {i: i for i in range(kv_pool.layer_num)}, kv_pool.layer_num


def _host_mirrors(
    *, kv_pool: Any, page_size: int, packed: tuple, sidecar_pools: tuple
) -> list[HostMirror]:
    """Host pools for the device KV, built as the HiCache host stack builds them
    (target KV plus its sidecar buffers such as a DSA indexer), and one host
    pool per EAGLE draft pool, all over the same page ids."""
    if isinstance(kv_pool, SWAKVPool):
        return _swa_mirrors(kv_pool=kv_pool, page_size=page_size)
    layer_mapping, transfer_layer_id_max = _kv_transfer_layers(kv_pool)
    group = build_host_pool_group(
        config=prepare_host_pool_config(
            root=kv_pool,
            full_layer_mapping=layer_mapping,
            transfer_layer_id_max=transfer_layer_id_max,
            transfer_page_size=page_size,
            packed_draft_device_pools=packed,
        )
    )
    mirrors = [
        HostMirror(
            device_pool=e.device_pool,
            host_pool=e.host_pool,
            layer_mapper=e.layer_mapper,
            packed_draft_pools=tuple(e.packed_draft_device_pools),
        )
        for e in group.entries
    ]
    for pool in sidecar_pools:
        host = build_kv_host_pool(kv_pool=pool, page_size=page_size, pool_label="draft")
        mirrors.append(HostMirror(device_pool=pool, host_pool=host))
    return mirrors


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
    swa_retain_window: Optional[int],
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
    mirrors = _host_mirrors(
        kv_pool=kv_pool,
        page_size=page_size,
        packed=packed,
        sidecar_pools=sidecar_pools,
    ) if enable_hierarchical_cache else None
    ratio = NewTokenRatioTracker.from_config()
    state = token_to_kv_pool_allocator.state
    config = ServiceConfig(
        page_size=page_size,
        num_device_pages=token_to_kv_pool_allocator.size // page_size,
        init_new_token_ratio=ratio.init,
        min_new_token_ratio=ratio.min,
        new_token_ratio_decay=ratio.decay,
        state=None
        if state is None
        else _state_config(state, req_to_token_pool=req_to_token_pool, page_size=page_size),
        window=None
        if not isinstance(kv_pool, BaseSWAKVPool)
        else _window_config(
            token_to_kv_pool_allocator, page_size=page_size, window=swa_retain_window
        ),
    )
    if isinstance(kv_pool, BaseSWAKVPool):
        # Full pages are no longer capped by the window pool.
        config = msgspec.structs.replace(
            config, num_device_pages=token_to_kv_pool_allocator.size_full // page_size
        )
    device = torch.device(_full_kv_pool(kv_pool).device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    io_backend = get_memory().hicache_io_backend
    _, transfer_layer_id_max = _kv_transfer_layers(kv_pool)
    state_host_pool = (
        _state_host_pool(state.pool)
        if state is not None and enable_hierarchical_cache
        else None
    )

    def make_service() -> HiCacheService:
        data_plane = state_plane = None
        if mirrors is not None:
            data_plane = GpuDataPlane(
                mirrors=mirrors,
                page_size=page_size,
                io_backend=io_backend,
                device=device,
                transfer_layer_id_max=transfer_layer_id_max,
            )
        if state is not None:
            state_plane = GpuStatePlane(
                device_pool=state.pool,
                host_pool=state_host_pool,
                io_backend=io_backend,
                device=device,
            )
        return HiCacheService(
            config=config, data_plane=data_plane, state_plane=state_plane
        )

    service = ThreadedService(
        make_service,
        cuda_device=device.index if device.type == "cuda" else None,
        name=f"hicache-service-tp{get_parallel().tp_rank}",
    )
    bind_pool_tables = _pool_table_binder(token_to_kv_pool_allocator, page_size=page_size)
    bind_pool_tables()
    logger.info(
        "Decoupled HiCache service started: %d device pages of %d tokens, "
        "%d state slots, host tier %s",
        config.num_device_pages,
        page_size,
        0 if config.state is None else config.state.num_slots,
        "on" if mirrors is not None else "off",
    )
    return BorrowTreeFacade(
        client=BorrowCacheClient(
            service=service,
            page_size=page_size,
            window=None
            if config.window is None
            else (config.window.window_tokens, config.window.retire_interval),
            # As the old tree caps a match under decoder bounded replay.
            reprefill_tail=swa_retain_window
            if get_exec().features.enable_decoder_swa_bounded_replay
            else 0,
        ),
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        bind_pool_tables=bind_pool_tables,
    )
