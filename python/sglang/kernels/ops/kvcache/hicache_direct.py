"""Page-granular HiCache transfers for the ``page_first_direct`` host layout.

These back ``--hicache-io-backend direct_kernel``: SM copy kernels that read the
transfer indices on the GPU and access the pinned host pool in place, instead of
the ``direct`` backend's per-page copy-engine calls driven by CPU indices.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Mirrors HiCacheDirectWriteBack in kvcacheio/hicache_direct.cuh.
WRITE_BACK_PAGE_LAYER = 0
WRITE_BACK_PAGE_FUSED = 1
WRITE_BACK_LINES = 2


class HiCacheDirectTiling(NamedTuple):
    write_back_variant: int
    # CTA threads for the page variants; lines in flight per thread for LINES.
    write_back_param: int
    load_chunks: int


# Measured on GB300 for GLM-5.2's fp8 MLA KV page (64 tokens x 576 B) and its
# DSA indexer page (64 tokens x (128 B keys + 4 B scale)), DEP4 and PP4 layer counts.
_TUNED_TILINGS = {
    64 * 576: HiCacheDirectTiling(WRITE_BACK_PAGE_LAYER, 256, 9),
    64 * 132: HiCacheDirectTiling(WRITE_BACK_LINES, 4, 2),
}

_LOAD_THREADS = 256


def _default_tiling(page_bytes: int) -> Optional[HiCacheDirectTiling]:
    if page_bytes <= 0 or page_bytes % 16 != 0:
        return None
    if page_bytes % (256 * 16) == 0:
        write_back = (WRITE_BACK_PAGE_LAYER, 256)
    elif page_bytes % 128 == 0:
        write_back = (WRITE_BACK_LINES, 4)
    else:
        return None
    # Fewest CTAs per page whose share is at most one 16 B vector per thread.
    page_vecs = page_bytes // 16
    chunks = -(-page_vecs // _LOAD_THREADS)
    while page_vecs % chunks != 0:
        chunks += 1
    return HiCacheDirectTiling(*write_back, chunks)


def hicache_direct_tiling(page_bytes: int) -> Optional[HiCacheDirectTiling]:
    """Tiling used for `page_bytes`-byte pages, or None if unsupported."""
    return _TUNED_TILINGS.get(page_bytes) or _default_tiling(page_bytes)


@cache_once
def _jit_hicache_direct_module(
    *, page_bytes: int, page_size: int, tiling: HiCacheDirectTiling
) -> Module:
    if is_hip_runtime():
        raise RuntimeError("HiCache direct_kernel transfers are CUDA-only")
    args = make_cpp_args(
        page_bytes,
        page_size,
        tiling.write_back_variant,
        tiling.write_back_param,
        tiling.load_chunks,
    )
    return load_jit(
        "hicache_direct",
        *args,
        cuda_files=["kvcacheio/hicache_direct.cuh"],
        cuda_wrappers=[
            ("write_back", f"&HiCacheDirectKernel<{args}>::write_back"),
            ("load", f"&HiCacheDirectKernel<{args}>::load"),
        ],
    )


def can_use_hicache_direct_kernel(*, page_bytes: int, page_size: int) -> bool:
    """Whether `page_bytes`-byte pages of `page_size` tokens can use these kernels.

    Compiles the module on first call, so call it at startup rather than on the
    first transfer.
    """
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    tiling = hicache_direct_tiling(page_bytes)
    if tiling is None:
        logger.warning(f"Unsupported {page_bytes = } for HiCache direct_kernel")
        return False
    try:
        _jit_hicache_direct_module(
            page_bytes=page_bytes, page_size=page_size, tiling=tiling
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load HiCache direct_kernel module: {e}")
        return False


def transfer_hicache_direct_write_back(
    *,
    host_pool: torch.Tensor,
    device_layer_ptrs: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    page_size: int,
    page_bytes: int,
    tiling: Optional[HiCacheDirectTiling] = None,
) -> None:
    """Back up every layer of whole pages from the device pools to the host pool.

    ``host_pool`` is the contiguous, pinned page_first_direct host buffer,
    ``[host_page, num_layers, ...]`` with ``page_bytes`` bytes per (page, layer).
    ``device_layer_ptrs`` is a uint64 CUDA tensor of ``num_layers`` device layer
    base addresses, in host layer order; each device layer is
    ``[device_page, page_bytes]`` bytes. ``host_indices`` / ``device_indices``
    are paired int64 CUDA token slots, whole pages of ``page_size``.
    Runs on the current stream.
    """
    module = _jit_hicache_direct_module(
        page_bytes=page_bytes,
        page_size=page_size,
        tiling=tiling or hicache_direct_tiling(page_bytes),
    )
    module.write_back(host_pool, device_layer_ptrs, host_indices, device_indices)


def transfer_hicache_direct_load(
    *,
    device_layer: torch.Tensor,
    host_pool: torch.Tensor,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    host_layer_id: int,
    page_size: int,
    page_bytes: int,
    tiling: Optional[HiCacheDirectTiling] = None,
) -> None:
    """Load host layer ``host_layer_id`` of whole pages into one device layer.

    ``device_layer`` is one contiguous device layer, ``[device_page, page_bytes]``
    bytes; the other arguments follow `transfer_hicache_direct_write_back`.
    Runs on the current stream.
    """
    module = _jit_hicache_direct_module(
        page_bytes=page_bytes,
        page_size=page_size,
        tiling=tiling or hicache_direct_tiling(page_bytes),
    )
    module.load(device_layer, host_pool, host_indices, device_indices, host_layer_id)
