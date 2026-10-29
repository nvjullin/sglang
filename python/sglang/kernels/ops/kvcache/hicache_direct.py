"""Page-granular HiCache transfers for the ``page_first_direct`` host layout.

These back ``--hicache-io-backend direct_kernel``: page-row copies
(``kvcacheio/page_row.cuh``) that read the transfer indices on the GPU and access
the pinned host pool in place. The ``direct`` backend instead syncs the indices to
the CPU and builds copy-engine descriptors there, submitted as one
cudaMemcpyBatchAsync when the runtime has it and as per-page copies otherwise.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime
from sglang.kernels.ops.kvcache.page_row import (
    page_row_host_link_coherent,
    transfer_page_rows,
    warmup_page_row_transfer,
)

logger = logging.getLogger(__name__)


class HiCacheDirectTables(NamedTuple):
    """Page-row pointer tables for one page_first_direct host pool.

    ``host[l]`` is the mapped address of host layer ``l`` in host page 0 and
    ``device[l]`` the base of the device layer stored as host layer ``l``; both are
    uint64 CUDA ``[num_layers]``. A page is ``row_bytes`` on both sides; host pages
    are ``host_stride`` apart.
    """

    host: torch.Tensor
    device: torch.Tensor
    host_stride: int
    row_bytes: int

    def layer(self, host_layer: int, device_layer: int) -> HiCacheDirectTables:
        return self._replace(
            host=self.host[host_layer : host_layer + 1],
            device=self.device[device_layer : device_layer + 1],
        )


def make_hicache_direct_tables(
    host_pool: torch.Tensor, device_layer_ptrs: torch.Tensor
) -> HiCacheDirectTables:
    """Tables for a contiguous ``[host_page, num_layers, ...]`` host pool whose
    device layers are ``device_layer_ptrs`` (uint64 CUDA, host layer order)."""
    num_layers = host_pool.shape[1]
    row_bytes = host_pool[0, 0].nbytes
    assert host_pool.is_contiguous() and device_layer_ptrs.numel() == num_layers
    base = host_pool.data_ptr()
    host = torch.tensor(
        [base + layer * row_bytes for layer in range(num_layers)],
        dtype=torch.uint64,
        device=device_layer_ptrs.device,
    )
    return HiCacheDirectTables(
        host, device_layer_ptrs, num_layers * row_bytes, row_bytes
    )


def can_use_hicache_direct_kernel(*, page_bytes: int) -> bool:
    """Whether `page_bytes`-byte pages can use these transfers.

    Compiles the kernels on first call, so call it at startup rather than on the
    first transfer.
    """
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    if page_bytes <= 0 or page_bytes % 128 != 0:
        logger.warning(f"Unsupported {page_bytes = } for HiCache direct_kernel")
        return False
    try:
        warmup_page_row_transfer()
    except Exception as e:
        logger.warning(f"Failed to load HiCache direct_kernel module: {e}")
        return False
    device_id = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_id)[
        0
    ] < 10 and not page_row_host_link_coherent(device_id):
        _warn_pre_blackwell_pcie(device_id)
    return True


@cache_once
def _warn_pre_blackwell_pcie(device_id: int) -> None:
    logger.warning(
        f"HiCache direct_kernel on GPU {device_id}: SM copies to host memory over "
        "PCIe on pre-Blackwell GPUs are expected to be slower than the copy engine; "
        "consider --hicache-io-backend direct."
    )


def transfer_hicache_direct(
    tables: HiCacheDirectTables,
    host_indices: torch.Tensor,
    device_indices: torch.Tensor,
    *,
    page_size: int,
    load: bool,
) -> None:
    """Copy whole pages of every layer in ``tables`` on the current stream.

    ``host_indices`` / ``device_indices`` are paired int64 CUDA token slots, whole
    pages of ``page_size``. Pass ``tables.layer(...)`` to move a single layer.
    """
    transfer_page_rows(
        tables.host,
        tables.device,
        host_indices,
        device_indices,
        host_stride=tables.host_stride,
        device_stride=tables.row_bytes,
        row_bytes=tables.row_bytes,
        load=load,
        index_stride=page_size,
        # Uncapped: caps of 152+ CTAs measured no faster on GB300. Revisit if these
        # copies come to overlap a forward, as under write_through.
        block_quota=0,
    )
