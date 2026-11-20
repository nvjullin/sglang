"""KV row bookkeeping for requests whose slots a cache service lends.

The service hands out slot ids; the scheduler writes them into its request
table (``req_to_token``) and never frees a slot itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool_func,
    write_cache_indices,
)
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.utils import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.borrow_cache.facade import BorrowTreeFacade
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


def _to_device(t: torch.Tensor, device, pin: bool) -> torch.Tensor:
    if pin:
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def alloc_for_extend(batch: ScheduleBatch):
    """Write each request's granted slots into its row; see
    `sglang.srt.mem_cache.allocation.alloc_for_extend` for the contract."""
    client = batch.tree_cache.client
    device = batch.device
    pin = is_pin_memory_available(device)
    req_to_token = batch.req_to_token_pool.req_to_token
    holds_row = [r.kv.holds_kv for r in batch.reqs]
    storage = kv_storage_of(batch.req_to_token_pool, batch.token_to_kv_pool_allocator)
    rows = storage.acquire(batch.reqs)
    if rows is None:
        raise RuntimeError(
            "alloc_for_extend: no free request rows; lower --max-running-requests"
        )
    prefix_tensors, mut_slots = [], []
    for req, held, row in zip(batch.reqs, holds_row, rows):
        prefix, mut = client.take_pending(req)
        assert len(mut) == req.extend_len, (req.rid, len(mut), req.extend_len)
        if held:
            assert prefix is None
            prefix_tensors.append(req_to_token[row, : req.prefix_len].to(torch.int64))
        else:
            assert len(prefix) == req.prefix_len, (req.rid, len(prefix), req.prefix_len)
            prefix_tensors.append(_to_device(prefix, device, pin))
        mut_slots.append(mut)
    out_cache_loc = _to_device(torch.cat(mut_slots), device, pin)

    rows_cpu = torch.tensor(rows, dtype=torch.int64, pin_memory=pin)
    prefix_lens_cpu = torch.tensor(batch.prefix_lens, dtype=torch.int64, pin_memory=pin)
    extend_lens_cpu = torch.tensor(batch.extend_lens, dtype=torch.int64, pin_memory=pin)
    rows_device = rows_cpu.to(device, non_blocking=True)
    write_cache_indices(
        out_cache_loc,
        rows_device,
        rows_cpu,
        prefix_lens_cpu.to(device, non_blocking=True),
        prefix_lens_cpu,
        batch.seq_lens,
        batch.seq_lens_cpu,
        extend_lens_cpu.to(device, non_blocking=True),
        extend_lens_cpu,
        prefix_tensors,
        batch.req_to_token_pool,
    )
    for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
        req.kv.set_row_len(seq_len)
    return out_cache_loc, rows_device, rows_cpu


def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:
    """Write the slots `grow_running_batch` was granted for this step."""
    assert token_per_req == 1, "multi-token decode steps go through spec decoding"
    client = batch.tree_cache.client
    slots = []
    for req in batch.reqs:
        prefix, mut = client.take_pending(req)
        assert prefix is None and len(mut) == 1, (req.rid, len(mut))
        slots.append(mut)
    out_cache_loc = _to_device(
        torch.cat(slots), batch.device, is_pin_memory_available(batch.device)
    )
    batch.req_to_token_pool.write(
        (batch.req_pool_indices, batch.seq_lens), out_cache_loc.to(torch.int32)
    )
    for req in batch.reqs:
        req.kv.allocate_decode(1)
    return out_cache_loc


def grow_running_batch(batch: ScheduleBatch, token_per_req: int) -> list[Req]:
    """Borrow the next decode step's positions; returns the requests the cache
    revoked, which the caller must retract."""
    client = batch.tree_cache.client
    if batch.spec_algorithm.is_none():
        ends = (batch.seq_lens_cpu + token_per_req).tolist()
    else:
        # The same ends eagle_prepare_for_decode allocates up to.
        _, ends, _ = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=get_alloc_reserve_per_decode(),
            page_size=batch.token_to_kv_pool_allocator.page_size,
        )
    return client.grow(batch.reqs, ends)


def alloc_for_spec_decode(
    tree_cache: BorrowTreeFacade,
    req_to_token_pool,
    *,
    reqs: list[Req],
    req_pool_indices: torch.Tensor,
    cur_kv_lens: torch.Tensor,
    cur_kv_lens_cpu: torch.Tensor,
    nxt_kv_lens: torch.Tensor,
    nxt_kv_lens_cpu: torch.Tensor,
    num_needed_tokens: int,
    batch: ScheduleBatch,
) -> None:
    """Write the slots `grow_running_batch` was granted for this spec step;
    see `sglang.srt.mem_cache.allocation.alloc_for_spec_decode`."""
    client = tree_cache.client
    nxt_list = nxt_kv_lens_cpu.tolist()
    slots = []
    for req, cur, nxt in zip(reqs, cur_kv_lens_cpu.tolist(), nxt_list, strict=True):
        prefix, mut = client.take_pending(req)
        assert prefix is None and len(mut) == nxt - cur, (req.rid, len(mut), cur, nxt)
        slots.append(mut)
    if num_needed_tokens > 0:
        out_cache_loc = _to_device(
            torch.cat(slots), batch.device, is_pin_memory_available(batch.device)
        )
        assign_req_to_token_pool_func(
            req_pool_indices,
            req_to_token_pool.req_to_token,
            cur_kv_lens,
            nxt_kv_lens,
            out_cache_loc,
            len(reqs),
        )
    for req, nxt in zip(reqs, nxt_list):
        req.kv.allocate_spec_decode(nxt)


def checkpoint_kv_cache(req: Req, tree_cache: BorrowTreeFacade) -> None:
    """Publish what a running request computed so far (a chunk end)."""
    if not req.skip_radix_cache_insert:
        tree_cache.client.publish(req, req.full_untruncated_fill_ids[: req.extend_end])
    req.prefix_len = req.extend_end


def release_kv_cache(req: Req, tree_cache: BorrowTreeFacade, *, checkpoint: bool) -> None:
    """End the request's lease (publishing its KV if ``checkpoint``) and give its
    row back."""
    if not req.kv.holds_kv:
        tree_cache.client.release(req, publish_token_ids=None)
        return
    publish = None
    if checkpoint and not req.skip_radix_cache_insert:
        req.refresh_fill_ids()
        publish = req.full_untruncated_fill_ids[: req.owned_kv_len()]
    tree_cache.client.release(req, publish_token_ids=publish)
    kv_storage_of(
        tree_cache.req_to_token_pool, tree_cache.token_to_kv_pool_allocator
    ).release_row(req)
    req.kv.mark_kv_released()
