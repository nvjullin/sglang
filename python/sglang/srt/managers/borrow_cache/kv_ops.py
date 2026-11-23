"""KV row bookkeeping for requests whose slots a cache service lends.

The service hands out slot ids; the scheduler writes them into its request
table (``req_to_token``) and never frees a slot itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.borrow.api import StateAnchor, StateGrant
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
    for req in batch.reqs:
        if (state := client.take_state(req)) is not None:
            _bind_state(req, state, table=batch.req_to_token_pool, device=device)
    storage = kv_storage_of(batch.req_to_token_pool, batch.token_to_kv_pool_allocator)
    # Requests already hold their state slots, so this only binds rows and
    # writes the state slot ids into the table's mapping columns.
    rows = storage.acquire(batch.reqs)
    if rows is None:
        raise RuntimeError(
            "alloc_for_extend: no free request rows; lower --max-running-requests"
        )
    prefix_tensors, mut_slots, window_pairs = [], [], []
    for req, held, row in zip(batch.reqs, holds_row, rows):
        prefix, mut = client.take_pending(req)
        assert len(mut) == req.extend_len, (req.rid, len(mut), req.extend_len)
        if client.window is not None:
            prefix_w, mut_w = client.take_pending_windows(req)
            window_pairs.append((mut, mut_w))
            if prefix_w is not None:
                window_pairs.append((prefix, prefix_w))
        if held:
            assert prefix is None
            prefix_tensors.append(req_to_token[row, : req.prefix_len].to(torch.int64))
        else:
            assert len(prefix) == req.prefix_len, (req.rid, len(prefix), req.prefix_len)
            prefix_tensors.append(_to_device(prefix, device, pin))
        mut_slots.append(mut)
    out_cache_loc = _to_device(torch.cat(mut_slots), device, pin)
    if window_pairs:
        _write_window_map(batch.token_to_kv_pool_allocator, window_pairs, device, pin)

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
    pin = is_pin_memory_available(batch.device)
    slots, window_pairs = [], []
    for req in batch.reqs:
        prefix, mut = client.take_pending(req)
        assert prefix is None and len(mut) == 1, (req.rid, len(mut))
        slots.append(mut)
        if client.window is not None:
            window_pairs.append((mut, client.take_pending_windows(req)[1]))
    out_cache_loc = _to_device(torch.cat(slots), batch.device, pin)
    if window_pairs:
        _write_window_map(batch.token_to_kv_pool_allocator, window_pairs, batch.device, pin)
    batch.req_to_token_pool.write(
        (batch.req_pool_indices, batch.seq_lens), out_cache_loc.to(torch.int32)
    )
    for req in batch.reqs:
        req.kv.allocate_decode(1)
    return out_cache_loc


def _write_window_map(allocator, pairs, device, pin: bool) -> None:
    """Point each borrowed KV slot at its window slot in the allocator's
    full -> window table. Zero window ids (a shared position outside this
    request's window) are skipped: another request sharing the page may
    still read its entry."""
    full = torch.cat([f for f, _ in pairs])
    window = torch.cat([w for _, w in pairs])
    keep = window != 0
    full, window = full[keep], window[keep]
    if full.numel() == 0:
        return
    mapping = allocator.full_to_swa_index_mapping
    mapping[_to_device(full, device, pin)] = _to_device(window, device, pin)


def _window_release_point(frontier: int, *, window: int, page_size: int) -> int:
    """Positions below this are out of every later forward's window."""
    keep = -(-max(window, page_size) // page_size) * page_size
    return max(0, frontier - keep) // page_size * page_size


def retire_windows_for_decode(batch: ScheduleBatch) -> None:
    """The scheduler's window retirement rule (``ScheduleBatch.maybe_evict_swa``)
    as window releases: every ``interval`` positions past the window, once the
    request's prefill forward is no longer in flight."""
    client = batch.tree_cache.client
    window, interval = client.window
    ps = client.page_size
    for req in batch.reqs:
        frontier = req.seqlen - 1
        if req.decode_batch_idx < 1 or frontier - window < client.window_floor(req) + interval:
            continue
        client.release_window(
            req, below=_window_release_point(frontier, window=window, page_size=ps)
        )


def grow_running_batch(
    batch: ScheduleBatch, token_per_req: int, *, revoke: bool = True
) -> Optional[list[Req]]:
    """Borrow the next decode step's positions; returns the requests the cache
    revoked, which the caller must retract (None: refused, see `client.grow`)."""
    client = batch.tree_cache.client
    if client.window is not None:
        retire_windows_for_decode(batch)
    if batch.spec_algorithm.is_none():
        ends = (batch.seq_lens_cpu + token_per_req).tolist()
    else:
        # The same ends eagle_prepare_for_decode allocates up to.
        _, ends, _ = page_aligned_decode_alloc_lens(
            batch.reqs,
            reserve=get_alloc_reserve_per_decode(),
            page_size=batch.token_to_kv_pool_allocator.page_size,
        )
    return client.grow(batch.reqs, ends, revoke=revoke)


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


def _bind_state(req: Req, state: StateGrant, *, table, device) -> None:
    """Hand the request the state slots its lease borrows, as
    `StateStorage.acquire` hands out its own (`_hand_out_tracking`). The
    service already started the working slot, so nothing is cleared or copied
    on the forward stream."""
    assert not req.kv.holds_mamba, f"{req.rid} already holds a state slot"
    size = table.mamba_ping_pong_track_buffer_size
    tracking = list(state.slots[1:])
    assert len(tracking) == table.mamba_initial_tracking_slots == size
    slots = torch.tensor(state.slots, dtype=torch.int64).to(device, non_blocking=True)
    kv = req.kv
    kv.mamba_pool_idx = slots[0]
    kv.mamba_ping_pong_track_buffer = slots[1:]
    kv.mamba_ping_pong_empty = [False] * size
    kv.mamba_prev_track_seqlen = None
    kv.mamba_last_track_seqlen = None
    kv.mamba_next_track_idx = 0
    kv.mamba_last_track_idx = table.get_mamba_ping_pong_other_idx(0)
    kv.mamba_cow_src_index = None
    kv.mamba_needs_clear = False


def _unbind_state(req: Req) -> None:
    """Forget the lease's state slots; the service reclaims them with the lease."""
    kv = req.kv
    kv.mamba_pool_idx = None
    kv.mamba_ping_pong_track_buffer = None
    kv.mamba_ping_pong_empty = None
    kv.mamba_next_track_idx = None
    kv.mamba_last_track_idx = None
    kv.mamba_last_track_seqlen = None
    kv.mamba_prev_track_seqlen = None
    kv.mamba_cow_src_index = None
    kv.mamba_needs_clear = False


def _state_anchor(req: Req, published_len: int) -> Optional[StateAnchor]:
    """The latest tracked checkpoint, if it lies within the published tokens.

    Only the most recent tracking slot is offered: the other ping-pong slot may
    be the target of a forward still in flight.
    """
    if not req.kv.holds_mamba:
        return None
    length = req.kv.mamba_last_track_seqlen
    if length is None or length == 0 or length > published_len:
        return None
    return StateAnchor(length=length, slot_index=1 + req.kv.mamba_last_track_idx)


def checkpoint_kv_cache(req: Req, tree_cache: BorrowTreeFacade) -> None:
    """Publish what a running request computed so far (a chunk end)."""
    client = tree_cache.client
    if not req.skip_radix_cache_insert:
        token_ids = req.full_untruncated_fill_ids[: req.extend_end]
        client.publish(req, token_ids, state=_state_anchor(req, len(token_ids)))
    if client.window is not None:
        # As the old tree's checkpoint does: the next extend starts at
        # extend_end and reads one window back.
        window, _ = client.window
        client.release_window(
            req,
            below=_window_release_point(
                req.extend_end - 1, window=window, page_size=client.page_size
            ),
        )
    req.prefix_len = req.extend_end


def release_kv_cache(req: Req, tree_cache: BorrowTreeFacade, *, checkpoint: bool) -> None:
    """End the request's lease (publishing its KV if ``checkpoint``) and give its
    row back."""
    if not req.kv.holds_kv:
        tree_cache.client.release(req, publish_token_ids=None)
        return
    publish = state = None
    if checkpoint and not req.skip_radix_cache_insert:
        req.refresh_fill_ids()
        publish = req.full_untruncated_fill_ids[: req.owned_kv_len()]
        state = _state_anchor(req, len(publish))
    tree_cache.client.release(req, publish_token_ids=publish, state=state)
    kv_storage_of(
        tree_cache.req_to_token_pool, tree_cache.token_to_kv_pool_allocator
    ).release_row(req)
    req.kv.mark_kv_released()
    if req.kv.holds_mamba:
        _unbind_state(req)
