"""Scheduler steps that change when a cache service lends the KV slots."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.managers.borrow_cache import kv_ops
from sglang.srt.managers.req_types import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.borrow_cache.client import BorrowCacheClient
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )

logger = logging.getLogger(__name__)


def grow_decode_or_retract(
    batch: ScheduleBatch, *, request_retirer: RequestRetirer, revoke: bool = True
) -> Optional[list[Req]]:
    """Borrow one more position per running request. The service may revoke
    leases to make room: those requests leave the batch and are returned for
    requeueing (or aborted when they cannot be recomputed). With
    ``revoke=False`` it refuses instead, and this returns None."""
    revoked = kv_ops.grow_running_batch(batch, token_per_req=1, revoke=revoke)
    if not revoked:
        return revoked
    revoked_ids = {id(r) for r in revoked}
    requeue = []
    for req in revoked:
        # The service already reclaimed the slots; this gives back the row.
        resumable = request_retirer.retract(req, offload_kv=False)
        req.reset_for_retract()
        if resumable:
            requeue.append(req)
            continue
        reason = FINISH_ABORT(
            "Retracted under KV memory pressure and cannot be recomputed. Retry later.",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
        req.request_finish(reason)
        request_retirer.abort_retracted(req, reason=reason)
    batch.filter_batch(
        keep_indices=[i for i, r in enumerate(batch.reqs) if id(r) not in revoked_ids]
    )
    logger.warning(
        "KV cache service revoked %d running requests (%d requeued).",
        len(revoked),
        len(requeue),
    )
    return requeue


def begin_forward(client: BorrowCacheClient, batch: ScheduleBatch, *, stream: Any) -> None:
    """Start the batch's in-stream loads; the forward on ``stream`` waits for them.

    The whole load completes before the forward starts, so a forward replayed
    from a CUDA graph (which skips per-layer hooks) is safe too.
    """
    fence = client.begin_batch(batch.reqs)
    if fence is not None:
        stream.wait_event(fence.final_event)
        # Marks the batch as loading (PD prefill then skips its early prefix send);
        # no per-layer hook is registered, so the worker's consumer stays unarmed.
        batch.hicache_consumer_index = fence.consumer_index


def end_forward(client: BorrowCacheClient, *, stream: Any) -> None:
    done = torch.cuda.Event()
    done.record(stream)
    client.end_batch(done)
