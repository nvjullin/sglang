"""The scheduler's side of the borrow API: request <-> lease bookkeeping.

Everything here runs on the scheduler thread. The service it talks to may run
anywhere; nothing here reads service state except through the API.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS
from sglang.srt.mem_cache.borrow.api import (
    AcquireReq,
    AcquireResult,
    AcquireStatus,
    CacheKey,
    Fence,
    KVCacheService,
    LeaseAttrs,
    Revoked,
    StateAnchor,
    StateGrant,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class LeaseView:
    """What the scheduler keeps of one lease."""

    __slots__ = (
        "lease",
        "prefix_slots",
        "pending_slots",
        "pending_state",
        "prefix_window_slots",
        "pending_window_slots",
        "window_floor",
    )

    def __init__(
        self,
        lease: int,
        prefix_slots: Optional[torch.Tensor],
        state: Optional[StateGrant],
        prefix_window_slots: Optional[torch.Tensor] = None,
    ):
        self.lease = lease
        # Slot ids of the shared prefix, until written into the request's row.
        self.prefix_slots = prefix_slots
        # Slot ids granted for positions not yet written into the row.
        self.pending_slots: list[torch.Tensor] = []
        # Recurrent-state slots, until bound to the request.
        self.pending_state = state
        # Window slot ids aligned with the pending KV slot ids (0: none).
        self.prefix_window_slots = prefix_window_slots
        self.pending_window_slots: list[torch.Tensor] = []
        # Window borrows below this position are released.
        self.window_floor = 0


class BorrowCacheClient:
    def __init__(
        self,
        *,
        service: KVCacheService,
        page_size: int,
        window: Optional[tuple[int, int]] = None,
        reprefill_tail: int = 0,
    ):
        self.service = service
        self.page_size = page_size
        # (window tokens, retirement interval) when KV has a window pool.
        self.window = window
        # Tokens at the end of a prompt never shared: decoder SWA bounded replay
        # writes late-layer window KV only for an extend's tail rows.
        self.reprefill_tail = reprefill_tail
        self._views: dict[str, LeaseView] = {}
        self._batch_id = 0
        # Arbitrary; often enough to follow a test run in the server log.
        self._stats_log_interval = 30.0
        self._last_stats_log = time.monotonic()

    # ===== Request -> API types =====

    def key_of(self, req: Req) -> CacheKey:
        tokens = req.full_untruncated_fill_ids
        limit = req._compute_max_prefix_len(len(tokens))
        if self.reprefill_tail:
            limit = min(limit, max(0, len(tokens) - self.reprefill_tail))
        if req.positional_embed_overrides is not None:
            limit = 0
        return CacheKey(
            token_ids=tokens[:limit], namespace=(req.extra_key, req.cache_salt)
        )

    @staticmethod
    def attrs_of(req: Req) -> LeaseAttrs:
        remaining = max(req.sampling_params.max_new_tokens - len(req.output_ids), 0)
        return LeaseAttrs(
            priority=req.priority or 0,
            prompt_len=len(req.full_untruncated_fill_ids),
            max_new_tokens=min(remaining, CLIP_MAX_NEW_TOKENS),
        )

    def view(self, req: Req) -> Optional[LeaseView]:
        return self._views.get(req.rid)

    # ===== Verbs =====

    def open(
        self,
        req: Req,
        *,
        end: int,
        max_mut_len: Optional[int],
        min_mut_len: Optional[int],
        cut_align: Optional[int] = None,
    ) -> AcquireResult:
        assert req.rid not in self._views, f"{req.rid} already holds a lease"
        res = self._acquire_one(
            AcquireReq(
                end=end,
                key=self.key_of(req),
                max_mut_len=max_mut_len,
                min_mut_len=min_mut_len,
                cut_align=cut_align,
                attrs=self.attrs_of(req),
            )
        )
        if res.status == AcquireStatus.GRANTED:
            view = LeaseView(
                res.lease, res.shared_slots, res.state, res.shared_window_slots
            )
            view.pending_slots.append(res.mut_slots)
            if self.window is not None:
                view.pending_window_slots.append(res.mut_window_slots)
            self._views[req.rid] = view
            if res.state is not None:
                req.mamba_branching_seqlen = res.state.branch_len
        return res

    def grow_chunk(
        self,
        req: Req,
        *,
        end: int,
        max_mut_len: Optional[int],
        min_mut_len: Optional[int],
        cut_align: Optional[int] = None,
    ) -> AcquireResult:
        view = self._views[req.rid]
        res = self._acquire_one(
            AcquireReq(
                end=end,
                lease=view.lease,
                max_mut_len=max_mut_len,
                min_mut_len=min_mut_len,
                cut_align=cut_align,
                attrs=self.attrs_of(req),
            )
        )
        if res.status == AcquireStatus.GRANTED:
            view.pending_slots.append(res.mut_slots)
            if self.window is not None:
                view.pending_window_slots.append(res.mut_window_slots)
        return res

    def grow(self, reqs: Sequence[Req], ends: Sequence[int]) -> list[Req]:
        """Grow every request's lease to its end; returns the revoked requests,
        whose leases are already gone."""
        views = [self._views[r.rid] for r in reqs]
        batch = self.service.acquire(
            [
                AcquireReq(end=e, lease=v.lease, attrs=self.attrs_of(r))
                for r, v, e in zip(reqs, views, ends)
            ]
        )
        revoked = []
        for req, view, res in zip(reqs, views, batch.results):
            if res.status == AcquireStatus.REVOKED:
                del self._views[req.rid]
                revoked.append(req)
            else:
                assert res.status == AcquireStatus.GRANTED, (
                    res,
                    req.rid,
                    len(req.full_untruncated_fill_ids),
                    len(req.origin_input_ids),
                    len(req.output_ids),
                    req.seqlen,
                    req.retraction_count,
                )
                view.pending_slots.append(res.mut_slots)
                if self.window is not None:
                    view.pending_window_slots.append(res.mut_window_slots)
        self._check_events(batch.events, expected={v.lease for v in views})
        return revoked

    def _acquire_one(self, req: AcquireReq) -> AcquireResult:
        batch = self.service.acquire([req])
        self._check_events(batch.events, expected=set())
        return batch.results[0]

    @staticmethod
    def _check_events(events: list, *, expected: set) -> None:
        for ev in events:
            if not (isinstance(ev, Revoked) and ev.lease in expected):
                raise NotImplementedError(f"unexpected cache service event {ev!r}")

    def take_pending(self, req: Req) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """(prefix slots if not yet written, slots of newly granted positions)."""
        view = self._views[req.rid]
        prefix, view.prefix_slots = view.prefix_slots, None
        pending = view.pending_slots
        view.pending_slots = []
        mut = pending[0] if len(pending) == 1 else torch.cat(pending)
        return prefix, mut

    def take_pending_windows(
        self, req: Req
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """Window slot ids aligned with what `take_pending` just returned."""
        view = self._views[req.rid]
        prefix, view.prefix_window_slots = view.prefix_window_slots, None
        pending = view.pending_window_slots
        view.pending_window_slots = []
        mut = pending[0] if len(pending) == 1 else torch.cat(pending)
        return prefix, mut

    def release_window(self, req: Req, *, below: int) -> None:
        """Drop the request's window borrows of positions < ``below``."""
        view = self._views[req.rid]
        if below > view.window_floor:
            view.window_floor = below
            self.service.release(view.lease, window_below=below, from_len=None)

    def window_floor(self, req: Req) -> int:
        return self._views[req.rid].window_floor

    def take_state(self, req: Req) -> Optional[StateGrant]:
        """The state slots granted with the lease, once, to bind to the request."""
        view = self._views[req.rid]
        state, view.pending_state = view.pending_state, None
        return state

    def publish(
        self, req: Req, token_ids: Sequence[int], *, state: Optional[StateAnchor] = None
    ) -> None:
        self.service.release_mut(self._views[req.rid].lease, token_ids, state=state)

    def rollback(self, req: Req, from_len: int) -> None:
        self.service.release(self._views[req.rid].lease, from_len=from_len)

    def release(
        self,
        req: Req,
        *,
        publish_token_ids: Optional[Sequence[int]],
        state: Optional[StateAnchor] = None,
    ) -> bool:
        """End the request's lease; False if it had none (e.g. revoked)."""
        view = self._views.pop(req.rid, None)
        if view is None:
            return False
        self.service.release(
            view.lease, publish_token_ids=publish_token_ids, state=state
        )
        return True

    # ===== Batch fences =====

    def begin_batch(self, reqs: Sequence[Req]) -> Optional[Fence]:
        self._batch_id += 1
        leases = [v.lease for r in reqs if (v := self._views.get(r.rid)) is not None]
        return self.service.begin_batch(self._batch_id, leases)

    def end_batch(self, done_event: Any) -> None:
        self._check_events(self.service.end_batch(self._batch_id, done_event), expected=set())

    def poll(self) -> None:
        self._check_events(self.service.poll(), expected=set())
        now = time.monotonic()
        if now - self._last_stats_log >= self._stats_log_interval:
            # Read-only, so ranks need not issue it in lockstep.
            self._last_stats_log = now
            logger.info("Cache service stats: %s", self.service.stats())

    def close(self) -> None:
        self.service.close()
