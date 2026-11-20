"""Prefill admission when a cache service owns KV memory.

The service decides whether a request fits and how much of its prefix it
shares: ``acquire`` refuses, or grants a shorter chunk. This adder keeps only
the scheduler's compute budgets (prefill tokens per batch, chunk size).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.borrow.api import AcquireResult, AcquireStatus

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class BorrowPrefillAdder(PrefillAdder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = self.tree_cache.client
        self.memory_refused = False

    def budget_state(self) -> AddReqResult:
        if self.chunked_req_waits or self.memory_refused:
            return AddReqResult.NO_TOKEN
        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER
        if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
            return AddReqResult.OTHER
        return AddReqResult.CONTINUE

    def _chunk_budget(self) -> Optional[int]:
        if self.rem_chunk_tokens is None:
            return None
        budget = self.rem_chunk_tokens
        if self.chunked_req_limit is not None:
            budget = min(budget, self.chunked_req_limit)
        return budget

    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ) -> AddReqResult:
        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER
        if self.chunked_req_waits:
            return AddReqResult.NO_TOKEN
        input_len = len(req.full_untruncated_fill_ids)
        chunking = self.rem_chunk_tokens is not None
        # At most one request is mid-prefill: the one carried over from the last
        # pass or one this pass already cut short (by budget or by memory).
        chunked_exists = has_chunked_req or self.new_chunked_req is not None
        res = self.client.open(
            req,
            end=input_len,
            max_mut_len=self._chunk_budget(),
            min_mut_len=self.page_size if chunking and not chunked_exists else None,
            cut_align=truncation_align_size,
        )
        if res.status != AcquireStatus.GRANTED:
            self.memory_refused = True
            return AddReqResult.NO_TOKEN
        extend_len = res.total_len - res.shared_len
        is_chunked = res.total_len < input_len
        over_input_cap = (
            not chunking
            and self.can_run_list
            and self.ceil_paged_tokens(extend_len) >= self.rem_input_tokens
        )
        if (is_chunked and chunked_exists) or over_input_cap:
            self.client.release(req, publish_token_ids=None)
            return AddReqResult.OTHER
        self._commit(req, res, is_chunked=is_chunked)
        return self.budget_state()

    def add_chunked_req(self, req: Req) -> Optional[Req]:
        """The resumed chunked request's next chunk; returns it while unfinished.
        Refused, it waits and closes this pass."""
        input_len = len(req.full_untruncated_fill_ids)
        res = self.client.grow_chunk(
            req,
            end=input_len,
            max_mut_len=self._chunk_budget(),
            min_mut_len=self.page_size,
        )
        if res.status != AcquireStatus.GRANTED:
            self.chunked_req_waits = True
            return req
        req.extend_end = res.total_len
        self.can_run_list.append(req)
        self._charge(req, prefix_len=0, extend_len=res.total_len - req.prefix_len)
        return req if res.total_len < input_len else None

    def _commit(self, req: Req, res: AcquireResult, *, is_chunked: bool) -> None:
        req.prefix_len = res.shared_len
        req.extend_end = res.total_len
        req.host_hit_length = res.loaded_len
        req.host_loaded_length = res.loaded_len
        req.num_matched_prefix_tokens = res.shared_len
        self.can_run_list.append(req)
        if is_chunked:
            self.new_chunked_req = req
        self._charge(
            req, prefix_len=res.shared_len, extend_len=res.total_len - res.shared_len
        )
        if not req.retracted_stain:
            self.log_device_hit_tokens += res.shared_len - res.loaded_len
            self.log_host_hit_tokens += res.loaded_len

    def _charge(self, req: Req, *, prefix_len: int, extend_len: int) -> None:
        compute = (
            extend_len if self.exact_chunk_fill else self.ceil_paged_tokens(extend_len)
        )
        self.rem_input_tokens -= compute
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= compute
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_len
        if req.retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += extend_len

    def add_one_req_ignore_eos(self, req: Req, truncation_align_size: Optional[int]):
        return self.add_one_req(req, False, truncation_align_size)

    def preempt_to_schedule(self, req: Req) -> bool:
        # Retraction is the cache service's decision (Revoked leases).
        return False
