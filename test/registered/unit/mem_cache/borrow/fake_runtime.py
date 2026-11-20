"""A CPU stand-in for the data plane and the runtime, for borrow API tests.

KV content of position ``p`` of a sequence is a function of the namespace and
tokens ``[0, p]``, like real KV. The fake data plane copies contents between a
device array and a host array, so tests can check that every slot a lease
borrows holds the KV its position needs.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import numpy as np

from sglang.srt.mem_cache.borrow.api import (
    AcquireReq,
    AcquireStatus,
    CacheKey,
    Fence,
    KVCacheService,
    LeaseAttrs,
)
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.transfer import DataPlane, TransferOp


def kv_content(namespace: tuple, tokens: Sequence[int], pos: int) -> int:
    return hash((namespace, tuple(tokens[: pos + 1]))) & 0x7FFFFFFFFFFF


class FakeOp(TransferOp):
    def __init__(self, launched: bool):
        self.launched = launched
        self.waits = 0

    def wait(self) -> None:
        assert self.launched, "waiting on a copy that never launched"
        self.waits += 1


class FakeDataPlane(DataPlane):
    def __init__(self, *, page_size: int, host_num_pages: int, device_kv: np.ndarray):
        self.page_size = page_size
        self.host_num_pages = host_num_pages
        self._host = PageAllocator(num_pages=host_num_pages)
        self.device_kv = device_kv
        self.host_kv = np.full(host_num_pages * page_size, -1, dtype=np.int64)
        self._queued: list = []
        self.launches = 0

    def _slots(self, pages: Sequence[int]) -> np.ndarray:
        ps = self.page_size
        return (np.asarray(pages, dtype=np.int64)[:, None] * ps + np.arange(ps)).ravel()

    def host_num_pages_free(self) -> int:
        return self._host.num_free

    def alloc_host(self, num_pages: int) -> Optional[list[int]]:
        return self._host.alloc(num_pages)

    def free_host(self, pages: Sequence[int]) -> None:
        self._host.free(pages)

    def backup(self, device_pages, host_pages, *, after: Any) -> TransferOp:
        self.host_kv[self._slots(host_pages)] = self.device_kv[self._slots(device_pages)]
        return FakeOp(launched=True)

    def queue_load(self, host_pages, device_pages) -> TransferOp:
        op = FakeOp(launched=False)
        self._queued.append((list(host_pages), list(device_pages), op))
        return op

    def launch_loads(self, *, after: Any) -> Optional[Fence]:
        if not self._queued:
            return None
        for host_pages, device_pages, op in self._queued:
            self.device_kv[self._slots(device_pages)] = self.host_kv[self._slots(host_pages)]
            op.launched = True
        self._queued = []
        self.launches += 1
        return Fence(consumer_index=self.launches)

    def layer_hook(self) -> Any:
        return None

    def close(self) -> None:
        pass


class FakeReq:
    def __init__(self, rid: int, prompt: list[int], max_new: int, namespace: tuple = ()):
        self.rid = rid
        self.prompt = prompt
        self.max_new = max_new
        self.namespace = namespace
        self.out: list[int] = []
        self.lease: Optional[int] = None
        self.slots: list[int] = []
        self.computed = 0

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out


class FakeRuntime:
    """Drives a service like a scheduler: open, write KV, publish, decode, end."""

    def __init__(self, service: KVCacheService, device_kv: np.ndarray, *, seed: int = 0):
        self.svc = service
        self.kv = device_kv
        self.rng = random.Random(seed)
        self.batch_id = 0
        self.revoked: list[FakeReq] = []

    def attrs(self, req: FakeReq) -> LeaseAttrs:
        return LeaseAttrs(prompt_len=len(req.prompt), max_new_tokens=req.max_new)

    def open(self, req: FakeReq, *, max_mut_len: Optional[int] = None, min_mut_len=None) -> bool:
        key = CacheKey(token_ids=req.prompt[: len(req.prompt) - 1], namespace=req.namespace)
        res = self.svc.acquire(
            [
                AcquireReq(
                    end=len(req.prompt),
                    key=key,
                    max_mut_len=max_mut_len,
                    min_mut_len=min_mut_len,
                    attrs=self.attrs(req),
                )
            ]
        ).results[0]
        if res.status != AcquireStatus.GRANTED:
            return False
        req.lease = res.lease
        req.slots = res.shared_slots.tolist() + res.mut_slots.tolist()
        req.computed = res.shared_len
        assert len(req.slots) == res.total_len
        return True

    def grow_chunk(self, req: FakeReq, *, max_mut_len: int) -> bool:
        res = self.svc.acquire(
            [
                AcquireReq(
                    end=len(req.prompt),
                    lease=req.lease,
                    max_mut_len=max_mut_len,
                    min_mut_len=1,
                    attrs=self.attrs(req),
                )
            ]
        ).results[0]
        if res.status != AcquireStatus.GRANTED:
            return False
        req.slots += res.mut_slots.tolist()
        assert len(req.slots) == res.total_len
        return True

    def forward(self, reqs: Sequence[FakeReq]) -> Optional[Fence]:
        """One batch: wait for loads, check shared KV, write the computed KV."""
        self.batch_id += 1
        fence = self.svc.begin_batch(self.batch_id, [r.lease for r in reqs])
        for r in reqs:
            toks = r.tokens
            for p in range(r.computed):
                assert self.kv[r.slots[p]] == kv_content(r.namespace, toks, p), (
                    f"rid={r.rid} pos={p} holds wrong KV"
                )
            for p in range(r.computed, len(r.slots)):
                self.kv[r.slots[p]] = kv_content(r.namespace, toks, p)
            r.computed = len(r.slots)
        self.svc.end_batch(self.batch_id, done_event=None)
        return fence

    def publish(self, req: FakeReq) -> None:
        self.svc.release_mut(req.lease, req.tokens[: req.computed])

    def decode(self, reqs: Sequence[FakeReq], *, k: int = 1) -> list[FakeReq]:
        """Grow every request by k positions; returns the survivors."""
        batch = self.svc.acquire(
            [
                AcquireReq(end=len(r.slots) + k, lease=r.lease, attrs=self.attrs(r))
                for r in reqs
            ]
        )
        alive = []
        for r, res in zip(reqs, batch.results):
            if res.status == AcquireStatus.REVOKED:
                r.lease, r.slots, r.computed = None, [], 0
                self.revoked.append(r)
                continue
            assert res.status == AcquireStatus.GRANTED
            r.slots += res.mut_slots.tolist()
            r.out += [self.rng.randrange(1, 50) for _ in range(k)]
            alive.append(r)
        return alive

    def finish(self, req: FakeReq, *, publish: bool = True) -> None:
        self.svc.release(
            req.lease,
            publish_token_ids=req.tokens[: req.computed] if publish else None,
        )
        req.lease = None
