"""A CPU stand-in for the data plane and the runtime, for borrow API tests.

KV content of position ``p`` of a sequence is a function of the namespace and
tokens ``[0, p]``, like real KV. The fake data plane copies contents between a
device array and a host array, so tests can check that every slot a lease
borrows holds the KV its position needs. Recurrent state after ``n`` tokens is
likewise a function of the namespace and tokens ``[0, n)``.
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
    StateAnchor,
)
from sglang.srt.mem_cache.borrow.hicache.page_allocator import PageAllocator
from sglang.srt.mem_cache.borrow.hicache.transfer import (
    DataPlane,
    StatePlane,
    TransferOp,
)


def kv_content(namespace: tuple, tokens: Sequence[int], pos: int) -> int:
    return hash((namespace, tuple(tokens[: pos + 1]))) & 0x7FFFFFFFFFFF


def state_content(namespace: tuple, tokens: Sequence[int], length: int) -> int:
    if length == 0:
        return 0
    return hash(("state", namespace, tuple(tokens[:length]))) & 0x7FFFFFFFFFFF


class FakeOp(TransferOp):
    def __init__(self, launched: bool):
        self.launched = launched
        self.waits = 0

    def wait(self) -> None:
        assert self.launched, "waiting on a copy that never launched"
        self.waits += 1


class FakeDataPlane(DataPlane):
    def __init__(
        self,
        *,
        page_size: int,
        host_num_pages: int,
        device_kv: np.ndarray,
        device_window_kv: Optional[np.ndarray] = None,
        host_num_window_pages: int = 0,
    ):
        self.page_size = page_size
        self.host_num_pages = host_num_pages
        self._host = PageAllocator(num_pages=host_num_pages)
        self.device_kv = device_kv
        self.host_kv = np.full(host_num_pages * page_size, -1, dtype=np.int64)
        self.device_window_kv = device_window_kv
        self.host_num_window_pages = host_num_window_pages
        self._host_windows = PageAllocator(num_pages=host_num_window_pages)
        self.host_window_kv = np.full(max(host_num_window_pages, 1) * page_size, -1, dtype=np.int64)
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

    def host_window_pages_free(self) -> int:
        return self._host_windows.num_free

    def alloc_host_windows(self, num_pages: int) -> Optional[list[int]]:
        return self._host_windows.alloc(num_pages)

    def free_host_windows(self, pages: Sequence[int]) -> None:
        self._host_windows.free(pages)

    def backup(self, device_pages, host_pages, *, after: Any, windows=None) -> TransferOp:
        self.host_kv[self._slots(host_pages)] = self.device_kv[self._slots(device_pages)]
        if windows is not None and windows[0]:
            dev_w, host_w = windows
            self.host_window_kv[self._slots(host_w)] = self.device_window_kv[self._slots(dev_w)]
        return FakeOp(launched=True)

    def queue_load(self, host_pages, device_pages, windows=None) -> TransferOp:
        op = FakeOp(launched=False)
        self._queued.append((list(host_pages), list(device_pages), windows, op))
        return op

    def launch_loads(self, *, after: Any) -> Optional[Fence]:
        if not self._queued:
            return None
        for host_pages, device_pages, windows, op in self._queued:
            self.device_kv[self._slots(device_pages)] = self.host_kv[self._slots(host_pages)]
            if windows is not None and windows[0]:
                host_w, dev_w = windows
                self.device_window_kv[self._slots(dev_w)] = self.host_window_kv[
                    self._slots(host_w)
                ]
            op.launched = True
        self._queued = []
        self.launches += 1
        return Fence(consumer_index=self.launches)

    def layer_hook(self) -> Any:
        return None

    def close(self) -> None:
        pass


class FakeStatePlane(StatePlane):
    """Runs every op at launch; the fake runtime's forwards come later, as
    real forwards wait on the fence."""

    def __init__(self, *, num_slots: int, host_num_slots: int):
        self.device = np.full(num_slots, -1, dtype=np.int64)
        self.host = np.full(max(host_num_slots, 1), -1, dtype=np.int64)
        self.host_num_slots = host_num_slots
        self._host = PageAllocator(num_pages=host_num_slots)
        self._unfenced = False

    def host_num_free(self) -> int:
        return self._host.num_free

    def alloc_host(self, num: int) -> Optional[list[int]]:
        return self._host.alloc(num)

    def free_host(self, slots: Sequence[int]) -> None:
        self._host.free(slots)

    def zero(self, slot: int, *, after: Any) -> None:
        self.device[slot] = 0
        self._unfenced = True

    def copy(self, src: int, dst: int, *, after: Any) -> None:
        self.device[dst] = self.device[src]
        self._unfenced = True

    def load(self, host_slot: int, slot: int, *, after: Any) -> None:
        self.device[slot] = self.host[host_slot]
        self._unfenced = True

    def backup(self, slot: int, host_slot: int, *, after: Any) -> None:
        self.host[host_slot] = self.device[slot]
        self._unfenced = True

    def fence_event(self, *, also_after: Any) -> Any:
        if not self._unfenced and also_after is None:
            return None
        self._unfenced = False
        return "state-fence"

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
        # Recurrent state: working slot, tracking slots, the last tracked
        # checkpoint (length, index into state_slots), next tracking index.
        self.state_slots: list[int] = []
        self.branch_len: Optional[int] = None
        self.tracked: Optional[tuple[int, int]] = None
        self.next_track = 0
        # Window slot per position (0: no window borrow).
        self.wslots: list[int] = []

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out


class FakeRuntime:
    """Drives a service like a scheduler: open, write KV, publish, decode, end."""

    def __init__(
        self,
        service: KVCacheService,
        device_kv: np.ndarray,
        *,
        seed: int = 0,
        state: Optional[FakeStatePlane] = None,
        checkpoint_grid: int = 0,
        window_kv: Optional[np.ndarray] = None,
        window_tokens: int = 0,
        page_size: int = 1,
    ):
        self.svc = service
        self.kv = device_kv
        self.state = state
        self.grid = checkpoint_grid
        self.window_kv = window_kv
        self.window = window_tokens
        self.ps = page_size
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
        if self.window:
            shared_w = [] if res.shared_window_slots is None else res.shared_window_slots.tolist()
            req.wslots = shared_w + res.mut_window_slots.tolist()
            assert len(req.wslots) == res.total_len
        assert (res.state is not None) == (self.state is not None)
        if res.state is not None:
            req.state_slots = list(res.state.slots)
            req.branch_len = res.state.branch_len
            req.tracked, req.next_track = None, 0
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
        if self.window:
            req.wslots += res.mut_window_slots.tolist()
        return True

    def forward(self, reqs: Sequence[FakeReq]) -> Optional[Fence]:
        """One batch: wait for loads, check shared KV, write the computed KV."""
        self.batch_id += 1
        fence = self.svc.begin_batch(self.batch_id, [r.lease for r in reqs])
        if self.state is not None:
            assert self.state._unfenced is False, "state ops launched after the fence"
        for r in reqs:
            toks = r.tokens
            for p in range(r.computed):
                assert self.kv[r.slots[p]] == kv_content(r.namespace, toks, p), (
                    f"rid={r.rid} pos={p} holds wrong KV"
                )
            for p in range(r.computed, len(r.slots)):
                self.kv[r.slots[p]] = kv_content(r.namespace, toks, p)
            if self.window:
                self._window_forward(r, toks)
            if self.state is not None:
                self._advance_state(r, toks, len(r.slots))
            r.computed = len(r.slots)
        self.svc.end_batch(self.batch_id, done_event=None)
        return fence

    def _window_forward(self, r: FakeReq, toks: list[int]) -> None:
        """Check the window KV the extend reads, write the new positions'."""
        for p in range(max(0, r.computed - self.window), r.computed):
            assert r.wslots[p] != 0, f"rid={r.rid} pos={p} has no window slot"
            assert self.window_kv[r.wslots[p]] == kv_content(r.namespace, toks, p), (
                f"rid={r.rid} pos={p} holds wrong window KV"
            )
        for p in range(r.computed, len(r.slots)):
            assert r.wslots[p] != 0, f"rid={r.rid} new pos={p} has no window slot"
            self.window_kv[r.wslots[p]] = kv_content(r.namespace, toks, p)

    def retire_windows(self, req: FakeReq) -> None:
        """Drop the window borrows a later forward will not read."""
        below = max(0, req.computed - self.window) // self.ps * self.ps
        self.svc.release(req.lease, window_below=below, from_len=None)
        for p in range(below):
            req.wslots[p] = 0

    def _advance_state(self, r: FakeReq, toks: list[int], end: int) -> None:
        """Check the working state, run it to ``end`` and track one checkpoint:
        the branch point if crossed, else the last grid multiple crossed."""
        dev = self.state.device
        assert dev[r.state_slots[0]] == state_content(r.namespace, toks, r.computed), (
            f"rid={r.rid} working state is not the state after {r.computed} tokens"
        )
        dev[r.state_slots[0]] = state_content(r.namespace, toks, end)
        if r.branch_len is not None and r.computed < r.branch_len <= end:
            boundary = r.branch_len
        else:
            boundary = end // self.grid * self.grid
            if boundary <= r.computed:
                return
        idx = 1 + r.next_track
        dev[r.state_slots[idx]] = state_content(r.namespace, toks, boundary)
        r.tracked = (boundary, idx)
        r.next_track = (r.next_track + 1) % (len(r.state_slots) - 1)

    def _anchor(self, req: FakeReq) -> Optional[StateAnchor]:
        if req.tracked is None:
            return None
        return StateAnchor(length=req.tracked[0], slot_index=req.tracked[1])

    def publish(self, req: FakeReq) -> None:
        self.svc.release_mut(
            req.lease, req.tokens[: req.computed], state=self._anchor(req)
        )

    def decode(
        self, reqs: Sequence[FakeReq], *, k: int = 1, revoke: bool = True
    ) -> Optional[list[FakeReq]]:
        """Grow every request by k positions; returns the survivors (None: the
        service refused the growth)."""
        batch = self.svc.acquire(
            [
                AcquireReq(
                    end=len(r.slots) + k, lease=r.lease, attrs=self.attrs(r), revoke=revoke
                )
                for r in reqs
            ]
        )
        if any(res.status == AcquireStatus.REFUSED for res in batch.results):
            assert not revoke
            assert all(res.status == AcquireStatus.REFUSED for res in batch.results)
            assert not batch.events
            return None
        alive = []
        for r, res in zip(reqs, batch.results):
            if res.status == AcquireStatus.REVOKED:
                r.lease, r.slots, r.computed = None, [], 0
                r.state_slots, r.tracked = [], None
                self.revoked.append(r)
                continue
            assert res.status == AcquireStatus.GRANTED
            r.slots += res.mut_slots.tolist()
            if self.window:
                r.wslots += res.mut_window_slots.tolist()
            r.out += [self.rng.randrange(1, 50) for _ in range(k)]
            alive.append(r)
        return alive

    def finish(self, req: FakeReq, *, publish: bool = True) -> None:
        self.svc.release(
            req.lease,
            publish_token_ids=req.tokens[: req.computed] if publish else None,
            state=self._anchor(req) if publish else None,
        )
        req.lease = None
        req.state_slots, req.tracked = [], None
