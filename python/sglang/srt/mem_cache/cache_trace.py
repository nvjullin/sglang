"""Per-rank event trace of prefix-cache mutations (opt-in, off by default).

Turned on with ``SGLANG_CACHE_TRACE=1``. Every scheduler process writes its own
stream, because each DP rank owns an independent radix tree, device KV pool and
host KV pool; mixing two ranks into one file makes the pool-occupancy fields
unreadable.

The stream is designed to be replayable: every record that changes tree or pool
state carries the node id, the token count moved, and the pool occupancies
observed at that instant, so a reader can either rebuild the tree and simulate
against it, or check each record locally (``a pool free-count must move by
exactly the tokens this record claims to have moved``) without rebuilding
anything.

Format
------
One record per line::

    <seq>|<t_us>|<EVENT>|<detail>|<common>|S=<stack>

``seq``     monotonic per process; gaps mean records were dropped by the byte cap.
``t_us``    microseconds since the tracer opened the file.
``detail``  space-separated ``k=v``, event-specific (see EVENTS below).
``common``  space-separated ``k=v``, present on every record:

    dF  device KV pool free tokens          hF  host KV pool free tokens
    dE  tree device-evictable tokens        nN  live tree nodes
    dP  tree device-protected tokens        bs  batch size of the last forward
    fc  forward counter                     rq  ambient request id ('-' if none)
    rl  ambient request length              (tokens; -1 if none)

``stack``   12 hex chars identifying the call stack. The first time a stack is
            seen the file gets a self-describing line::

                #S|<stack>|<file>:<func>:<line>;<file>:<func>:<line>;...

            innermost frame first. The same table is also written to
            ``<stream>.stacks.json`` and ``<stream>.stacks.pkl`` at exit, so the
            trace can be read either standalone or with the side table.

Everything the tracer does is wrapped: an error inside it disables tracing and
lets the server keep running rather than losing the job.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

# Env knobs. Only SGLANG_CACHE_TRACE is required.
_ENABLED = os.environ.get("SGLANG_CACHE_TRACE", "0") not in ("0", "", "false", "False")
_DIR = os.environ.get("SGLANG_CACHE_TRACE_DIR", "/logs/cachetrace")
_DEPTH = int(os.environ.get("SGLANG_CACHE_TRACE_DEPTH", "14"))
_GZIP = os.environ.get("SGLANG_CACHE_TRACE_GZIP", "1") not in ("0", "", "false")
_MAX_BYTES = int(os.environ.get("SGLANG_CACHE_TRACE_MAX_BYTES", str(4 << 30)))
_FLUSH_EVERY = int(os.environ.get("SGLANG_CACHE_TRACE_FLUSH_EVERY", "1000"))
_NVTX = os.environ.get("SGLANG_CACHE_TRACE_NVTX", "0") not in ("0", "", "false", "False")
_MAX_ERRORS = 20


def _frame_label(code, lineno: int) -> str:
    path = code.co_filename
    cut = path.rfind("/")
    return f"{path[cut + 1:]}:{code.co_name}:{lineno}"


class _NvtxMirror:
    """Mirror trace records onto an Nsight Systems timeline (``SGLANG_CACHE_TRACE_NVTX=1``).

    A capture says which copies ran and when; this trace says which node moved and why.
    The two are written in different clocks, so pairing them afterwards means aligning two
    timebases by hand. Emitting each record as an NVTX marker from the same process puts
    both on one timeline instead, which is the point of running the tracer and the profiler
    together at all.

    Begin/end records additionally open a range, so a host-to-device load-back reads as one
    bar spanning the copies it issued rather than as two unrelated points. The range is
    process-scoped (``nvtxRangeStartA`` / ``nvtxRangeEnd``) rather than the thread-stacked
    push/pop pair, because these events interleave across threads and do not nest.

    Nothing here raises into the tracer: the first failure disables the mirror and leaves
    the file stream untouched.
    """

    # Opening record -> range name. Both records of a pair open their detail field with the
    # same token -- ``n=<node>`` for the transfer pair, ``want=<tokens>`` for the two
    # reclaim pairs -- which is what an open range is keyed on.
    _BEGINS = {"H2D_BEGIN": "H2D", "EVICT_BEGIN": "EVICT", "RECLAIM_RUN": "RECLAIM"}
    _ENDS = {"H2D_END": "H2D", "EVICT_END": "EVICT", "RECLAIM_DONE": "RECLAIM"}
    # Detail fields run to a few hundred characters; a span name that long is unreadable on
    # a timeline, and the full record is in the file anyway.
    _MAX_DETAIL = 120

    def __init__(self) -> None:
        self.on = False
        self._mark = None
        self._start = None
        self._stop = None
        self._open: dict[tuple, int] = {}

    def enable(self) -> None:
        if not (_ENABLED and _NVTX):
            return
        try:
            # Bound off torch's raw bindings rather than the torch.cuda.nvtx wrappers,
            # which run msg.format() on the span name and so raise on any brace in a
            # detail field.
            from torch._C import _nvtx

            self._mark = _nvtx.markA
            self._start = getattr(_nvtx, "rangeStartA", None)
            self._stop = getattr(_nvtx, "rangeEnd", None)
        except Exception:
            logger.warning(
                "SGLANG_CACHE_TRACE_NVTX=1 but torch's NVTX bindings are unavailable; "
                "cache events will not reach the profiler timeline."
            )
            return
        if self._start is None or self._stop is None:
            logger.warning(
                "SGLANG_CACHE_TRACE_NVTX=1: torch exposes no NVTX start/end range; "
                "emitting markers only, so transfers appear as points rather than bars."
            )
            self._start = self._stop = None
        self.on = True

    def mirror(self, event: str, detail: str) -> None:
        try:
            short = detail[: self._MAX_DETAIL]
            self._mark(f"ct.{event} {short}" if short else f"ct.{event}")
            if self._start is None:
                return
            key = detail.split(" ", 1)[0]
            name = self._BEGINS.get(event)
            if name is not None:
                slot = (name, key)
                prior = self._open.pop(slot, None)
                if prior is not None:
                    self._stop(prior)  # an opener that never closed; do not leak it
                self._open[slot] = self._start(f"ct.{name} {short}")
                return
            name = self._ENDS.get(event)
            if name is not None:
                prior = self._open.pop((name, key), None)
                if prior is not None:
                    self._stop(prior)
        except Exception:
            self.on = False

    def close(self) -> None:
        """End ranges still open, so none is drawn running to the end of the capture."""
        if not self.on:
            return
        self.on = False
        try:
            for handle in self._open.values():
                self._stop(handle)
        except Exception:
            pass
        self._open.clear()


class _Tracer:
    """Single-process trace writer. Never raises into its callers."""

    def __init__(self) -> None:
        self.on = False
        self._fh = None
        self._seq = 0
        self._t0 = time.perf_counter()
        self._stacks: dict[tuple, str] = {}
        self._table: dict[str, str] = {}
        self._pending = 0
        self._bytes = 0
        self._capped = False
        self._errors = 0
        self._lock = threading.RLock()
        # Ambient context, set by the scheduler hooks.
        self._bs = 0
        self._fc = 0
        self._rq = "-"
        self._rl = -1
        # Pool accessors, bound by attach().
        self._dev_free = None
        self._host_free = None
        self._tree = None
        self._cache = None

    # ---- lifecycle -------------------------------------------------------

    def _open(self) -> None:
        if self._fh is not None or not _ENABLED:
            return
        try:
            os.makedirs(_DIR, exist_ok=True)
            rank = self._guess_rank()
            base = os.path.join(_DIR, f"cachetrace_{rank}_pid{os.getpid()}.log")
            if _GZIP:
                import gzip

                self._fh = gzip.open(base + ".gz", "wt", compresslevel=1)
                self._path = base + ".gz"
            else:
                self._fh = open(base, "w", buffering=1 << 20)
                self._path = base
            self._t0 = time.perf_counter()
            atexit.register(self.close)
            self._raw(
                f"#H|version=1 pid={os.getpid()} rank={rank} depth={_DEPTH} "
                f"wallclock={time.time():.3f} max_bytes={_MAX_BYTES} "
                f"ranks={self._rank_detail()}"
            )
        except Exception:
            logger.exception("cache_trace: could not open trace stream; disabled")
            self.on = False

    @staticmethod
    def _ranks() -> dict[str, int]:
        """Whatever rank identifiers are resolvable right now."""
        out = {}
        try:
            from sglang.srt.layers.dp_attention import get_attention_dp_rank

            out["dp"] = get_attention_dp_rank()
        except Exception:
            pass
        try:
            from sglang.srt.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            out["tp"] = get_tensor_model_parallel_rank()
        except Exception:
            pass
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                out["world"] = dist.get_rank()
        except Exception:
            pass
        return out

    def _guess_rank(self) -> str:
        r = self._ranks()
        for key in ("dp", "world", "tp"):
            if key in r:
                return f"{key}{r[key]}"
        return "rX"

    def _rank_detail(self) -> str:
        r = self._ranks()
        return "/".join(f"{k}{v}" for k, v in sorted(r.items())) or "unknown"

    def attach(self, cache) -> None:
        """Bind pool accessors from a UnifiedRadixCache and log its capacities."""
        if not self.on:
            return
        try:
            alloc = cache.token_to_kv_pool_allocator
            self._dev_free = alloc.available_size
            self._tree = cache.tree_core
            self._cache = cache
            self.emit(
                "INIT",
                f"dev_cap={alloc.size} page={cache.page_size} "
                f"write_back={int(bool(cache.is_write_back))} "
                f"components={'/'.join(str(c) for c in cache.tree_components)}",
            )
        except Exception:
            self._note_error("attach")

    def bind_host(self, cache) -> None:
        """Bind the host pool accessor; the pool only exists after init_hicache."""
        if not self.on:
            return
        try:
            pool = cache.cache_controller.mem_pool_host
            self._host_free = pool.available_size
            self.emit(
                "HOSTPOOL",
                f"host_cap={pool.size} host_logical={pool.logical_size} "
                f"host_free={pool.available_size()} "
                f"policy={cache.cache_controller.write_policy}",
            )
        except Exception:
            self._note_error("bind_host")

    def close(self) -> None:
        _NVTX_MIRROR.close()
        if self._fh is None:
            return
        self.on = False
        try:
            self._raw(
                f"#E|seq={self._seq} bytes={self._bytes} "
                f"capped={int(self._capped)} errors={self._errors}"
            )
            self._fh.close()
            import json
            import pickle

            with open(self._path + ".stacks.json", "w") as f:
                json.dump(self._table, f, indent=0)
            with open(self._path + ".stacks.pkl", "wb") as f:
                pickle.dump(self._table, f)
        except Exception:
            pass
        finally:
            self._fh = None

    # ---- ambient context -------------------------------------------------

    def set_batch(self, bs: int, fc: int) -> None:
        self._bs = bs
        self._fc = fc

    def set_req(self, rid: str, rlen: int) -> None:
        self._rq = rid
        self._rl = rlen

    # ---- emit ------------------------------------------------------------

    def _raw(self, line: str) -> None:
        self._fh.write(line)
        self._fh.write("\n")
        self._bytes += len(line) + 1

    def _note_error(self, where: str) -> None:
        # Capped: a failure that repeats per call site would otherwise turn the
        # stream into tracebacks and stall the worker on flushes.
        self._errors += 1
        if self._errors > _MAX_ERRORS:
            return
        detail = traceback.format_exc().strip().replace("\n", " // ")
        logger.exception("cache_trace: error in %s", where)
        if self._fh is None and self.on:
            self._open()
        if self._fh is not None:
            try:
                self._raw(f"#X|{where}|{detail}")
                self._fh.flush()
            except Exception:
                pass

    def _fail(self, where: str) -> None:
        """Give up on tracing entirely; only emit() uses this."""
        self._note_error(where)
        self.on = False

    def _stack(self) -> str:
        # Skip _stack + emit; start at the instrumented call site.
        f = sys._getframe(2)
        key = []
        n = _DEPTH
        while f is not None and n:
            key.append(f.f_code)
            key.append(f.f_lineno)
            f = f.f_back
            n -= 1
        tkey = tuple(key)
        h = self._stacks.get(tkey)
        if h is not None:
            return h
        frames = [_frame_label(tkey[i], tkey[i + 1]) for i in range(0, len(tkey), 2)]
        joined = ";".join(frames)
        h = hashlib.sha256(joined.encode()).hexdigest()[:12]
        self._stacks[tkey] = h
        self._table[h] = joined
        self._raw(f"#S|{h}|{joined}")
        return h

    def _pools(self):
        """(device free, host free, evictable, protected, nodes); -1 when unknown."""
        try:
            dF = self._dev_free() if self._dev_free is not None else -1
            hF = self._host_free() if self._host_free is not None else -1
            tree = self._tree
            if tree is None:
                return dF, hF, -1, -1, -1
            return (
                dF,
                hF,
                tree.evictable_size(),
                tree.protected_size(),
                len(tree._node_arena),
            )
        except Exception:
            return -1, -1, -1, -1, -1

    def emit(self, event: str, detail: str = "") -> None:
        if not self.on:
            return
        # Ahead of the file write: the pool reads and stack hash below cost tens of
        # microseconds, and a marker is only useful if it lands where the event did.
        if _NVTX_MIRROR.on:
            _NVTX_MIRROR.mirror(event, detail)
        try:
            if self._fh is None:
                self._open()
                if self._fh is None:
                    return
            if self._bytes >= _MAX_BYTES:
                if not self._capped:
                    self._capped = True
                    self._raw(f"#C|byte cap {_MAX_BYTES} reached at seq={self._seq}")
                    self._fh.flush()
                self._seq += 1
                return
            with self._lock:
                self._seq += 1
                dF, hF, dE, dP, nN = self._pools()
                stack = self._stack()
                self._raw(
                    f"{self._seq}|{int((time.perf_counter() - self._t0) * 1e6)}|"
                    f"{event}|{detail}|"
                    f"dF={dF} dE={dE} dP={dP} hF={hF} nN={nN} "
                    f"bs={self._bs} fc={self._fc} rq={self._rq} rl={self._rl}|"
                    f"S={stack}"
                )
                self._pending += 1
                if self._pending >= _FLUSH_EVERY:
                    self._pending = 0
                    self._fh.flush()
        except Exception:
            self._fail("emit")


TRACE = _Tracer()
TRACE.on = _ENABLED
_NVTX_MIRROR = _NvtxMirror()
_NVTX_MIRROR.enable()


def node_flags(node) -> str:
    """Residency + lock state of one tree node, for the detail field.

    dv/hv: Full KV present on device / host. lk/hlk: Full lock refs. wt/lb:
    in-flight write-through / load-back anchors.
    """
    try:
        cd = node.component_data[0]
        return (
            f"dv={0 if cd.value is None else len(cd.value)} "
            f"hv={0 if cd.host_value is None else len(cd.host_value)} "
            f"lk={cd.lock_ref} hlk={cd.host_lock_ref} "
            f"wt={-1 if node.write_through_pending_id is None else node.write_through_pending_id} "
            f"lb={-1 if node.load_back_pending_id is None else node.load_back_pending_id} "
            f"kl={0 if node.key is None else len(node.key)} "
            f"nc={len(node.children)} "
            f"par={-1 if node.parent is None else node.parent.id}"
        )
    except Exception:
        return "dv=? hv=? lk=? hlk=? wt=? lb=? kl=? nc=? par=?"
