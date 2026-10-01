"""Per-process event trace of PD KV-cache transfers over NIXL (opt-in, off by default).

Turned on with ``SGLANG_KV_XFER_TRACE=1``. Every PD scheduler process writes its
own stream into ``SGLANG_KV_XFER_TRACE_DIR``. A prefill stream records what each
transfer moved, between which ranks, page indices and pools, and how long each
phase took, in seconds and in local forward passes; a decode stream records when
each destination was published and when its notifications were drained. The two
sides join on the bootstrap room, and a prefill peer joins a decode stream on
the NIXL agent name.

Format
------
One record per line, in the layout of ``mem_cache/cache_trace.py``::

    <seq>|<t_us>|<EVENT>|<detail>|fc=<n> th=<n>|S=<stack>

``seq``     per-process sequence; a gap means records were dropped by the byte cap.
``t_us``    microseconds since the stream opened. ``#H`` carries the wall clock at
            that instant as ``wall0``; every ``t*`` detail field uses the same base.
``detail``  space-separated ``k=v``, per event below.
``fc``      the scheduler's forward-pass counter (``Scheduler.forward_ct``) when
            the record was written; ``f*`` detail fields are the same counter.
``th``      small thread id, introduced by ``#T|<th>|<thread name>``.
``stack``   12 hex chars over the innermost ``SGLANG_KV_XFER_TRACE_DEPTH`` frames
            of the call that initiated the work, introduced by
            ``#S|<stack>|<file>:<func>:<line>;...`` (innermost first); ``-`` if none.

Other ``#`` lines: ``#H`` stream header (identity, ranks, clock base), ``#C`` byte
cap reached, ``#X`` tracer error, ``#E`` clean close.

Index summaries (``s`` source, ``d`` destination) are ``<p>n`` count, ``<p>lo`` /
``<p>hi`` bounds, ``<p>r`` contiguous runs, ``<p>c`` crc32 of the int32 array.
A src/dst pair also carries ``jr``, the runs in which both sides advance by one
(the descriptor grouping of the non-prepped NIXL path), and ``rh``, the histogram
of those run lengths in log2 buckets (1, 2-3, 4-7, ...).

Events (P = prefill stream, D = decode stream):

``LAYOUT``  P/D  one per run of alike registered regions: ``reg`` (kv, draft,
                 st.<state type>, aux), entry range ``e``, memory kind ``mk``,
                 bytes per slot ``item``, ``slots``, backing tensor ``t``, ``dt``, ``shp``.
``PEER``    P    a decode agent registered: endpoint, GPU, decode rank, the send
                 ``path`` (prep, slice, mixed, dcp) and ``dent``, the decode entry
                 each local KV entry lands in.
``ROOM``    P    a room bootstrapped: decode prefix and, per peer, the published
                 destination pages (``agent8:dummy:n:crc``).
``ENQ``     P    a chunk was queued for a transfer worker; ``S`` is the initiator.
``CHUNK``   P    a transfer worker finished one pass over a chunk: ``st`` (ok, err,
                 skip, defer), ``te``/``td``/``tp``/``ts`` enqueue, dequeue, last
                 post, settle; ``fe``/``fd``/``fs`` the counter at the same points.
``H``       P    one posted NIXL request: ``k`` kind (kv, kvm, kvs, kvd, kvf, stg,
                 aux, st.<type>), ``peer``, bytes ``B``, descriptors ``nd``, entries
                 ``ne``, ``t0``/``t1`` post window, ``tD`` first seen DONE (or ERR).
``IDX``     P    sampled full source/destination runs of one ``H``: ``s+d*len;``.
``RETIRE``  P    the scheduler released a room after its transfer concluded.
``META``    D    destination pages published to the prefill ranks ``tgt``; ``ts`` is
                 taken before the first send, so it precedes the prefill's ``ROOM``.
``RXDONE``  D    a receiver concluded: first/last KV notification drained
                 (``tk0``/``tk1``), aux (``ta``), state (``tst``), polls ``np``.
``COMMIT``  D    a transferred request's metadata was committed.
``ADMIT``   D    a transferred request entered a decode batch.
``STAT``    P/D  periodic queue depths and the tracer's own cost (``self_s``).

Everything the tracer does is wrapped: an error inside it is logged as ``#X``
and the server keeps running.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import logging
import os
import socket
import sys
import threading
import time
import traceback
import zlib
from collections import deque
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_ERRORS = 20
_NO_STACK = "-"
# Arbitrary: enough to show an index list's shape without a multi-MB line.
_MAX_IDX_RUNS = 4096


def _frame_label(code, lineno: int) -> str:
    path = code.co_filename
    cut = path.rfind("/")
    return f"{path[cut + 1 :]}:{code.co_name}:{lineno}"


def _as_i32(idx) -> np.ndarray:
    # Page and slot indices fit int32; a fixed dtype keeps crcs comparable across P and D.
    return np.ascontiguousarray(np.asarray(idx).reshape(-1), dtype=np.int32)


def _one_summary(prefix: str, a: np.ndarray) -> str:
    if a.size == 0:
        return f"{prefix}n=0"
    runs = 1 + int(np.count_nonzero(np.diff(a) != 1))
    return (
        f"{prefix}n={a.size} {prefix}lo={int(a.min())} {prefix}hi={int(a.max())} "
        f"{prefix}r={runs} {prefix}c={zlib.crc32(a.tobytes()):08x}"
    )


def _joint_breaks(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.flatnonzero((np.diff(s) != 1) | (np.diff(d) != 1)) + 1


def summarize_indices(src, dst) -> str:
    """Summary fields of a source/destination index pair; either may be None or a scalar."""
    parts = []
    s = d = None
    if src is not None:
        if np.isscalar(src):
            parts.append(f"si={int(src)}")
        else:
            s = _as_i32(src)
            parts.append(_one_summary("s", s))
    if dst is not None:
        if np.isscalar(dst):
            parts.append(f"di={int(dst)}")
        else:
            d = _as_i32(dst)
            parts.append(_one_summary("d", d))
    if s is not None and d is not None and s.size and s.size == d.size:
        brk = _joint_breaks(s, d)
        lens = np.diff(np.concatenate(([0], brk, [s.size])))
        hist = np.bincount(np.log2(lens).astype(np.int64))
        parts.append(f"jr={brk.size + 1} rh={'.'.join(str(int(x)) for x in hist)}")
    return " ".join(parts)


def index_runs(src, dst) -> str:
    """Joint runs of a src/dst pair as ``s+d*len`` tokens, capped at _MAX_IDX_RUNS."""
    s = _as_i32(src)
    d = _as_i32(dst)
    if s.size == 0 or s.size != d.size:
        return f"mismatch={s.size}/{d.size}"
    brk = _joint_breaks(s, d)
    starts = np.concatenate(([0], brk))
    lens = np.diff(np.concatenate((starts, [s.size])))
    n = min(starts.size, _MAX_IDX_RUNS)
    body = ";".join(
        f"{int(s[i])}+{int(d[i])}*{int(k)}" for i, k in zip(starts[:n], lens[:n])
    )
    more = f" more={starts.size - n}" if starts.size > n else ""
    return f"runs={body}{more}"


def _tensor_index(objs: Sequence[Any]) -> Dict[int, str]:
    """data_ptr -> "<owner>.<attr>[i] <dtype> [shape]" for tensors reachable from objs.

    Walks instance attributes two levels deep, descending only into objects
    defined in sglang, so a pool's sub-caches (e.g. the DSA index-K cache) are found.
    """
    import torch

    out: Dict[int, str] = {}

    def add(label: str, t) -> None:
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            out.setdefault(t.data_ptr(), f"{label} {str(t.dtype)[6:]} {list(t.shape)}")

    def walk(obj, name: str, depth: int) -> None:
        if depth > 2 or obj is None:
            return
        try:
            items = list(vars(obj).items())
        except TypeError:
            return
        for attr, val in items:
            label = f"{name}.{attr}"
            if isinstance(val, torch.Tensor):
                add(label, val)
            elif isinstance(val, (list, tuple)) and val:
                for i, v in enumerate(val):
                    add(f"{label}[{i}]", v)
            elif type(val).__module__.startswith("sglang.srt"):
                walk(val, label, depth + 1)

    for obj in objs:
        if obj is not None:
            walk(obj, type(obj).__name__, 0)
    return out


class _HandleTrace:
    __slots__ = (
        "peer",
        "kind",
        "nbytes",
        "ndesc",
        "nent",
        "t0",
        "t1",
        "t_done",
        "state",
        "src",
        "dst",
    )

    def __init__(self, peer, kind, nbytes, ndesc, nent, t0, t1, src, dst):
        self.peer = peer
        self.kind = kind
        self.nbytes = nbytes
        self.ndesc = ndesc
        self.nent = nent
        self.t0 = t0
        self.t1 = t1
        self.t_done = 0.0
        self.state = "-"
        self.src = src
        self.dst = dst


class ChunkTrace:
    """One transfer-worker pass over a TransferKVChunk."""

    __slots__ = (
        "room",
        "chunk_id",
        "last",
        "pages",
        "tokens",
        "worker",
        "enq",
        "t_deq",
        "fc_deq",
        "qd_deq",
        "handles",
        "pending",
        "status",
        "err",
        "t_settle",
        "fc_settle",
    )

    def __init__(self, chunk, worker: int, t_deq: float, fc_deq: int, qd: int):
        self.room = chunk.room
        self.chunk_id = chunk.chunk_id
        self.last = int(chunk.is_last_chunk)
        self.pages = len(chunk.prefill_kv_indices)
        self.tokens = -1 if chunk.num_kv_tokens is None else chunk.num_kv_tokens
        self.worker = worker
        # (t_enq, fc_enq, stack, qd_enq), set by KVXferTracer.enqueued().
        self.enq = chunk.trace
        self.t_deq = t_deq
        self.fc_deq = fc_deq
        self.qd_deq = qd
        self.handles: List[_HandleTrace] = []
        # id(handle) -> entries still waiting for their first DONE/ERR observation.
        self.pending: Dict[int, List[_HandleTrace]] = {}
        self.status = "skip"
        self.err = ""
        self.t_settle = 0.0
        self.fc_settle = -1

    def observe(self, handle, state: str) -> None:
        """Called from the settle poll loop for every state read; must stay cheap."""
        if state != "DONE" and state != "ERR":
            return
        waiting = self.pending.get(id(handle))
        if not waiting:
            return
        entry = waiting.pop(0)
        entry.t_done = time.perf_counter()
        entry.state = state


class _RxTrace:
    __slots__ = (
        "t_meta",
        "fc_meta",
        "polls",
        "t_k0",
        "fc_k0",
        "t_k1",
        "fc_k1",
        "nk",
        "pps",
        "t_aux",
        "t_state",
        "nst",
    )

    def __init__(self, t_meta: float, fc_meta: int):
        self.t_meta = t_meta
        self.fc_meta = fc_meta
        self.polls = 0
        self.t_k0 = 0.0
        self.fc_k0 = -1
        self.t_k1 = 0.0
        self.fc_k1 = -1
        self.nk = 0
        self.pps: Dict[int, int] = {}
        self.t_aux = 0.0
        self.t_state = 0.0
        self.nst = 0


class _Tls(threading.local):
    cur: Optional[ChunkTrace] = None


def _guarded(fn):
    """Never raise into the caller; account the time spent as tracer self-time."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            self._note_error(fn.__name__)
            return None
        finally:
            self._self_s += time.perf_counter() - t

    return wrapper


class KVXferTracer:
    """Single-process KV-transfer trace writer. Never raises into its callers."""

    def __init__(self) -> None:
        self.on = envs.SGLANG_KV_XFER_TRACE.get()
        self._dir = envs.SGLANG_KV_XFER_TRACE_DIR.get()
        self._depth = envs.SGLANG_KV_XFER_TRACE_DEPTH.get()
        self._sample = max(1, envs.SGLANG_KV_XFER_TRACE_SAMPLE.get())
        self._idx_every = envs.SGLANG_KV_XFER_TRACE_IDX_EVERY.get()
        self._max_bytes = envs.SGLANG_KV_XFER_TRACE_MAX_BYTES.get()
        self._flush_s = envs.SGLANG_KV_XFER_TRACE_FLUSH_S.get()
        self._stat_s = envs.SGLANG_KV_XFER_TRACE_STAT_S.get()
        self._t0 = time.perf_counter()
        self._wall0 = time.time()
        self._seq = count(1)
        self._q: deque = deque()
        self._io_lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._raw = None
        self._gz = None
        self._path = ""
        self._capped = False
        self._closed = False
        self._dropped = 0
        self._emitted = 0
        self._raw_bytes = 0
        self._errors = 0
        self._self_s = 0.0
        self._stacks: Dict[tuple, str] = {}
        self._threads: Dict[int, int] = {}
        self._tls = _Tls()
        self._fc: Callable[[], int] = lambda: -1
        self._mgr = None
        self._role = "?"
        self._rx: Dict[int, _RxTrace] = {}
        # Hetero-TP slice dlists carry a per-peer descriptor length that the send
        # path does not keep; recorded when the dlist is built.
        self.slice_desc_bytes: Dict[str, int] = {}
        self._flusher: Optional[threading.Thread] = None

    # ---- clock / identity -------------------------------------------------

    def _us(self, t: float) -> int:
        return int((t - self._t0) * 1e6) if t else -1

    def _fc_now(self) -> int:
        try:
            return int(self._fc())
        except Exception:
            return -1

    def sampled(self, room) -> bool:
        return self._sample == 1 or (room is not None and room % self._sample == 0)

    def _th(self) -> int:
        ident = threading.get_ident()
        th = self._threads.get(ident)
        if th is None:
            with self._meta_lock:
                th = self._threads.get(ident)
                if th is None:
                    th = len(self._threads)
                    self._threads[ident] = th
                    self._q.append(f"#T|{th}|{threading.current_thread().name}")
        return th

    def _stack(self, skip: int) -> str:
        """Hash of the innermost frames, starting ``skip`` frames above the caller."""
        f = sys._getframe(skip + 1)
        key = []
        n = self._depth
        while f is not None and n:
            key.append(f.f_code)
            key.append(f.f_lineno)
            f = f.f_back
            n -= 1
        tkey = tuple(key)
        h = self._stacks.get(tkey)
        if h is not None:
            return h
        frames = ";".join(
            _frame_label(tkey[i], tkey[i + 1]) for i in range(0, len(tkey), 2)
        )
        h = hashlib.sha256(frames.encode()).hexdigest()[:12]
        self._stacks[tkey] = h
        self._q.append(f"#S|{h}|{frames}")
        return h

    # ---- stream -----------------------------------------------------------

    def _emit(self, event: str, detail: str, stack: str = _NO_STACK) -> None:
        # Invariant a reader can check: records in the stream == emitted - dropped.
        self._emitted += 1
        if self._gz is None or self._capped:
            self._dropped += 1
            return
        self._q.append(
            f"{next(self._seq)}|{self._us(time.perf_counter())}|{event}|{detail}|"
            f"fc={self._fc_now()} th={self._th()}|S={stack}"
        )

    def _open(self, header: str) -> None:
        if self._gz is not None:
            self._q.append(header)
            return
        os.makedirs(self._dir, exist_ok=True)
        name = (
            f"kvxfer_{self._role}_{socket.gethostname()}_"
            f"g{self._mgr.kv_args.gpu_id}_pid{os.getpid()}.log.gz"
        )
        self._path = os.path.join(self._dir, name)
        self._raw = open(self._path, "wb")
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="wb", compresslevel=1)
        self._q.append(header)
        self._flusher = threading.Thread(
            target=self._flush_loop, name="kvxfer-trace-flush", daemon=True
        )
        self._flusher.start()

    def _flush(self) -> None:
        with self._io_lock:
            if self._gz is None:
                return
            n = len(self._q)
            if not n:
                return
            lines = [self._q.popleft() for _ in range(n)]
            if self._capped:
                # Keep the bounded "#" lines (stack table, errors, close marker).
                kept = [line for line in lines if line[0] == "#"]
                self._dropped += len(lines) - len(kept)
                lines = kept
                if not lines:
                    return
            data = ("\n".join(lines) + "\n").encode()
            self._gz.write(data)
            # Z_SYNC_FLUSH: a reader recovers everything up to here even if the
            # process is killed without closing the stream.
            self._gz.flush()
            self._raw_bytes += len(data)
            if not self._capped and self._raw.tell() >= self._max_bytes:
                self._capped = True
                self._gz.write(
                    f"#C|byte cap {self._max_bytes} reached after "
                    f"{self._emitted - self._dropped} records\n".encode()
                )
                self._gz.flush()

    def _flush_loop(self) -> None:
        next_stat = time.monotonic() + self._stat_s
        while not self._closed:
            time.sleep(self._flush_s)
            try:
                if time.monotonic() >= next_stat:
                    next_stat += self._stat_s
                    self._stat()
                self._flush()
            except Exception:
                self._note_error("flush")

    def close(self) -> None:
        if self._gz is None or self._closed:
            return
        try:
            self._stat()
            self._q.append(
                f"#E|emitted={self._emitted} dropped={self._dropped} "
                f"errors={self._errors} self_s={self._self_s:.3f}"
            )
            self._flush()
            self._closed = True
            with self._io_lock:
                self._gz.close()
                self._raw.close()
                self._gz = None
        except Exception:
            pass

    def _note_error(self, where: str) -> None:
        # Capped: a failure that repeats per call would otherwise flood the stream.
        self._errors += 1
        if self._errors > _MAX_ERRORS:
            return
        logger.exception("kv_xfer_trace: error in %s", where)
        detail = traceback.format_exc().strip().replace("\n", " // ")
        self._q.append(f"#X|{where}|{detail}")

    def _stat(self) -> None:
        mgr = self._mgr
        parts = []
        if mgr is not None:
            parts.append(f"rs={len(mgr.request_status)}")
            if self._role == "P":
                depths = [len(q) for q in mgr.transfer_queues]
                parts.append(
                    f"qd={sum(depths)} qmax={max(depths, default=0)} "
                    f"rooms={len(mgr.transfer_infos)} "
                    f"peers={len(mgr.decode_kv_args_table)}"
                )
            else:
                parts.append(f"rx={len(mgr.transfer_statuses)} rxt={len(self._rx)}")
        gz_bytes = self._raw.tell() if self._raw is not None else 0
        parts.append(
            f"recs={self._emitted} drop={self._dropped} err={self._errors} "
            f"raw_b={self._raw_bytes} gz_b={gz_bytes} self_s={self._self_s:.3f}"
        )
        self._emit("STAT", " ".join(parts))

    # ---- binding ----------------------------------------------------------

    @_guarded
    def bind(
        self,
        *,
        kv_manager,
        role: str,
        pools: Sequence[Any],
        forward_ct: Callable[[], int],
    ) -> None:
        """Open this process's stream and describe the registered KV regions."""
        from sglang.srt.runtime_context import get_parallel

        self._mgr = kv_manager
        self._role = role
        self._fc = forward_ct
        kv_args = kv_manager.kv_args
        par = get_parallel()
        if self._gz is None:
            # Restart the clock with the stream so wall0 and t_us=0 are one instant.
            self._t0 = time.perf_counter()
            self._wall0 = time.time()
        ranks = " ".join(
            f"{k}={getattr(par, k + '_rank')}/{getattr(par, k + '_size')}"
            for k in ("tp", "pp", "attn_tp", "attn_dp", "dp")
        )
        self._open(
            f"#H|version=1 role={role} pid={os.getpid()} host={socket.gethostname()} "
            f"ip={kv_manager.local_ip} port={kv_manager.rank_port} "
            f"gpu={kv_args.gpu_id} agent={kv_manager.agent.name} {ranks} "
            f"engine_rank={kv_args.engine_rank} sys_dp={kv_args.system_dp_rank} "
            f"page={kv_args.page_size} mla={int(bool(kv_manager.is_mla_backend))} "
            f"start_layer={kv_args.prefill_start_layer if role == 'P' else -1} "
            f"kv_dtype={kv_args.kv_cache_dtype_str} wall0={self._wall0:.6f} "
            f"t0_perf={self._t0:.6f} sample={self._sample} "
            f"idx_every={self._idx_every} depth={self._depth} "
            f"max_bytes={self._max_bytes}"
        )
        self._layout(kv_args, pools)
        self._flush()

    def _layout(self, kv_args, pools: Sequence[Any]) -> None:
        tensors = _tensor_index(pools)
        n_kv = len(kv_args.kv_data_ptrs)
        n_target = n_kv - kv_args.num_draft_entries
        kinds = kv_args.kv_data_mem_kinds
        rows = [
            (
                "kv" if i < n_target else "draft",
                kinds[i],
                kv_args.kv_item_lens[i],
                kv_args.kv_data_lens[i],
                tensors.get(kv_args.kv_data_ptrs[i], "? ? ?"),
            )
            for i in range(n_kv)
        ]
        self._emit_layout_rows(rows)
        for c, st in enumerate(kv_args.state_types):
            ptrs = kv_args.state_data_ptrs[c]
            rows = [
                (
                    f"st.{st.value}",
                    "VRAM",
                    kv_args.state_item_lens[c][i],
                    kv_args.state_data_lens[c][i],
                    tensors.get(ptrs[i], "? ? ?"),
                )
                for i in range(len(ptrs))
            ]
            self._emit_layout_rows(rows)
        rows = [
            ("aux", "DRAM", item, length, tensors.get(ptr, "? ? ?"))
            for ptr, length, item in zip(
                kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens
            )
        ]
        self._emit_layout_rows(rows)

    def _emit_layout_rows(self, rows) -> None:
        start = 0
        for i in range(1, len(rows) + 1):
            if i < len(rows) and self._layout_key(rows[i]) == self._layout_key(
                rows[start]
            ):
                continue
            reg, kind, item, length, tensor = rows[start]
            label, dtype, shape = tensor.split(" ", 2)
            slots = length // item if item else 0
            self._emit(
                "LAYOUT",
                f"reg={reg} e={start}-{i - 1} n={i - start} mk={kind} item={item} "
                f"slots={slots} t={label} dt={dtype} shp={shape.replace(' ', '')}",
            )
            start = i

    @staticmethod
    def _layout_key(row) -> tuple:
        reg, kind, item, length, tensor = row
        label, dtype, shape = tensor.split(" ", 2)
        return (reg, kind, item, length, label.split("[", 1)[0], dtype, shape)

    # ---- prefill ----------------------------------------------------------

    @_guarded
    def peer_added(self, info, *, mgr) -> None:
        if info.requires_dcp_relayout:
            path = "dcp"
        elif info.kv_xfer_segments is not None:
            path = "mixed"
        elif info.agent_name in mgr.prep_handles:
            path = "prep"
        elif info.agent_name in mgr.prep_handles_slice_dst:
            path = "slice"
        else:
            path = "none"
        dent = info.dst_entry_indices
        self._emit(
            "PEER",
            f"agent={info.agent_name} ep={info.endpoint}:{info.dst_port} "
            f"gpu={info.gpu_id} dtp={info.decode_tp_rank}/{info.decode_tp_size} "
            f"slots={info.dst_num_slots} ne={len(info.dst_kv_ptrs)} "
            f"dmk={'/'.join(sorted(set(info.dst_kv_mem_kinds)))} path={path} "
            f"dent={_compact_ranges(dent) if dent is not None else '-'}",
        )

    @_guarded
    def room_ready(self, room: int, infos: Dict[str, Any], prefix_len) -> None:
        if not self.sampled(room):
            return
        peers = ",".join(
            f"{name[:8]}:{int(info.is_dummy)}:{len(info.dst_kv_indices)}:"
            f"{zlib.crc32(_as_i32(info.dst_kv_indices).tobytes()):08x}"
            for name, info in infos.items()
        )
        self._emit(
            "ROOM",
            f"room={room} npeer={len(infos)} dpl={prefix_len} peers={peers}",
        )

    @_guarded
    def enqueued(self, chunk, *, shard: int, queue) -> None:
        if not self.sampled(chunk.room):
            return
        qd = len(queue)
        # Frames above add_transfer_request: the sender and the scheduler path.
        stack = self._stack(3)
        chunk.trace = (time.perf_counter(), self._fc_now(), stack, qd)
        self._emit(
            "ENQ",
            f"room={chunk.room} ck={chunk.chunk_id} last={int(chunk.is_last_chunk)} "
            f"pg={len(chunk.prefill_kv_indices)} tok={chunk.num_kv_tokens} "
            f"nst={len(chunk.state_indices) if chunk.state_indices else 0} "
            f"q={shard} qd={qd}",
            stack,
        )

    @_guarded
    def chunk_begin(self, chunk, *, worker: int, queue) -> Optional[ChunkTrace]:
        if not self.sampled(chunk.room):
            return None
        try:
            qd = len(queue)
        except TypeError:
            qd = -1
        ctx = ChunkTrace(chunk, worker, time.perf_counter(), self._fc_now(), qd)
        self._tls.cur = ctx
        return ctx

    def cur(self) -> Optional[ChunkTrace]:
        return self._tls.cur

    @_guarded
    def posted(
        self,
        *,
        handle,
        peer: str,
        kind: str,
        t0: float,
        nbytes: int,
        ndesc: int,
        nent: int,
        src=None,
        dst=None,
    ) -> None:
        ctx = self._tls.cur
        entry = _HandleTrace(
            peer, kind, int(nbytes), int(ndesc), nent, t0, time.perf_counter(), src, dst
        )
        if ctx is None:
            self._emit_handle(None, entry)
            return
        ctx.handles.append(entry)
        ctx.pending.setdefault(id(handle), []).append(entry)

    @_guarded
    def chunk_settled(self, ctx: ChunkTrace) -> None:
        ctx.t_settle = time.perf_counter()
        ctx.fc_settle = self._fc_now()

    @_guarded
    def chunk_failed(self, ctx: ChunkTrace, exc: BaseException) -> None:
        ctx.status = "err"
        ctx.err = type(exc).__name__

    @_guarded
    def chunk_end(self, ctx: ChunkTrace) -> None:
        self._tls.cur = None
        t_enq, fc_enq, stack, qd_enq = ctx.enq if ctx.enq else (0.0, -1, _NO_STACK, -1)
        t_post = max((h.t1 for h in ctx.handles), default=0.0)
        self._emit(
            "CHUNK",
            f"room={ctx.room} ck={ctx.chunk_id} last={ctx.last} pg={ctx.pages} "
            f"tok={ctx.tokens} w={ctx.worker} st={ctx.status}"
            f"{' err=' + ctx.err if ctx.err else ''} "
            f"te={self._us(t_enq)} td={self._us(ctx.t_deq)} tp={self._us(t_post)} "
            f"ts={self._us(ctx.t_settle)} fe={fc_enq} fd={ctx.fc_deq} "
            f"fs={ctx.fc_settle} qe={qd_enq} qd={ctx.qd_deq} nh={len(ctx.handles)} "
            f"B={sum(h.nbytes for h in ctx.handles)}",
            stack,
        )
        dump_idx = self._idx_every > 0 and ctx.room % self._idx_every == 0
        for entry in ctx.handles:
            self._emit_handle(ctx, entry)
            if dump_idx and entry.src is not None and not np.isscalar(entry.src):
                self._emit(
                    "IDX",
                    f"room={ctx.room} ck={ctx.chunk_id} k={entry.kind} "
                    f"peer={entry.peer[:8]} {index_runs(entry.src, entry.dst)}",
                )

    def _emit_handle(self, ctx: Optional[ChunkTrace], h: _HandleTrace) -> None:
        where = f"room={ctx.room} ck={ctx.chunk_id}" if ctx else "room=- ck=-"
        self._emit(
            "H",
            f"{where} k={h.kind} peer={h.peer[:8]} B={h.nbytes} nd={h.ndesc} "
            f"ne={h.nent} t0={self._us(h.t0)} t1={self._us(h.t1)} "
            f"tD={self._us(h.t_done)} hs={h.state} "
            f"{summarize_indices(h.src, h.dst)}",
        )

    @_guarded
    def retire(self, *, room: int, rid: str, ok: bool, sender) -> None:
        from sglang.srt.disaggregation.nixl.conn import NixlKVSender

        if not self.sampled(room):
            return
        extra = ""
        if isinstance(sender, NixlKVSender):
            lat = sender._transfer_metric.transfer_latency_s
            extra = (
                f" nck={sender.chunk_id} pg={sender._transfer_num_kv_indices} "
                f"t1st={self._us(sender._transfer_start_time or 0.0)} "
                f"lat={-1 if lat is None else round(lat, 6)}"
            )
        self._emit(
            "RETIRE", f"room={room} rid={rid} st={'ok' if ok else 'fail'}{extra}"
        )

    # ---- decode -----------------------------------------------------------

    @_guarded
    def meta_sent(
        self,
        *,
        room: int,
        t_publish: float,
        kv_indices,
        aux_index,
        state_indices,
        decode_prefix_len,
        targets: Sequence[dict],
        nresp,
    ) -> None:
        if not self.sampled(room):
            return
        # Frames above send_metadata: the prealloc path that published it.
        stack = self._stack(3)
        self._rx[room] = _RxTrace(t_publish, self._fc_now())
        tgt = ",".join(
            f"{t['rank_ip']}:{t['rank_port']}:{t['pp_rank']}:{int(t['is_dummy'])}"
            for t in targets
        )
        nst = (
            sum(len(x) for x in state_indices if x is not None) if state_indices else 0
        )
        self._emit(
            "META",
            f"room={room} ts={self._us(t_publish)} {summarize_indices(None, kv_indices)} "
            f"aux={aux_index} nst={nst} dpl={decode_prefix_len} nresp={nresp} tgt={tgt}",
            stack,
        )

    @_guarded
    def rx_notif(self, room: int, components: List[str], t: float) -> None:
        rx = self._rx.get(room)
        if rx is None:
            return
        tag = components[1]
        if tag in ("kv", "stg"):
            pp = int(components[4]) if len(components) > 4 else 0
            if not rx.nk:
                rx.t_k0 = t
                rx.fc_k0 = self._fc_now()
            rx.t_k1 = t
            rx.fc_k1 = self._fc_now()
            rx.nk += 1
            rx.pps[pp] = rx.pps.get(pp, 0) + 1
        elif tag == "aux":
            rx.t_aux = t
        elif tag == "state":
            rx.t_state = t
            rx.nst += 1

    def rx_poll(self, room: int) -> None:
        rx = self._rx.get(room)
        if rx is not None:
            rx.polls += 1

    @_guarded
    def rx_end(self, room: int, status: str) -> None:
        rx = self._rx.pop(room, None)
        if rx is None:
            return
        self._emit(
            "RXDONE",
            f"room={room} st={status} tm={self._us(rx.t_meta)} "
            f"tk0={self._us(rx.t_k0)} tk1={self._us(rx.t_k1)} "
            f"ta={self._us(rx.t_aux)} tst={self._us(rx.t_state)} nk={rx.nk} "
            f"nst={rx.nst} pp={'/'.join(f'{p}:{n}' for p, n in sorted(rx.pps.items()))} "
            f"np={rx.polls} fm={rx.fc_meta} fk0={rx.fc_k0} fk1={rx.fc_k1}",
        )

    @_guarded
    def req_event(self, event: str, *, room, rid: str) -> None:
        if room is None or not self.sampled(room):
            return
        self._emit(event, f"room={room} rid={rid}")


def _compact_ranges(values: Sequence[int]) -> str:
    """[3, 4, 5, 9] -> "3-5,9"."""
    out = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(f"{values[i]}" if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ",".join(out)


TRACE = KVXferTracer()
