"""CPU unit tests for the NIXL KV-transfer trace (SGLANG_KV_XFER_TRACE)."""

import gzip
import os
import tempfile
import threading
import unittest
import zlib
from collections import defaultdict
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import sglang.srt.disaggregation.nixl.conn as nixl_conn
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.kv_xfer_trace import KVXferTracer, summarize_indices
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
    NixlKVReceiver,
    TransferInfo,
    TransferStatus,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE = 4
ROOM = 4242


def _fields(blob: str) -> dict:
    return dict(tok.split("=", 1) for tok in blob.split(" ") if "=" in tok)


def _read_stream(path: str):
    """(records by event, '#'-line kinds -> lines) of one trace stream."""
    records = defaultdict(list)
    meta = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                kind, _, rest = line.partition("|")
                meta[kind].append(rest)
                continue
            seq, t_us, event, detail, common, stack = line.split("|")
            rec = _fields(detail)
            rec.update(_fields(common))
            rec["S"] = stack[2:]
            rec["seq"] = int(seq)
            rec["t_us"] = int(t_us)
            records[event].append(rec)
    return records, meta


class _RecordingAgent:
    """Fake nixl_agent that records what each request posted and settles each
    handle on its second state read, as a transfer in flight would."""

    def __init__(self, name="prefill-agent"):
        self.name = name
        self.prepped = []
        self.descs = []
        self.reads = defaultdict(int)
        self.on_read = None

    def make_prepped_xfer(self, op, src_prep, src_idx, dst_prep, dst_idx, notif):
        handle = object()
        self.prepped.append((handle, np.asarray(src_idx), np.asarray(dst_idx)))
        return handle

    def get_xfer_descs(self, reqs, mem_kind):
        arr = np.asarray(reqs, dtype=np.uint64).reshape(-1, 3)
        self.descs.append(arr)
        return arr

    def initialize_xfer(self, op, src_descs, dst_descs, peer, notif):
        return object()

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        self.reads[id(handle)] += 1
        if self.on_read is not None:
            self.on_read()
        return "DONE" if self.reads[id(handle)] >= 2 else "PROC"


def _tracer_env(stack: ExitStack, trace_dir: str, **extra):
    values = dict(
        SGLANG_KV_XFER_TRACE=True,
        SGLANG_KV_XFER_TRACE_DIR=trace_dir,
        SGLANG_KV_XFER_TRACE_FLUSH_S=60.0,
        SGLANG_KV_XFER_TRACE_STAT_S=600.0,
    )
    values.update(extra)
    for name, value in values.items():
        stack.enter_context(getattr(envs, name).override(value))
    stack.enter_context(
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(
                **{
                    f"{k}_{x}": (1 if x == "size" else 0)
                    for k in ("tp", "pp", "attn_tp", "attn_dp", "dp")
                    for x in ("rank", "size")
                }
            ),
        )
    )


def _bind(tracer, mgr, role, pools=()):
    counter = SimpleNamespace(fc=0)
    tracer.bind(kv_manager=mgr, role=role, pools=pools, forward_ct=lambda: counter.fc)
    return counter


class TestIndexSummary(CustomTestCase):
    def test_pair_summary_matches_wire_crc_and_nixl_descriptor_grouping(self):
        """P and D join destination pages by crc, so the crc must not depend on the
        index dtype each side happens to hold; and ``jr`` must count the
        descriptors the non-prepped NIXL path builds per layer."""
        src = np.array([3, 4, 5, 9, 10, 30], dtype=np.int64)
        dst = np.array([7, 8, 9, 20, 22, 23], dtype=np.int64)
        fields = _fields(summarize_indices(src, dst))

        self.assertEqual(
            fields["dc"], f"{zlib.crc32(dst.astype(np.int32).tobytes()):08x}"
        )
        src_groups, _ = group_concurrent_contiguous(src, dst)
        self.assertEqual(int(fields["jr"]), len(src_groups))
        # Runs of length 3, 1, 1, 1 -> log2 buckets [1]: 3, [2-3]: 1.
        self.assertEqual(fields["rh"], "3.1")


class TestPrefillWorkerTrace(CustomTestCase):
    def _make_manager(self, agent):
        kv_buffers = [torch.zeros((16, 1, 8), dtype=torch.uint8) for _ in range(3)]
        index_k = [torch.zeros((4, 32), dtype=torch.uint8) for _ in range(2)]
        aux = torch.zeros((8, 2), dtype=torch.int32)
        pools = (
            SimpleNamespace(kv_buffer=kv_buffers),
            SimpleNamespace(index_k=index_k),
            SimpleNamespace(aux=aux),
        )
        mgr = object.__new__(NixlKVManager)
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.agent = agent
        mgr.local_ip = "10.0.0.1"
        mgr.rank_port = 5000
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.pp_size = 1
        mgr.transfer_source_rank = 0
        mgr.src_mem_kind = "VRAM"
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr.exceptions = {}
        mgr.failure_lock = threading.Lock()
        mgr.failure_records = {}
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr.req_to_decode_prefix_len = {ROOM: 0}
        mgr.transfer_queues = [FastQueue()]
        mgr.prep_handles = {"": "src_prep", "decode-agent": "dst_prep"}
        mgr._num_slots_src = 16
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            gpu_id=0,
            system_dp_rank=0,
            page_size=PAGE,
            prefill_start_layer=0,
            kv_cache_dtype_str="fp8_e4m3",
            kv_data_ptrs=[t.data_ptr() for t in kv_buffers],
            kv_data_lens=[t.nbytes for t in kv_buffers],
            kv_item_lens=[t[0].nbytes * PAGE for t in kv_buffers],
            num_draft_entries=1,
            kv_data_mem_kinds=["VRAM"] * 3,
            state_types=[StateType.DSA],
            state_data_ptrs=[[t.data_ptr() for t in index_k]],
            state_data_lens=[[t.nbytes for t in index_k]],
            state_item_lens=[[t[0].nbytes for t in index_k]],
            state_layer_ids=[[]],
            aux_data_ptrs=[aux.data_ptr()],
            aux_data_lens=[aux.nbytes],
            aux_item_lens=[aux[0].nbytes],
        )
        mgr.decode_kv_args_table = {
            "decode-agent": SimpleNamespace(
                agent_name="decode-agent",
                decode_tp_size=1,
                decode_tp_rank=0,
                dst_kv_ptrs=[0x1000, 0x2000, 0x3000],
                dst_aux_ptrs=[0x4000],
                dst_state_data_ptrs=[[0x5000, 0x6000]],
                dst_state_item_lens=[[32, 32]],
                dst_state_dim_per_tensor=[],
                dst_state_layer_ids=[],
                dst_num_slots=16,
                gpu_id=1,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                requires_dcp_relayout=False,
            )
        }
        mgr.transfer_infos = {
            ROOM: {
                "decode-agent": TransferInfo(
                    room=ROOM,
                    endpoint="10.0.0.2",
                    dst_port=6000,
                    agent_name="decode-agent",
                    dst_kv_indices=np.array([7, 8, 9, 20], dtype=np.int32),
                    dst_aux_index=5,
                    required_dst_info_num=1,
                    dst_state_indices=[[7, 8, 9, 20]],
                )
            }
        }
        return mgr, pools

    def test_chunk_trace_accounts_what_was_posted_and_who_initiated_it(self):
        """Each H record must carry the bytes and descriptor count NIXL was
        actually handed, its own completion, and the chunk the scheduler-side
        stack that queued it; a new or changed post path breaks this."""
        agent = _RecordingAgent()
        mgr, pools = self._make_manager(agent)
        with tempfile.TemporaryDirectory() as trace_dir, ExitStack() as stack:
            _tracer_env(stack, trace_dir)
            tracer = KVXferTracer()
            stack.enter_context(patch.object(nixl_conn, "_KVXT", tracer))
            counter = _bind(tracer, mgr, "P", pools)

            counter.fc = 10
            kv_pages = np.array([3, 4, 5, 9], dtype=np.int64)
            mgr.add_transfer_request(
                ROOM,
                kv_pages,
                slice(0, 4),
                True,
                0,
                aux_index=2,
                state_indices=[kv_pages],
                num_kv_tokens=4 * PAGE,
            )
            chunk = mgr.transfer_queues[0].get()
            counter.fc = 12

            def advance():
                counter.fc += 1

            agent.on_read = advance
            queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
            with self.assertRaises(SystemExit):
                mgr.transfer_worker(queue)
            tracer.close()

            self.assertEqual(mgr.request_status[ROOM], KVPoll.Success)
            records, meta = _read_stream(tracer._path)

        layouts = {r["reg"]: r for r in records["LAYOUT"]}
        self.assertEqual(layouts["kv"]["e"], "0-1")
        self.assertEqual(layouts["draft"]["e"], "2-2")
        self.assertEqual(layouts["kv"]["dt"], "uint8")
        self.assertEqual(layouts["kv"]["shp"], "[16,1,8]")
        self.assertEqual(layouts["st.dsa"]["item"], "32")

        (chunk_rec,) = records["CHUNK"]
        self.assertEqual(chunk_rec["st"], "ok")
        self.assertEqual(chunk_rec["nh"], "3")
        fe, fd, fs = (int(chunk_rec[k]) for k in ("fe", "fd", "fs"))
        self.assertEqual((fe, fd), (10, 12))
        self.assertGreater(fs, fd)
        te, td, tp, ts = (int(chunk_rec[k]) for k in ("te", "td", "tp", "ts"))
        self.assertTrue(0 <= te <= td <= tp <= ts)

        stacks = {line.split("|", 1)[0]: line.split("|", 1)[1] for line in meta["#S"]}
        innermost = stacks[chunk_rec["S"]].split(";")[0]
        self.assertIn(
            "test_chunk_trace_accounts_what_was_posted_and_who_initiated_it",
            innermost,
        )

        handles = {r["k"]: r for r in records["H"]}
        self.assertEqual(set(handles), {"kv", "st.dsa", "aux"})
        for rec in handles.values():
            self.assertEqual(rec["hs"], "DONE")
            self.assertTrue(int(rec["t0"]) <= int(rec["t1"]) <= int(rec["tD"]))

        (_, src_idx, _) = agent.prepped[0]
        self.assertEqual(int(handles["kv"]["nd"]), len(src_idx))
        self.assertEqual(
            int(handles["kv"]["B"]), len(kv_pages) * sum(mgr.kv_args.kv_item_lens)
        )
        self.assertEqual(
            handles["kv"]["dc"], f"{zlib.crc32(np.int32([7, 8, 9, 20])):08x}"
        )

        # Non-prepped posts: the source descriptor list is every other
        # get_xfer_descs call (src, then dst).
        dsa_src, aux_src = agent.descs[0], agent.descs[2]
        self.assertEqual(int(handles["st.dsa"]["nd"]), len(dsa_src))
        self.assertEqual(int(handles["st.dsa"]["B"]), int(dsa_src[:, 1].sum()))
        self.assertEqual(int(handles["aux"]["B"]), int(aux_src[:, 1].sum()))
        self.assertEqual((handles["aux"]["si"], handles["aux"]["di"]), ("2", "5"))


class TestDecodeReceiverTrace(CustomTestCase):
    def test_receiver_reports_once_with_per_pp_notification_counts(self):
        """RXDONE fires when the last PP rank's notifications land, carries the
        per-rank counts, and the success-path clear() does not report again."""
        notifs = [f"{ROOM}_kv_0_1_0", f"{ROOM}_kv_0_1_1", f"{ROOM}_aux"]
        agent = SimpleNamespace(
            name="decode-agent",
            get_new_notifs=MagicMock(
                side_effect=[
                    {"p0": [notifs[0].encode()]},
                    {"p1": [m.encode() for m in notifs[1:]]},
                ]
            ),
        )
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent
        mgr.local_ip = "10.0.0.2"
        mgr.rank_port = 6000
        mgr.is_mla_backend = True
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = {ROOM: 2}
        mgr.waiting_timeout = 300
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            gpu_id=1,
            system_dp_rank=0,
            page_size=PAGE,
            kv_cache_dtype_str="fp8_e4m3",
            kv_data_ptrs=[],
            kv_data_lens=[],
            kv_item_lens=[],
            num_draft_entries=0,
            kv_data_mem_kinds=[],
            state_types=[],
            aux_data_ptrs=[],
            aux_data_lens=[],
            aux_item_lens=[],
        )
        receiver = object.__new__(NixlKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = ROOM
        receiver.bootstrap_addr = "10.0.0.1:8998"
        receiver.conclude_state = None
        receiver.started_transfer = False
        receiver.required_prefill_response_num = 2
        receiver.required_dst_info_num = 1
        receiver.bootstrap_infos = [
            {
                "rank_ip": "10.0.0.1",
                "rank_port": 5000 + pp,
                "is_dummy": False,
                "pp_rank": pp,
            }
            for pp in (0, 1)
        ]
        sock = SimpleNamespace(send_multipart=MagicMock())

        with tempfile.TemporaryDirectory() as trace_dir, ExitStack() as stack:
            _tracer_env(stack, trace_dir)
            tracer = KVXferTracer()
            stack.enter_context(patch.object(nixl_conn, "_KVXT", tracer))
            stack.enter_context(
                patch.object(
                    NixlKVReceiver,
                    "_connect_to_bootstrap_server",
                    return_value=(sock, threading.Lock()),
                )
            )
            counter = _bind(tracer, mgr, "D")

            counter.fc = 3
            receiver.send_metadata(np.array([7, 8, 9, 20], dtype=np.int32), 5)
            counter.fc = 4
            self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
            counter.fc = 6
            self.assertEqual(receiver.poll(), KVPoll.Success)
            self.assertEqual(receiver.poll(), KVPoll.Success)
            receiver.clear()
            tracer.close()
            records, _ = _read_stream(tracer._path)

        (meta,) = records["META"]
        # The publish stamp is taken before the sends, so it precedes the record and
        # anchors the receive timeline; cross-node clock bounds rely on that order.
        self.assertLessEqual(int(meta["ts"]), meta["t_us"])
        self.assertEqual(meta["dc"], f"{zlib.crc32(np.int32([7, 8, 9, 20])):08x}")
        self.assertEqual(meta["tgt"], "10.0.0.1:5000:0:0,10.0.0.1:5001:1:0")
        (done,) = records["RXDONE"]
        self.assertEqual(done["st"], "ok")
        self.assertEqual(done["tm"], meta["ts"])
        self.assertEqual((done["nk"], done["pp"], done["np"]), ("2", "0:1/1:1", "2"))
        self.assertEqual((done["fm"], done["fk0"], done["fk1"]), ("3", "4", "6"))
        self.assertTrue(
            int(done["tm"]) <= int(done["tk0"]) <= int(done["tk1"]) <= int(done["ta"])
        )


class TestStreamCap(CustomTestCase):
    def test_capped_stream_stays_readable_and_counts_drops(self):
        """Past the byte cap records are dropped and counted, while the stream
        keeps its close marker and stays a valid gzip."""
        mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                engine_rank=0,
                gpu_id=0,
                system_dp_rank=0,
                page_size=PAGE,
                kv_cache_dtype_str="auto",
                kv_data_ptrs=[],
                kv_data_lens=[],
                kv_item_lens=[],
                num_draft_entries=0,
                kv_data_mem_kinds=[],
                state_types=[],
                aux_data_ptrs=[],
                aux_data_lens=[],
                aux_item_lens=[],
            ),
            local_ip="10.0.0.2",
            rank_port=6000,
            agent=SimpleNamespace(name="decode-agent"),
            is_mla_backend=True,
            request_status={},
            transfer_statuses={},
        )
        with tempfile.TemporaryDirectory() as trace_dir, ExitStack() as stack:
            _tracer_env(stack, trace_dir, SGLANG_KV_XFER_TRACE_MAX_BYTES=900)
            tracer = KVXferTracer()
            _bind(tracer, mgr, "D")
            for i in range(200):
                tracer.req_event("ADMIT", room=i, rid=f"rid-{i}")
                if i % 50 == 0:
                    tracer._flush()
            tracer.close()
            records, meta = _read_stream(tracer._path)
            self.assertTrue(os.path.getsize(tracer._path) > 0)

        self.assertEqual(len(meta["#C"]), 1)
        (end,) = meta["#E"]
        end = _fields(end)
        written = sum(len(recs) for recs in records.values())
        self.assertEqual(written, int(end["emitted"]) - int(end["dropped"]))
        self.assertGreater(len(records["ADMIT"]), 0)
        self.assertLess(len(records["ADMIT"]), 200)


if __name__ == "__main__":
    unittest.main()
