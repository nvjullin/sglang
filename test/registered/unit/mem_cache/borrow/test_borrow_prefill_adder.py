"""BorrowPrefillAdder against a stub cache client: the prefill delayer gates
admission as in PrefillAdder."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.borrow_cache.adder import BorrowPrefillAdder
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.borrow.api import AcquireResult, AcquireStatus
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase


class _RecordingDelayer:
    """Stand-in for PrefillDelayerSinglePassExecutor: records local_prefillable
    of every negotiation and returns a fixed verdict."""

    def __init__(self, allow: bool):
        self.allow = allow
        self.calls = []

    def negotiate_should_allow_prefill(self, local_prefillable, **kwargs):
        self.calls.append(local_prefillable)
        return self.allow


def _granted(total_len: int) -> AcquireResult:
    return AcquireResult(status=AcquireStatus.GRANTED, lease=1, total_len=total_len)


def _req(num_tokens: int, rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(num_tokens)),
        prefix_len=0,
        retracted_stain=False,
    )


class TestBorrowPrefillAdderDelayer(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.client = MagicMock()
        self.tree_cache = MagicMock()
        self.tree_cache.client = self.client

    def _adder(self, delayer, *, rem_chunk_tokens=None) -> BorrowPrefillAdder:
        running_batch = MagicMock()
        running_batch.reqs = []
        running_batch.batch_size.return_value = 0
        allocator = MagicMock()
        allocator.page_size = 1
        return BorrowPrefillAdder(
            page_size=1,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=rem_chunk_tokens,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
            prefill_delayer_single_pass=delayer,
        )

    def test_delay_verdict_releases_the_granted_lease(self):
        delayer = _RecordingDelayer(allow=False)
        adder = self._adder(delayer)
        req = _req(50)
        self.client.open.return_value = _granted(50)

        result = adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(delayer.calls, [True])
        self.assertEqual(adder.can_run_list, [])
        self.client.release.assert_called_once_with(req, publish_token_ids=None)

    def test_allow_verdict_admits(self):
        delayer = _RecordingDelayer(allow=True)
        adder = self._adder(delayer)
        req = _req(50)
        self.client.open.return_value = _granted(50)

        result = adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(delayer.calls, [True])
        self.assertEqual(adder.can_run_list, [req])
        self.client.release.assert_not_called()

    def test_refused_candidate_is_not_reported_prefillable(self):
        delayer = _RecordingDelayer(allow=True)
        adder = self._adder(delayer)
        self.client.open.return_value = AcquireResult(status=AcquireStatus.REFUSED)

        result = adder.add_one_req(_req(50), has_chunked_req=False, truncation_align_size=None)

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(delayer.calls, [])

    def test_resumed_chunk_runs_whatever_the_verdict(self):
        delayer = _RecordingDelayer(allow=False)
        adder = self._adder(delayer, rem_chunk_tokens=500)
        req = _req(200)
        self.client.grow_chunk.return_value = _granted(200)

        self.assertIsNone(adder.add_chunked_req(req))
        self.assertEqual(delayer.calls, [True])
        self.assertEqual(adder.can_run_list, [req])


if __name__ == "__main__":
    unittest.main()
