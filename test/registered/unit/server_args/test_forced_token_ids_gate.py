"""SGLANG_ENABLE_FORCED_TOKEN_IDS refuses, at startup, every config whose tokens
would not go through the forcing hooks (the sampler and the EAGLE verify).

    python -m pytest test/registered/unit/server_args/test_forced_token_ids_gate.py -v
"""

import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.arg_groups.validation_hook import validate_forced_token_ids
from sglang.srt.configs.model_config import UNLIMITED_OCR_V2_ARCH
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _validate(
    *,
    enabled=True,
    algorithm=None,
    rejection_sampling=False,
    dllm_algorithm=None,
    architecture="LlamaForCausalLM",
    simulate_acc_len=-1.0,
):
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "speculative_algorithm": algorithm,
        "speculative_use_rejection_sampling": rejection_sampling,
        "dllm_algorithm": dllm_algorithm,
    }.items():
        msgspec.structs.force_setattr(sa, name, value)
    sa._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture])
    )
    with (
        envs.SGLANG_ENABLE_FORCED_TOKEN_IDS.override(enabled),
        envs.SGLANG_SIMULATE_ACC_LEN.override(simulate_acc_len),
    ):
        validate_forced_token_ids(sa)


class TestForcedTokenIdsGate(CustomTestCase):
    def test_accepts_forced_paths(self):
        for algorithm in (None, "EAGLE", "EAGLE3"):
            with self.subTest(algorithm=algorithm):
                _validate(algorithm=algorithm)

    def test_rejects_unforced_paths(self):
        cases = {
            "STANDALONE": dict(algorithm="STANDALONE"),
            "NGRAM": dict(algorithm="NGRAM"),
            "DFLASH": dict(algorithm="DFLASH"),
            "FROZEN_KV_MTP": dict(algorithm="FROZEN_KV_MTP"),
            "rejection sampling": dict(algorithm="EAGLE", rejection_sampling=True),
            "SGLANG_SIMULATE_ACC_LEN": dict(algorithm="EAGLE", simulate_acc_len=3.6),
            "--dllm-algorithm": dict(dllm_algorithm="LowConfidence"),
            "Unlimited-OCR v2": dict(architecture=UNLIMITED_OCR_V2_ARCH),
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    _validate(**kwargs)

    def test_disabled_feature_rejects_nothing(self):
        _validate(enabled=False, algorithm="NGRAM", rejection_sampling=True)


if __name__ == "__main__":
    unittest.main()
