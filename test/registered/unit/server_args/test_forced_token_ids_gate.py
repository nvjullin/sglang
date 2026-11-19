"""SGLANG_ENABLE_FORCED_TOKEN_IDS refuses, at startup, every config whose tokens
would not go through the forcing hooks (the sampler and the EAGLE verify).

    python -m pytest test/registered/unit/server_args/test_forced_token_ids_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest import mock

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
    is_encoder_decoder=False,
    sampling_backend=None,
    megakernel=False,
    msa_served=False,
    monokernel=False,
    simulate_acc_len=-1.0,
    use_mlx=False,
    rust_server=False,
):
    sa = ServerArgs(model_path="dummy")
    for name, value in {
        "speculative_algorithm": algorithm,
        "speculative_use_rejection_sampling": rejection_sampling,
        "dllm_algorithm": dllm_algorithm,
        "sampling_backend": sampling_backend,
        "enable_nvidia_megakernel": megakernel,
        "enable_amd_monokernel": monokernel,
    }.items():
        msgspec.structs.force_setattr(sa, name, value)
    sa._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
        is_encoder_decoder=is_encoder_decoder,
    )
    with (
        envs.SGLANG_ENABLE_FORCED_TOKEN_IDS.override(enabled),
        envs.SGLANG_SIMULATE_ACC_LEN.override(simulate_acc_len),
        envs.SGLANG_USE_MLX.override(use_mlx),
        envs.SGLANG_RUST_SERVER.override(rust_server),
        mock.patch(
            "sglang.srt.kernel_plugins.amd.binding.msa_serves",
            return_value=msa_served,
        ),
    ):
        validate_forced_token_ids(sa)


class TestForcedTokenIdsGate(CustomTestCase):
    def test_accepts_forced_paths(self):
        for algorithm in (None, "EAGLE", "EAGLE3"):
            with self.subTest(algorithm=algorithm):
                _validate(algorithm=algorithm)
        for backend in ("flashinfer", "pytorch"):
            with self.subTest(sampling_backend=backend):
                _validate(sampling_backend=backend)
        # sgl-nvidia-mega returns logits to the sampler and eagle_sample.
        with self.subTest("--enable-nvidia-megakernel"):
            _validate(algorithm="EAGLE", megakernel=True)

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
            "encoder-decoder": dict(is_encoder_decoder=True),
            "registered sampling backend": dict(sampling_backend="token_oracle"),
            "MSA-served": dict(megakernel=True, msa_served=True),
            "--enable-amd-monokernel": dict(monokernel=True),
            "MLX": dict(use_mlx=True),
            "SGLANG_RUST_SERVER": dict(rust_server=True),
        }
        for reason, kwargs in cases.items():
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    _validate(**kwargs)

    def test_disabled_feature_rejects_nothing(self):
        _validate(enabled=False, algorithm="NGRAM", rejection_sampling=True)


if __name__ == "__main__":
    unittest.main()
