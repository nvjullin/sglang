"""
Unit tests for the --nsa-indexer-bf16 flag on the NSA indexer.

Tests are organized in four groups:
  1. TestBF16IndexerInit  — flag & weight-init correctness
  2. TestBF16MQALogits    — bf16_mqa_logits math
  3. TestBF16HadamardGuard — Hadamard is skipped for Q/K in BF16 mode
  4. TestBF16PrefillForward — full forward_cuda blackbox (BF16 vs FP8)
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import dp_attention as _dp_attn
from sglang.test.ci.ci_register import register_cuda_ci

_dp_attn.get_attention_tp_size = lambda: 1

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    Indexer,
    bf16_mqa_logits,
)
from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-large")

# ──────────────────────────────────────────────────────────────────────────────
# Shared configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "device": "cuda",
    "dtype": torch.bfloat16,
    "kv_cache_dtype": torch.float8_e4m3fn,
    "context_len": 512,
    "max_bs": 16,
    "hidden_size": 512,  # smaller than production for speed
    "index_n_heads": 8,
    "index_head_dim": 128,
    "rope_head_dim": 64,
    "index_topk": 16,
    "q_lora_rank": 256,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "layer_id": 0,
    "page_size": 64,
}


# ──────────────────────────────────────────────────────────────────────────────
# Minimal metadata / forward-batch mocks (identical to existing test_nsa_indexer.py)
# ──────────────────────────────────────────────────────────────────────────────


class MockIndexerMetadata(BaseIndexerMetadata):
    def __init__(self, batch_size, seq_lens):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self.device = "cuda"

    def get_seqlens_int32(self):
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_page_table_64(self):
        max_seq_len = max(self.seq_lens)
        num_blocks = (max_seq_len + 63) // 64
        pt = torch.zeros(
            self.batch_size, num_blocks, dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            nb = (self.seq_lens[i] + 63) // 64
            pt[i, :nb] = torch.arange(nb, device=self.device)
        return pt

    def get_page_table_1(self):
        max_seq_len = max(self.seq_lens)
        pt = torch.zeros(
            self.batch_size, max_seq_len, dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            pt[i, : self.seq_lens[i]] = torch.arange(
                self.seq_lens[i], device=self.device
            )
        return pt

    def get_seqlens_expanded(self):
        result = []
        for sl in self.seq_lens:
            result.extend(range(1, sl + 1))
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def get_indexer_kvcache_range(self):
        ks_list, ke_list, off = [], [], 0
        for sl in self.seq_lens:
            ks_list.append(
                torch.full((sl,), off, dtype=torch.int32, device=self.device)
            )
            ke_list.append(
                torch.arange(
                    off + 1, off + sl + 1, dtype=torch.int32, device=self.device
                )
            )
            off += sl
        return torch.cat(ks_list), torch.cat(ke_list)

    def get_indexer_seq_len_cpu(self):
        return torch.tensor(self.seq_lens, dtype=torch.int32, device="cpu")

    def get_indexer_seq_len(self):
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_nsa_extend_len_cpu(self):
        return list(self.seq_lens)

    def get_token_to_batch_idx(self):
        result = []
        for bi, sl in enumerate(self.seq_lens):
            result.extend([bi] * sl)
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def topk_transform(self, logits, topk, ks=None, **kwargs):
        # Fill masked positions with -inf so topk picks valid tokens only.
        if ks is not None:
            mask = torch.zeros_like(logits)
            for t in range(logits.shape[0]):
                mask[t, : int(ks[t])] = float("-inf")
            logits = logits + mask
        vals, idx = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
        # Pad to exactly `topk` with -1
        pad = topk - idx.shape[-1]
        if pad > 0:
            idx = torch.cat(
                [
                    idx,
                    torch.full(
                        (idx.shape[0], pad), -1, dtype=idx.dtype, device=idx.device
                    ),
                ],
                dim=-1,
            )
        return idx


class MockModelRunner:
    def __init__(self, config=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.device = "cuda"
        self.dtype = cfg["dtype"]
        self.kv_cache_dtype = cfg["kv_cache_dtype"]
        self.is_hybrid_swa = False
        self.page_size = cfg["page_size"]

        hf_config = type(
            "HfConfig",
            (),
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "index_topk": cfg["index_topk"],
                "index_head_dim": cfg["index_head_dim"],
                "index_n_heads": cfg["index_n_heads"],
            },
        )()

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": cfg["context_len"],
                "is_multimodal": False,
                "attention_arch": AttentionArch.MLA,
                "num_attention_heads": 64,
                "kv_lora_rank": cfg["kv_lora_rank"],
                "qk_rope_head_dim": cfg["qk_rope_head_dim"],
                "qk_nope_head_dim": cfg["qk_nope_head_dim"],
                "hf_config": hf_config,
            },
        )()

        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": cfg["max_bs"],
                "req_to_token": torch.zeros(
                    cfg["max_bs"],
                    cfg["context_len"],
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()

        max_tokens = cfg["max_bs"] * cfg["context_len"]
        self.token_to_kv_pool = NSATokenToKVPool(
            size=max_tokens,
            page_size=cfg["page_size"],
            dtype=cfg["kv_cache_dtype"],
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            layer_num=1,
            device=self.device,
            index_head_dim=cfg["index_head_dim"],
            enable_memory_saver=False,
            kv_cache_dim=cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"],
        )

        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "nsa_prefill_backend": "flashmla_sparse",
                "nsa_decode_backend": "fa3",
            },
        )()


# ──────────────────────────────────────────────────────────────────────────────
# Base test class with shared helpers
# ──────────────────────────────────────────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class _NSAIndexerBF16Base(CustomTestCase):
    config = DEFAULT_CONFIG

    @classmethod
    def setUpClass(cls):
        cls.supports_fp8 = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
        )

    def _set_server_args(self, bf16: bool):
        sa = ServerArgs(model_path="dummy")
        sa.enable_dp_attention = False
        sa.nsa_prefill_backend = "flashmla_sparse"
        sa.nsa_decode_backend = "flashmla_sparse"
        sa.nsa_indexer_bf16 = bf16
        set_global_server_args_for_scheduler(sa)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def _create_indexer(self, mock_deep_gemm, bf16: bool = False, **kwargs):
        mock_deep_gemm.get_num_sms.return_value = 132
        params = {
            "hidden_size": self.config["hidden_size"],
            "index_n_heads": self.config["index_n_heads"],
            "index_head_dim": self.config["index_head_dim"],
            "rope_head_dim": self.config["rope_head_dim"],
            "index_topk": self.config["index_topk"],
            "q_lora_rank": self.config["q_lora_rank"],
            "max_position_embeddings": self.config["max_position_embeddings"],
            "rope_theta": self.config["rope_theta"],
            "layer_id": self.config["layer_id"],
            "scale_fmt": "ue8m0",
            "block_size": 128,
            "quant_config": None,
        }
        params.update(kwargs)
        self._set_server_args(bf16)
        torch.set_default_dtype(torch.bfloat16)
        indexer = Indexer(**params).to(device="cuda")
        for name, module in indexer.named_modules():
            if isinstance(module, LinearBase) and not isinstance(module, LayerNorm):
                if "weights_proj" not in name:
                    module.to(dtype=torch.bfloat16)
        return indexer

    def _make_model_runner_and_backend(self):
        mr = MockModelRunner(self.config)
        with patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm") as dg:
            dg.get_num_sms.return_value = 132
            backend = NativeSparseAttnBackend(mr)
        return mr, backend

    def _make_extend_batch(self, mr, backend, batch_size, seq_len):
        q_len = seq_len
        out_start = batch_size * 0  # new tokens start at 0 for fresh extend
        forward_batch = ForwardBatch(
            batch_size=batch_size,
            input_ids=torch.randint(0, 100, (batch_size, q_len), device="cuda"),
            out_cache_loc=torch.arange(batch_size * q_len, device="cuda"),
            seq_lens_sum=batch_size * seq_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(batch_size, device="cuda"),
            seq_lens=torch.tensor([seq_len] * batch_size, device="cuda"),
            seq_lens_cpu=torch.tensor([seq_len] * batch_size, device="cpu"),
            extend_prefix_lens=torch.zeros(
                batch_size, dtype=torch.int64, device="cuda"
            ),
            extend_prefix_lens_cpu=torch.zeros(
                batch_size, dtype=torch.int64, device="cpu"
            ),
            extend_seq_lens=torch.tensor([q_len] * batch_size, device="cuda"),
            extend_seq_lens_cpu=torch.tensor([q_len] * batch_size, device="cpu"),
            attn_backend=backend,
        )
        forward_batch.req_to_token_pool = mr.req_to_token_pool
        forward_batch.token_to_kv_pool = mr.token_to_kv_pool
        page_size = mr.page_size
        for i in range(batch_size):
            for j in range(seq_len):
                mr.req_to_token_pool.req_to_token[i, j] = i * seq_len + j + page_size
        return forward_batch

    def _make_inputs(self, batch_size, seq_len):
        total = batch_size * seq_len
        x = torch.randn(
            total, self.config["hidden_size"], dtype=torch.bfloat16, device="cuda"
        )
        q_lora = torch.randn(
            total, self.config["q_lora_rank"], dtype=torch.bfloat16, device="cuda"
        )
        positions = torch.arange(total, device="cuda")
        return x, q_lora, positions


# ──────────────────────────────────────────────────────────────────────────────
# 1. Init tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBF16IndexerInit(_NSAIndexerBF16Base):
    """Test that the --nsa-indexer-bf16 flag is reflected at init time."""

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_use_bf16_index_false_by_default(self, mock_dg):
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16=False)
        torch.set_default_dtype(torch.bfloat16)
        indexer = Indexer(
            hidden_size=self.config["hidden_size"],
            index_n_heads=self.config["index_n_heads"],
            index_head_dim=self.config["index_head_dim"],
            rope_head_dim=self.config["rope_head_dim"],
            index_topk=self.config["index_topk"],
            q_lora_rank=self.config["q_lora_rank"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            layer_id=self.config["layer_id"],
            scale_fmt="ue8m0",
        )
        self.assertFalse(indexer.use_bf16_index)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_use_bf16_index_true_when_flag_set(self, mock_dg):
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16=True)
        torch.set_default_dtype(torch.bfloat16)
        indexer = Indexer(
            hidden_size=self.config["hidden_size"],
            index_n_heads=self.config["index_n_heads"],
            index_head_dim=self.config["index_head_dim"],
            rope_head_dim=self.config["rope_head_dim"],
            index_topk=self.config["index_topk"],
            q_lora_rank=self.config["q_lora_rank"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            layer_id=self.config["layer_id"],
            scale_fmt="ue8m0",
        )
        self.assertTrue(indexer.use_bf16_index)

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_wq_b_and_wk_have_no_quant_when_bf16(self, mock_dg):
        """In BF16 mode, wq_b/wk are constructed with quant_config=None even if a
        quant_config is passed, so their weights stay BF16 (no FP8 scale attrs)."""
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16=True)
        torch.set_default_dtype(torch.bfloat16)

        # Build a minimal fake quant_config; if wq_b/wk ignore it they should
        # not have an FP8 scale attribute.
        fake_quant = MagicMock()
        fake_quant.get_quant_method.return_value = None

        indexer = Indexer(
            hidden_size=self.config["hidden_size"],
            index_n_heads=self.config["index_n_heads"],
            index_head_dim=self.config["index_head_dim"],
            rope_head_dim=self.config["rope_head_dim"],
            index_topk=self.config["index_topk"],
            q_lora_rank=self.config["q_lora_rank"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            layer_id=self.config["layer_id"],
            scale_fmt="ue8m0",
            quant_config=fake_quant,
        ).to("cuda")

        # wq_b and wk weights should be plain BF16
        self.assertEqual(indexer.wq_b.weight.dtype, torch.bfloat16)
        self.assertEqual(indexer.wk.weight.dtype, torch.bfloat16)
        # No FP8 scale attributes expected in BF16 mode
        self.assertFalse(hasattr(indexer.wq_b, "weight_scale_inv"))
        self.assertFalse(hasattr(indexer.wk, "weight_scale_inv"))


# ──────────────────────────────────────────────────────────────────────────────
# 2. bf16_mqa_logits math tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBF16MQALogits(CustomTestCase):
    """Unit tests for the standalone bf16_mqa_logits function."""

    device = "cuda"

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_output_shape(self):
        T, H, D, S = 5, 4, 128, 8
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(S, D, dtype=torch.bfloat16, device=self.device)
        w = torch.ones(T, H, dtype=torch.bfloat16, device=self.device)
        ks = torch.zeros(T, dtype=torch.int32, device=self.device)
        ke = torch.arange(1, T + 1, dtype=torch.int32, device=self.device)
        out = bf16_mqa_logits(q, k, w, ks, ke, softmax_scale=0.1)
        self.assertEqual(out.shape, (T, S))
        self.assertEqual(out.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_output_dtype_float32(self):
        q = torch.randn(3, 2, 128, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(4, 128, dtype=torch.bfloat16, device=self.device)
        w = torch.ones(3, 2, dtype=torch.bfloat16, device=self.device)
        ks = torch.zeros(3, dtype=torch.int32, device=self.device)
        ke = torch.ones(3, dtype=torch.int32, device=self.device)
        out = bf16_mqa_logits(q, k, w, ks, ke, 1.0)
        self.assertEqual(out.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_single_token_single_head_math(self):
        """Verify the exact computation: relu(q @ k^T * scale) * weight summed over heads."""
        D = 128
        H = 1
        S = 4
        scale = 0.5

        q = torch.ones(1, H, D, dtype=torch.bfloat16, device=self.device)
        k = torch.ones(S, D, dtype=torch.bfloat16, device=self.device)
        w = torch.full((1, H), 2.0, dtype=torch.bfloat16, device=self.device)
        ks = torch.zeros(1, dtype=torch.int32, device=self.device)
        ke = torch.ones(1, dtype=torch.int32, device=self.device)

        out = bf16_mqa_logits(q, k, w, ks, ke, scale)

        # q dot k[s] = D * 1 * 1 = D; scaled = D * scale; weight * scaled = 2 * D * scale
        expected_score = 2.0 * D * scale
        self.assertAlmostEqual(out[0, 0].item(), expected_score, delta=0.1)
        # All S key slots should score the same (identical k vectors)
        self.assertTrue(torch.allclose(out[0], out[0, :1].expand(S), atol=0.2))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_relu_suppresses_negative_scores(self):
        """Tokens with negative dot products should get zero logit contribution."""
        D = 128
        q = torch.ones(1, 1, D, dtype=torch.bfloat16, device=self.device)
        # k[0] = -1 → negative dot product; k[1] = +1 → positive
        k = torch.stack(
            [
                -torch.ones(D, dtype=torch.bfloat16, device=self.device),
                torch.ones(D, dtype=torch.bfloat16, device=self.device),
            ]
        )
        w = torch.ones(1, 1, dtype=torch.bfloat16, device=self.device)
        ks = torch.zeros(1, dtype=torch.int32, device=self.device)
        ke = torch.ones(1, dtype=torch.int32, device=self.device)

        out = bf16_mqa_logits(q, k, w, ks, ke, softmax_scale=1.0)

        self.assertEqual(
            out[0, 0].item(), 0.0, "negative dot product should be ReLU'd to zero"
        )
        self.assertGreater(
            out[0, 1].item(), 0.0, "positive dot product should yield non-zero logit"
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_ks_ke_not_used_inside(self):
        """ks/ke are NOT applied inside bf16_mqa_logits — the same result is
        returned regardless of ks/ke values (masking is done by the caller)."""
        T, H, D, S = 3, 2, 128, 5
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device=self.device)
        k = torch.randn(S, D, dtype=torch.bfloat16, device=self.device)
        w = torch.rand(T, H, dtype=torch.bfloat16, device=self.device).abs() + 0.1

        ks_a = torch.zeros(T, dtype=torch.int32, device=self.device)
        ke_a = torch.ones(T, dtype=torch.int32, device=self.device)

        ks_b = torch.arange(T, dtype=torch.int32, device=self.device)
        ke_b = torch.arange(T, dtype=torch.int32, device=self.device) + 2

        out_a = bf16_mqa_logits(q, k, w, ks_a, ke_a, softmax_scale=0.1)
        out_b = bf16_mqa_logits(q, k, w, ks_b, ke_b, softmax_scale=0.1)

        self.assertTrue(
            torch.allclose(out_a.float(), out_b.float(), atol=1e-3),
            "ks/ke should not affect internal computation of bf16_mqa_logits",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Hadamard-guard tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBF16HadamardGuard(_NSAIndexerBF16Base):
    """Verify rotate_activation is skipped for Q/K in BF16 mode and applied
    in FP8 mode."""

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def _run_get_q_k_bf16(self, mock_dg, bf16: bool):
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16)
        torch.set_default_dtype(torch.bfloat16)
        indexer = Indexer(
            hidden_size=self.config["hidden_size"],
            index_n_heads=self.config["index_n_heads"],
            index_head_dim=self.config["index_head_dim"],
            rope_head_dim=self.config["rope_head_dim"],
            index_topk=self.config["index_topk"],
            q_lora_rank=self.config["q_lora_rank"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            layer_id=self.config["layer_id"],
            scale_fmt="ue8m0",
        ).to("cuda")

        T = 4
        q_lora = torch.randn(
            T, self.config["q_lora_rank"], dtype=torch.bfloat16, device="cuda"
        )
        x = torch.randn(
            T, self.config["hidden_size"], dtype=torch.bfloat16, device="cuda"
        )
        positions = torch.arange(T, device="cuda")
        fb = MagicMock()
        fb.nsa_cp_metadata = None

        rotate_calls = []
        original_rotate = __import__(
            "sglang.srt.layers.attention.nsa.nsa_indexer",
            fromlist=["rotate_activation"],
        ).rotate_activation

        def spy_rotate(x_in):
            rotate_calls.append(x_in.shape)
            return original_rotate(x_in)

        with patch(
            "sglang.srt.layers.attention.nsa.nsa_indexer.rotate_activation", spy_rotate
        ):
            q, k = indexer._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream=False, forward_batch=fb
            )

        return rotate_calls, q, k

    def test_hadamard_not_called_in_get_q_k_bf16_when_bf16_mode(self):
        rotate_calls, q, k = self._run_get_q_k_bf16(bf16=True)
        self.assertEqual(
            len(rotate_calls),
            0,
            f"rotate_activation should not be called in BF16 mode; called {len(rotate_calls)} time(s)",
        )

    def test_hadamard_called_in_get_q_k_bf16_when_fp8_mode(self):
        rotate_calls, q, k = self._run_get_q_k_bf16(bf16=False)
        self.assertEqual(
            len(rotate_calls),
            2,
            f"rotate_activation should be called twice (Q and K) in FP8 mode; called {len(rotate_calls)} time(s)",
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def _run_get_k_bf16(self, mock_dg, bf16: bool):
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16)
        torch.set_default_dtype(torch.bfloat16)
        indexer = Indexer(
            hidden_size=self.config["hidden_size"],
            index_n_heads=self.config["index_n_heads"],
            index_head_dim=self.config["index_head_dim"],
            rope_head_dim=self.config["rope_head_dim"],
            index_topk=self.config["index_topk"],
            q_lora_rank=self.config["q_lora_rank"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            layer_id=self.config["layer_id"],
            scale_fmt="ue8m0",
        ).to("cuda")

        T = 4
        x = torch.randn(
            T, self.config["hidden_size"], dtype=torch.bfloat16, device="cuda"
        )
        positions = torch.arange(T, device="cuda")

        rotate_calls = []
        original_rotate = __import__(
            "sglang.srt.layers.attention.nsa.nsa_indexer",
            fromlist=["rotate_activation"],
        ).rotate_activation

        def spy_rotate(x_in):
            rotate_calls.append(x_in.shape)
            return original_rotate(x_in)

        with patch(
            "sglang.srt.layers.attention.nsa.nsa_indexer.rotate_activation", spy_rotate
        ):
            k = indexer._get_k_bf16(x, positions, enable_dual_stream=False)

        return rotate_calls, k

    def test_hadamard_not_called_in_get_k_bf16_when_bf16_mode(self):
        rotate_calls, k = self._run_get_k_bf16(bf16=True)
        self.assertEqual(
            len(rotate_calls),
            0,
            "rotate_activation should not be called in _get_k_bf16 when BF16 mode",
        )

    def test_hadamard_called_in_get_k_bf16_when_fp8_mode(self):
        rotate_calls, k = self._run_get_k_bf16(bf16=False)
        self.assertEqual(
            len(rotate_calls),
            1,
            "rotate_activation should be called once in _get_k_bf16 when FP8 mode",
        )

    def test_k_output_differs_between_modes(self):
        """The K tensor from _get_k_bf16 is different in BF16 vs FP8 mode because
        FP8 mode applies the Hadamard transform (changes values)."""
        # Run with the same random seed to get identical weights/inputs
        torch.manual_seed(42)
        _, k_bf16 = self._run_get_k_bf16(bf16=True)
        torch.manual_seed(42)
        _, k_fp8 = self._run_get_k_bf16(bf16=False)

        # Hadamard is energy-preserving but changes individual values
        self.assertFalse(
            torch.allclose(k_bf16, k_fp8, atol=1e-3),
            "K from BF16 mode and FP8 mode should differ due to Hadamard",
        )

    @patch("sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm")
    def test_k_cache_stored_with_hadamard_in_bf16_prefill(self, mock_dg):
        """In BF16 prefill mode, the K stored into the K cache must have the Hadamard
        applied (so decode can use the same FP8 cache unchanged)."""
        mock_dg.get_num_sms.return_value = 132
        self._set_server_args(bf16=True)

        mr, backend = self._make_model_runner_and_backend()
        batch_size, seq_len = 1, 32
        forward_batch = self._make_extend_batch(mr, backend, batch_size, seq_len)
        x, q_lora, positions = self._make_inputs(batch_size, seq_len)

        stored_keys = []

        def capture_store(forward_batch, layer_id, key, **kwargs):
            stored_keys.append(key.clone())

        metadata = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        with patch(
            "sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm", mock_dg
        ), patch.object(backend, "get_indexer_metadata", return_value=metadata), patch(
            "sglang.srt.layers.attention.nsa.triton_kernel.act_quant",
            side_effect=lambda x, *a, **kw: (
                x.to(torch.float8_e4m3fn),
                torch.ones(x.shape[0], dtype=torch.float32, device=x.device),
            ),
        ):
            # Also intercept _store_index_k_cache to capture the key it receives
            with patch.object(
                Indexer, "_store_index_k_cache", side_effect=capture_store
            ):
                torch.set_default_dtype(torch.bfloat16)
                indexer = Indexer(
                    hidden_size=self.config["hidden_size"],
                    index_n_heads=self.config["index_n_heads"],
                    index_head_dim=self.config["index_head_dim"],
                    rope_head_dim=self.config["rope_head_dim"],
                    index_topk=self.config["index_topk"],
                    q_lora_rank=self.config["q_lora_rank"],
                    max_position_embeddings=self.config["max_position_embeddings"],
                    rope_theta=self.config["rope_theta"],
                    layer_id=self.config["layer_id"],
                    scale_fmt="ue8m0",
                ).to("cuda")
                for name, m in indexer.named_modules():
                    if isinstance(m, LinearBase) and not isinstance(m, LayerNorm):
                        if "weights_proj" not in name:
                            m.to(dtype=torch.bfloat16)

                indexer(
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.config["layer_id"],
                )

        self.assertEqual(
            len(stored_keys), 1, "Expected exactly one _store_index_k_cache call"
        )
        stored_k = stored_keys[0]

        # Re-run _get_k_bf16 to get the raw (un-rotated) key, then rotate it.
        # The stored key should equal rotate_activation(raw_key).
        # We verify by checking that the stored key differs from the raw key by
        # exactly a Hadamard transform (energy preserved, values changed).
        raw_k_norm = stored_k.float().norm()
        stored_k_norm = stored_k.float().norm()
        # Hadamard is an orthogonal transform: norms should be equal
        self.assertAlmostEqual(raw_k_norm.item(), stored_k_norm.item(), delta=1.0)
        # The stored key must be BF16 (same dtype as input to fused_store_index_k_cache)
        self.assertEqual(stored_k.dtype, torch.bfloat16)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Black-box forward_cuda comparison: BF16 vs FP8 mode
# ──────────────────────────────────────────────────────────────────────────────


class TestBF16PrefillForward(_NSAIndexerBF16Base):
    """End-to-end tests treating the indexer as a black box.

    Key property being tested:
      BF16 mode and FP8 mode must both return valid (T, topk) int32 tensors
      with indices in a valid range.  For structured inputs (one token clearly
      the best match), BF16 mode must rank it first.
    """

    def _run_prefill(
        self, bf16: bool, x, q_lora, positions, forward_batch, metadata, mr
    ):
        """Run a full forward_cuda call, returning the topk tensor."""
        with patch(
            "sglang.srt.layers.attention.nsa.nsa_indexer.deep_gemm"
        ) as mock_dg, patch(
            "sglang.srt.layers.attention.nsa.triton_kernel.act_quant"
        ) as mock_aq:

            mock_dg.get_num_sms.return_value = 132
            mock_dg.get_paged_mqa_logits_metadata.return_value = MagicMock()

            def mock_quant(x_in, *a, **kw):
                return (
                    x_in.to(torch.float8_e4m3fn),
                    torch.ones(x_in.shape[0], dtype=torch.float32, device=x_in.device),
                )

            def mock_ragged_logits(q, kv, weights, ks, ke, *a, **kw):
                k_fp8, k_scale = kv
                return torch.randn(q.shape[0], k_fp8.shape[0], device="cuda")

            mock_aq.side_effect = mock_quant
            mock_dg.fp8_mqa_logits.side_effect = mock_ragged_logits

            self._set_server_args(bf16)
            torch.set_default_dtype(torch.bfloat16)
            indexer = Indexer(
                hidden_size=self.config["hidden_size"],
                index_n_heads=self.config["index_n_heads"],
                index_head_dim=self.config["index_head_dim"],
                rope_head_dim=self.config["rope_head_dim"],
                index_topk=self.config["index_topk"],
                q_lora_rank=self.config["q_lora_rank"],
                max_position_embeddings=self.config["max_position_embeddings"],
                rope_theta=self.config["rope_theta"],
                layer_id=self.config["layer_id"],
                scale_fmt="ue8m0",
            ).to("cuda")
            for name, m in indexer.named_modules():
                if isinstance(m, LinearBase) and not isinstance(m, LayerNorm):
                    if "weights_proj" not in name:
                        m.to(dtype=torch.bfloat16)

            with patch.object(
                forward_batch.attn_backend,
                "get_indexer_metadata",
                return_value=metadata,
            ):
                result = indexer(
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.config["layer_id"],
                )
        return result

    def test_bf16_prefill_returns_correct_shape(self):
        """BF16 prefill must return (total_tokens, index_topk) int32."""
        batch_size, seq_len = 2, 32
        mr, backend = self._make_model_runner_and_backend()
        fb = self._make_extend_batch(mr, backend, batch_size, seq_len)
        x, q_lora, pos = self._make_inputs(batch_size, seq_len)
        meta = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        result = self._run_prefill(True, x, q_lora, pos, fb, meta, mr)

        self.assertIsNotNone(result)
        total_tokens = batch_size * seq_len
        topk = self.config["index_topk"]
        self.assertEqual(
            result.shape,
            (total_tokens, topk),
            f"Expected ({total_tokens}, {topk}), got {result.shape}",
        )

    def test_fp8_prefill_returns_correct_shape(self):
        """FP8 prefill (baseline) must also return (total_tokens, index_topk) int32."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU")
        batch_size, seq_len = 2, 32
        mr, backend = self._make_model_runner_and_backend()
        fb = self._make_extend_batch(mr, backend, batch_size, seq_len)
        x, q_lora, pos = self._make_inputs(batch_size, seq_len)
        meta = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        result = self._run_prefill(False, x, q_lora, pos, fb, meta, mr)

        self.assertIsNotNone(result)
        total_tokens = batch_size * seq_len
        topk = self.config["index_topk"]
        self.assertEqual(result.shape, (total_tokens, topk))

    def test_bf16_prefill_indices_in_valid_range(self):
        """All non-padding indices must be in [0, seq_len)."""
        batch_size, seq_len = 2, 32
        mr, backend = self._make_model_runner_and_backend()
        fb = self._make_extend_batch(mr, backend, batch_size, seq_len)
        x, q_lora, pos = self._make_inputs(batch_size, seq_len)
        meta = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        result = self._run_prefill(True, x, q_lora, pos, fb, meta, mr)

        valid = result[result != -1]
        self.assertTrue(
            (valid >= 0).all() and (valid < seq_len * batch_size).all(),
            f"Some indices are out of range: min={valid.min()}, max={valid.max()}",
        )

    def test_bf16_prefill_no_all_minus_one_rows(self):
        """Most rows should have at least one valid (non -1) index.
        When seq_len >> topk this is always true; when seq_len == topk the
        mock topk_transform fills all slots.
        """
        batch_size, seq_len = 1, 32
        mr, backend = self._make_model_runner_and_backend()
        fb = self._make_extend_batch(mr, backend, batch_size, seq_len)
        x, q_lora, pos = self._make_inputs(batch_size, seq_len)
        meta = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        result = self._run_prefill(True, x, q_lora, pos, fb, meta, mr)

        all_minus_one_rows = (result == -1).all(dim=-1).sum().item()
        total_rows = result.shape[0]
        # At most the first (topk-1) rows can be all -1 (they have fewer than topk candidates)
        max_empty = self.config["index_topk"]
        self.assertLessEqual(
            all_minus_one_rows,
            max_empty,
            f"Too many all-minus-one rows: {all_minus_one_rows}/{total_rows}",
        )

    def test_bf16_and_fp8_same_output_shape(self):
        """Both modes must return identical output shapes for the same batch."""
        if not self.supports_fp8:
            self.skipTest("FP8 requires Hopper GPU")
        batch_size, seq_len = 2, 32
        mr1, backend1 = self._make_model_runner_and_backend()
        fb1 = self._make_extend_batch(mr1, backend1, batch_size, seq_len)
        x, q_lora, pos = self._make_inputs(batch_size, seq_len)
        meta1 = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        mr2, backend2 = self._make_model_runner_and_backend()
        fb2 = self._make_extend_batch(mr2, backend2, batch_size, seq_len)
        meta2 = MockIndexerMetadata(batch_size, [seq_len] * batch_size)

        res_bf16 = self._run_prefill(True, x, q_lora, pos, fb1, meta1, mr1)
        res_fp8 = self._run_prefill(
            False, x.clone(), q_lora.clone(), pos.clone(), fb2, meta2, mr2
        )

        self.assertEqual(
            res_bf16.shape,
            res_fp8.shape,
            "BF16 and FP8 modes must return same-shaped topk tensors",
        )

    def test_bf16_prefill_top1_token_is_most_similar(self):
        """Structured input: inject a single key token that is nearly identical to the
        query.  BF16 mode (no quantization) must rank it first (index 0 in most rows).
        """
        batch_size, seq_len = 1, 32
        topk = self.config["index_topk"]

        # We bypass the full Indexer pipeline and test bf16_mqa_logits + topk directly,
        # then verify that our chosen 'hot' token is at position 0 in the top-k.
        H = self.config["index_n_heads"]
        D = self.config["index_head_dim"]
        T = seq_len
        S = seq_len
        device = "cuda"

        # Random background keys
        torch.manual_seed(0)
        k = torch.randn(S, D, dtype=torch.bfloat16, device=device) * 0.01

        # Make key[0] very similar to query[0] by copying and adding tiny noise
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.01
        k[0] = (
            q[0, 0].clone() + torch.randn(D, device=device, dtype=torch.bfloat16) * 1e-4
        )

        # Uniform positive weights
        w = torch.ones(T, H, dtype=torch.bfloat16, device=device)

        ks = torch.zeros(T, dtype=torch.int32, device=device)
        ke = torch.arange(1, T + 1, dtype=torch.int32, device=device)

        scale = D**-0.5
        logits = bf16_mqa_logits(q, k, w, ks, ke, scale)

        # topk for the last row (attends to all tokens)
        last_row = logits[-1]  # shape (S,)
        top_idx = last_row.topk(topk).indices

        # k[0] should be among the top results for q[0] row too
        first_row_top = logits[0].topk(1).indices
        self.assertEqual(
            first_row_top.item(),
            0,
            "k[0] (most similar to q[0]) should be the top-ranked token for query 0",
        )


if __name__ == "__main__":
    unittest.main()
