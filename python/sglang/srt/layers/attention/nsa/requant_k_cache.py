import torch
import triton
import triton.language as tl


def requantize_fp8_to_block_scale(
    kv_cache: torch.Tensor,
    dv: int = 512,
    group_size: int = 128,
) -> torch.Tensor:
    """Transform FP8_NOPE_FP8_ROPE (576 bytes) -> FP8_NOPE_WITH_BLOCK_SCALE_BF16_ROPE (656 bytes).

    Operates on the entire pool (same pattern as quantize_k_cache); the flashmla
    kernel uses indices/cache_seqlens to select relevant pages.

    Input layout per token (576 bytes as fp8):
        [fp8_nope(512) | fp8_rope(64)]

    Output layout per token (656 bytes as fp8):
        [fp8_nope(512) | fp32_scales(16) | bf16_rope(128)]

    The nope part is re-quantized with per-128-element block scales.
    The rope part is cast from fp8 to bf16.
    """
    num_blocks, block_size, h_k, dim_in = kv_cache.shape
    assert h_k == 1
    assert dim_in == dv + 64, f"Expected {dv + 64}, got {dim_in}"

    kv_2d = kv_cache.view(-1, dim_in)
    num_tokens = kv_2d.shape[0]

    dim_nope = dv
    dim_rope = dim_in - dv

    k_nope = kv_2d[:, :dim_nope]
    k_rope = kv_2d[:, dim_nope:]

    num_tiles = dim_nope // group_size
    out_dim = (
        dim_nope + num_tiles * 4 + dim_rope * torch.finfo(torch.bfloat16).bits // 8
    )

    output = torch.empty(
        (num_tokens, out_dim),
        dtype=torch.float8_e4m3fn,
        device=kv_cache.device,
    )
    output_nope_q = output[:, :dim_nope]
    output_nope_s = output[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    output_rope = output[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    assert dim_nope % group_size == 0
    NUM_NOPE_BLOCKS = dim_nope // group_size
    num_blocks_per_token = NUM_NOPE_BLOCKS + triton.cdiv(dim_rope, group_size)

    _requantize_fp8_to_block_scale_kernel[(num_tokens, num_blocks_per_token)](
        output_nope_q,
        output_nope_s,
        output_rope,
        k_nope,
        k_rope,
        output_nope_q.stride(0),
        output_nope_s.stride(0),
        output_rope.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=NUM_NOPE_BLOCKS,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output.view(num_blocks, block_size, 1, -1)


@triton.jit
def _requantize_fp8_to_block_scale_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    """Repack FP8_NOPE_FP8_ROPE into FP8_NOPE_WITH_BLOCK_SCALE_BF16_ROPE.

    Nope: copy FP8 bytes as-is, write scale=1.0 (no precision loss).
    Rope: cast FP8 to BF16.
    """
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # Copy nope FP8 bytes, set block scale = 1.0
        effective_block_id = raw_block_id

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE

        src_ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs
        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = (
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id
        )

        nope_data = tl.load(src_ptr, mask=mask)
        tl.store(dst_q_ptr, nope_data, mask=mask)
        tl.store(dst_s_ptr, 1.0)
    else:
        # Cast rope from FP8 to BF16
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs

        rope_data = tl.load(src_ptr, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(dst_ptr, rope_data, mask=mask)


if __name__ == "__main__":
    print("Testing requantize_fp8_to_block_scale...")

    for num_blocks, block_size in [(1, 1), (10, 64)]:
        dim_nope = 512
        dim_rope = 64
        dim_total = dim_nope + dim_rope
        group_size = 128
        num_tiles = dim_nope // group_size

        # Simulate the write path: bf16 -> fp8 (FP8_NOPE_FP8_ROPE)
        original_bf16 = torch.randn(
            (num_blocks, block_size, 1, dim_total),
            dtype=torch.bfloat16,
            device="cuda",
        )
        fp8_cache = original_bf16.to(torch.float8_e4m3fn)

        # Run kernel
        result = requantize_fp8_to_block_scale(fp8_cache)

        # Expected output dim: 512 (nope fp8) + 16 (4 fp32 scales) + 128 (64 bf16 rope)
        expected_out_dim = dim_nope + num_tiles * 4 + dim_rope * 2
        assert result.shape == (
            num_blocks,
            block_size,
            1,
            expected_out_dim,
        ), f"Shape: expected (..., {expected_out_dim}), got {result.shape}"

        # Verify layout by unpacking the output
        result_2d = result.view(-1, expected_out_dim)
        out_nope = result_2d[:, :dim_nope]
        out_scales = result_2d[:, dim_nope : dim_nope + num_tiles * 4].view(
            torch.float32
        )
        out_rope = result_2d[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

        # Nope: FP8 bytes should be identical to input
        in_nope = fp8_cache.view(-1, dim_total)[:, :dim_nope]
        assert torch.equal(
            out_nope.view(torch.uint8), in_nope.view(torch.uint8)
        ), "Nope bytes mismatch"

        # Scales: should all be 1.0
        assert torch.all(
            out_scales == 1.0
        ), f"Expected all scales=1.0, got min={out_scales.min()}, max={out_scales.max()}"

        # Rope: should equal fp8 input cast to bf16
        in_rope_fp8 = fp8_cache.view(-1, dim_total)[:, dim_nope:]
        expected_rope = in_rope_fp8.to(torch.bfloat16)
        assert torch.equal(out_rope, expected_rope), "Rope mismatch"

        print(f"  num_blocks={num_blocks}, block_size={block_size}: PASSED")

    print("All tests passed!")

    print("\nBenchmark...")
    for num_blocks in [1, 3, 7, 64, 127, 128, 255, 256, 512, 1024, 2048]:
        block_size = 64
        dim_total = 576
        fp8_cache = torch.randn(
            (num_blocks, block_size, 1, dim_total),
            dtype=torch.bfloat16,
            device="cuda",
        ).to(torch.float8_e4m3fn)

        def run():
            return requantize_fp8_to_block_scale(fp8_cache)

        time_ms = triton.testing.do_bench(run, warmup=10, rep=20)
        seq_kv = num_blocks * block_size
        print(f"  seq_kv: {seq_kv}, time: {time_ms * 1e3:.0f} us")
