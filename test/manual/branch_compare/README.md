# Branch vs Baseline Numerical-Divergence Test

Compares per-step logit/logprob distributions between a **record** run on
the **baseline** (typically `main`) and a **verify** run on a feature
branch. The verify run is forced (via the forced-token-ids debug
feature) to emit exactly the baseline's token sequence at every step, so
prefill and decode are both exercised on the branch with comparable
inputs and small kernel differences show up as small distribution drift
instead of catastrophic prefix divergence.

## Prerequisites

The test relies on the forced-token-ids debug feature in sglang core:
- request transport in `sampling_params.custom_params`: `__sglang_forced_token_ids`
  (a list of token ids) or `__sglang_forced_token_ids_path` (a server-side
  torch file; see `FORCED_TOKEN_IDS_KEY` in `sglang/srt/sampling/sampling_params.py`)
- `SamplingBatchInfo.prepare_forced_next_token_ids`
- `Sampler.forward` override step

These must be present on **both** branches under test.

Set `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` in the server's environment for both
phases. Without the env var, forced-token requests are rejected at
admission with no disk I/O. The server forces `--disable-overlap-schedule`
while it is set.

`--eval-name` takes prompts from [sgl-eval](https://github.com/sgl-project/sgl-eval)
(gsm8k, mmlu, mmlu_pro, gpqa, aime24/25/26), so install it in the client
environment (`pip install sgl-eval`, version pinned in the `test` extra of
`python/pyproject.toml`); humaneval and mgsm come from
`sglang.test.simple_eval_*`. `--prompts-file` needs neither.

## Workflow

Run from `sglang-source/test/manual/` (so `branch_compare` is importable
as a top-level package — `python -m test.manual.branch_compare.run`
collides with Python's stdlib `test` package and won't work):

```bash
cd sglang-source/test/manual

# === Phase 1: record baseline ===
git checkout main
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m branch_compare.run \
    --mode record \
    --artifact-dir /tmp/run1/main \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 \
    --eval-name gpqa --num-examples 16 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Phase 2: verify on branch ===
git checkout my-feature-branch
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m branch_compare.run \
    --mode verify \
    --artifact-dir /tmp/run1/branch \
    --record-dir   /tmp/run1/main \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 \
    --max-new-tokens 1024 \
    --topk-logprobs 128

# === Compare ===
python -m branch_compare.compare \
    --record-dir /tmp/run1/main \
    --branch-dir /tmp/run1/branch \
    --out-dir    /tmp/run1
```

`run.py` accepts every argument that `python -m sglang.launch_server`
accepts (via `ServerArgs.add_cli_args`), so pass any production server
config (`--quantization`, `--attention-backend`, `--mem-fraction-static`,
`--enable-deterministic-inference`, etc.) through directly.

If `--base-url` is set, the script attaches to an already-running server
instead of launching one. The user is responsible for ensuring the server
config matches between phases.

```bash
# Terminal A: launch the server once (e.g. slow multi-node config) and
# leave it running.
SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --tp-size 8 --port 30000

# Terminal B: point both phases at the running server. Server args are
# advisory in attach mode (recorded into meta.json but not used to launch).
cd sglang-source/test/manual
python -m branch_compare.run \
    --mode record \
    --artifact-dir /tmp/run1/main \
    --base-url http://127.0.0.1:30000 \
    --eval-name gpqa --num-examples 16 \
    --max-new-tokens 1024 --topk-logprobs 128

# then restart the server on the feature branch and:
python -m branch_compare.run \
    --mode verify \
    --artifact-dir /tmp/run1/branch \
    --record-dir   /tmp/run1/main \
    --base-url http://127.0.0.1:30000 \
    --max-new-tokens 1024 --topk-logprobs 128
```

## Output

In `--out-dir`:
- `summary.json` — `mean_abs_diff`: per step, the mean absolute logprob
  difference (nats) over the tokens that are in both sides' top-K, aggregated
  as mean, p50, p90, p99, min, max and `n`; `common_topk_count`: per-step size
  of that intersection; a per-prompt rollup; the steps where verify-side
  `output_ids` deviated from record-side.
- `histograms.txt` — ASCII bin charts (abs-diff, common-top-K), also printed to stdout.

A step whose two top-K sets share fewer than 2 tokens has no abs-diff and is
not counted in `n`. An `n` below the number of steps, or a mean common top-K
far below K, means the two sides disagree on which tokens are likely at all.

## Interpreting results

Reference measurements, to calibrate what a given `mean_abs_diff` means.
Every verify below forced the record's tokens on every step.

**Qwen3-8B.** Record: `Qwen/Qwen3-8B` bf16, 16 gsm8k prompts (`--eval-name
gsm8k`, chat template), greedy, `--max-new-tokens 512` (8192 steps),
`--topk-logprobs 20`, one GB300 GPU, default attention backend, no
deterministic inference. Each row is one verify of that record; the
same-config row reused the record's running server, the others launched their
own.

| Verify server | abs-diff mean | p50 | p99 | common top-20 | steps measured |
|---|---|---|---|---|---|
| Same config (noise floor) | 0.130 | 0.100 | 0.49 | 19.8 | 8192 |
| `--tp-size 2` | 0.152 | 0.123 | 0.54 | 19.8 | 8192 |
| `--attention-backend triton` | 0.154 | 0.122 | 0.56 | 19.8 | 8192 |
| `Qwen/Qwen3-8B-FP8` | 0.95 | 0.72 | 3.56 | 19.0 | 8192 |
| `Qwen/Qwen3-8B-AWQ` (int4) | 1.56 | 1.23 | 5.82 | 18.1 | 8192 |
| `nvidia/Qwen3-8B-NVFP4` | 2.74 | 2.22 | 8.69 | 17.1 | 8192 |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (same base, other fine-tune) | 11.1 | 10.5 | 23.9 | 12.1 | 8176 |
| `Qwen/Qwen3-8B-Base` (no post-training) | 13.0 | 12.4 | 26.8 | 11.6 | 8176 |
| `--load-format dummy` (random weights) | none | none | none | 0.0 | 0 |

**Llama-3-8B.** Record: `meta-llama/Meta-Llama-3-8B-Instruct`, same settings;
generations stop at end of turn, giving 2310 steps.

| Verify server | abs-diff mean | p50 | p99 | common top-20 | steps measured |
|---|---|---|---|---|---|
| Same config (noise floor) | 0.003 | 0.000 | 0.089 | 20.0 | 2310 |
| `meta-llama/Meta-Llama-3-8B` (base) | 6.35 | 5.89 | 14.6 | 12.4 | 2308 |
| `--load-format dummy` (random weights) | none | none | none | 0.01 | 0 |

Measured with sglang at commit 797af717 plus this feature. Takeaways:

- Kernel-level changes of the same math (parallelism, attention backend) stay
  near the noise floor. Quantization lands around 1 to 3, a differently
  trained model of the same architecture above 5.
- Unrelated weights share no top-K tokens, so abs-diff has no value and `n`
  is 0. Read that as a total mismatch, not as a pass.
- The noise floor depends on the model and on determinism. In the default
  mode it is 0.13 for Qwen3-8B but 0.003 for Llama-3-8B. With
  `--enable-deterministic-inference` on both phases, the Qwen3-8B same-config
  verify is bitwise identical (abs-diff exactly 0). Use it when the change
  under test should be small; otherwise judge a branch against a same-config
  verify of the baseline, not against 0.

## Limitations

- Disagg and speculative decoding are WIP. Use agg servers without speculative decoding for now.
- The forced-tokens path-loader reads any path the (trusted) caller provides.
  Don't enable `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` on a server exposed to
  untrusted clients.
- PD-disagg caps top-K logprobs at 128. The script defaults to 128 to stay
  safe; raise it for non-disagg deployments.
