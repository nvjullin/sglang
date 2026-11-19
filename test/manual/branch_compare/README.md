# Branch vs Baseline Numerical-Divergence Test

Compares per-step logit/logprob distributions between a **record** run on
the **baseline** (typically `main`) and a **verify** run on a feature
branch. The verify run is forced (via the forced-token-ids debug
feature) to emit exactly the baseline's token sequence at every step, so
prefill and decode are both exercised on the branch with comparable
inputs and small kernel differences show up as small distribution drift
instead of catastrophic prefix divergence.

## Prerequisites

The test relies on the forced-token-ids debug feature in sglang core, which must
be present on **both** sides under test:
- request transport in `sampling_params.custom_params`: `__sglang_forced_token_ids`
  (a list of token ids) or `__sglang_forced_token_ids_path` (a server-side
  torch file; see `FORCED_TOKEN_IDS_KEY` in `sglang/srt/sampling/sampling_params.py`);
- `sglang/srt/sampling/forced_token_ids.py`: the token emitted from a logits row
  at KV position `p` takes forced index `p + 1 - prompt_len`, looked up on the
  device, so forcing holds under the overlap scheduler, speculative verify and
  PD disaggregation;
- the hooks in `Sampler.forward` and in EAGLE's `eagle_sample`.

Set `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` in the environment of every server
process (prefill and decode under PD) for both phases. Without it, forced-token
requests are rejected at admission with no disk I/O. With it, the server refuses
to start on a config that would emit tokens without forcing them: speculative
algorithms other than EAGLE/EAGLE3 (NEXTN counts as EAGLE),
`--speculative-use-rejection-sampling`, `SGLANG_SIMULATE_ACC_LEN`, dLLM and
Unlimited-OCR v2. Nothing else changes: the overlap scheduler, CUDA graphs, DP
attention and the deployment's speculative settings stay as configured.

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
    --eval-name gpqa --num-examples 64 --concurrency 64 \
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
    --tp-size 8

# === Compare ===
python -m branch_compare.compare \
    --record-dir /tmp/run1/main \
    --branch-dir /tmp/run1/branch \
    --out-dir    /tmp/run1
```

Prompts are tokenized on the client (`--api chat` applies the chat template
with `--chat-template-kwargs`, e.g. `'{"enable_thinking": true}'` to match a
benchmark's thinking setting), and the server receives `input_ids`. The record
stores those ids and verify replays them, so both phases see the same prompt
tokens whatever tokenizer the frontend runs. Verify also inherits the record's
sampling settings, `--topk-logprobs` and `--concurrency`; a flag given on the
verify command line overrides them. Verify asks for exactly as many tokens as
the record produced and forces each one; a request whose output differs from the
record is listed under `failed_steps`.

`--concurrency` is the number of requests kept in flight. Set it to the
concurrency of the benchmark the deployment serves, so the verify runs the batch
shapes (and CUDA graph sizes) production runs; the run needs at least that many
prompts.

`--forced-transport inline` (default) sends the record's ids in each request.
`--forced-transport path` sends the record file's absolute path instead, which
every server process that admits the request must be able to read; under PD
that is both prefill and decode.

`run.py` accepts every argument that `python -m sglang.launch_server`
accepts (via `ServerArgs.add_cli_args`), so pass any production server
config (`--quantization`, `--attention-backend`, `--mem-fraction-static`,
`--enable-deterministic-inference`, etc.) through directly.

### Attaching to a deployment

If `--base-url` is set, the script attaches to an already-running server
instead of launching one; server args are then only recorded in `meta.json`.
The user is responsible for the deployment being the same in both phases,
except for the code under test.

```bash
cd sglang-source/test/manual
python -m branch_compare.run --mode record --artifact-dir /tmp/run1/main \
    --base-url http://127.0.0.1:30000 --model-path MODEL \
    --eval-name gsm8k --num-examples 512 --concurrency 128 --max-new-tokens 2048
# restart the deployment on the feature branch, then:
python -m branch_compare.run --mode verify --artifact-dir /tmp/run1/branch \
    --record-dir /tmp/run1/main --base-url http://127.0.0.1:30000 --model-path MODEL
```

`--transport sglang` (default) speaks sglang's native `/generate`: a single
server, or the sglang router in front of a PD deployment.

`--transport dynamo` speaks Dynamo's native SGLang `/generate` (SSE), which
forwards the request body to the prefill and decode workers unchanged except for
`n` and `max_new_tokens` on prefill. It needs, in addition to
`SGLANG_ENABLE_FORCED_TOKEN_IDS=1` on every worker:
- `DYN_SGLANG_ENABLE_GENERATE=1` on the Dynamo frontend (serves the route);
- `DYN_SGL_ALLOW_TOP_LOGPROBS=1` on every sglang worker (lets requests ask for
  top-K logprobs).

## Output

In `--out-dir`:
- `summary.json` — `mean_abs_diff`: per step, the mean absolute logprob
  difference (nats) over the tokens that are in both sides' top-K, aggregated
  as mean, p50, p90, p99, min, max and `n`; `common_topk_count`: per-step size
  of that intersection; `chosen_abs_diff`: per step, the absolute difference of
  the emitted (forced) token's logprob; a per-prompt rollup; the steps where
  verify-side `output_ids` deviated from record-side.
- `verdict.json` — pass/fail under policy `branch_compare.v1`. It always fails on
  a forced-token mismatch, on a record prompt missing from the verify, and when
  no step has two common top-K tokens. `--max-mean-abs-diff`,
  `--max-p99-abs-diff`, `--max-chosen-abs-diff-mean` and `--min-common-topk-frac`
  (mean common top-K / K) add threshold checks; each check is listed with its
  value and limit.
- `histograms.txt` — ASCII bin charts (abs-diff, common-top-K), also printed to stdout.

A step whose two top-K sets share fewer than 2 tokens has no abs-diff and is
not counted in `n`. An `n` below the number of steps, or a mean common top-K
far below K, means the two sides disagree on which tokens are likely at all.

## Interpreting results

Reference measurements, to calibrate what a given `mean_abs_diff` means.
Every verify below forced the record's tokens on every step. They were taken on
one GPU, aggregated serving, no speculative decoding, overlap scheduler off.

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

### Noise floor at a deployed config

The tables above do not transfer to a different deployment: DP attention, EP,
PD and speculative decoding each change batch composition and kernel choice
from run to run, and deterministic inference is not what production runs. To
calibrate a deployment, record the baseline and then verify the **same
baseline** at the same deployment and concurrency; that compare is the
deployment's floor. Judge a branch's verify of the same record against it, and
derive the `compare.py` thresholds from it (the pass rule is the caller's
policy, not part of this harness).

At a large K the top-K mean is dominated by the low-probability tail. On a
GLM-5.2 NVFP4 PD deployment (DP attention, EP, EAGLE, K=128), the same-config
top-K mean was 0.79 nats, and two verifies of one record differed by as much as
the record and a verify did. 93% of the compared entries were tokens below -10
nats (mean difference 0.79); tokens above -1 nats differed by 0.05, and the
emitted token's logprob by 0.07 on average (median 0.0005). `chosen_abs_diff`
and a smaller K track the tokens that matter.

## Limitations

- The forced-tokens path loader reads any path the (trusted) caller provides.
  Don't enable `SGLANG_ENABLE_FORCED_TOKEN_IDS=1` on a server exposed to
  untrusted clients.
- PD disaggregation carries at most 128 top-K logprobs for the first output
  token; a request asking for more is rejected at admission. The script
  defaults to 128.
- Speculative decoding: EAGLE and EAGLE3 (single- and multi-layer) only; other
  algorithms are refused at startup. Batches with forced requests skip the
  optional EAGLE post-verify CUDA graphs (`SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH`)
  and run that step eagerly; the target forward and the returned logprobs are
  unaffected.
