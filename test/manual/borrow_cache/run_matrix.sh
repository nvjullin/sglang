#!/bin/bash
# Run two run_pair.sh comparisons side by side on a 4-GPU node.
#   run_matrix.sh <src-root> <log-root> <case-a> <case-b>
# A case name selects compare/server args from the table below; "<case>@aa"
# runs both servers with the flag off.
set -uo pipefail
SRC=$1; ROOT=$2; shift 2
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
DET="--enable-deterministic-inference --attention-backend triton"
# --enable-metrics: deployments (srt-slurm) always set it; its idle stats path reads the cache.
HI="--enable-hierarchical-cache --hicache-ratio 4 --enable-metrics"

case_args() {
  case $1 in
    page1)   echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 -- --model-path $MODEL $HI $DET --max-total-tokens 8192" ;;
    page16_chunk)
             echo "--num-prefixes 6 --prefix-len 1500 --rounds 3 --max-new-tokens 128 -- --model-path $MODEL $HI $DET --max-total-tokens 6144 --page-size 16 --chunked-prefill-size 1024" ;;
    page64_retract)
             echo "--num-prefixes 4 --prefix-len 800 --rounds 2 --max-new-tokens 600 --concurrency 32 --prompts-per-round 48 -- --model-path $MODEL $HI $DET --max-total-tokens 12288 --page-size 64" ;;
    no_overlap)
             echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 -- --model-path $MODEL $HI $DET --max-total-tokens 8192 --page-size 16 --disable-overlap-schedule" ;;
    device_only)
             echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 -- --model-path $MODEL $DET --max-total-tokens 8192 --page-size 16" ;;
    eagle3)  echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 64 -- --model-path Qwen/Qwen3-8B --speculative-algorithm EAGLE3 --speculative-draft-model-path ${EAGLE_DRAFT:?set EAGLE_DRAFT} --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 $HI $DET --max-total-tokens 12288 --page-size 16" ;;
    mla)     echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 64 -- --model-path ${MLA_MODEL:?set MLA_MODEL} --trust-remote-code $HI $DET --max-total-tokens 8192 --page-size 16" ;;
    dsa)     echo "--num-prefixes 6 --prefix-len 3000 --rounds 3 --max-new-tokens 32 --prompts-per-round 24 -- --model-path ${DSA_MODEL:?set DSA_MODEL} --trust-remote-code --load-format dummy --json-model-override-args {\"num_hidden_layers\":4,\"first_k_dense_replace\":1,\"n_routed_experts\":32,\"num_nextn_predict_layers\":0} $HI --max-total-tokens 16384 --page-size 64" ;;
    gdn)     echo "--num-prefixes 6 --prefix-len 3000 --rounds 3 --max-new-tokens 64 -- --model-path ${GDN_MODEL:?set GDN_MODEL} $HI $DET --max-total-tokens 16384 --page-size 64 --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 48" ;;
    kda)     echo "--num-prefixes 6 --prefix-len 3000 --rounds 3 --max-new-tokens 64 -- --model-path ${KDA_MODEL:?set KDA_MODEL} --trust-remote-code $HI $DET --max-total-tokens 16384 --page-size 64 --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 48" ;;
    gdn_seq) echo "--num-prefixes 6 --prefix-len 3000 --rounds 2 --prompts-per-round 24 --max-new-tokens 64 --concurrency 1 -- --model-path ${GDN_MODEL:?set GDN_MODEL} $HI $DET --max-total-tokens 131072 --page-size 64 --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 48" ;;
    kda_seq) echo "--num-prefixes 6 --prefix-len 3000 --rounds 2 --prompts-per-round 24 --max-new-tokens 64 --concurrency 1 -- --model-path ${KDA_MODEL:?set KDA_MODEL} --trust-remote-code $HI $DET --max-total-tokens 131072 --page-size 64 --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 48" ;;
    swa)     echo "--num-prefixes 6 --prefix-len 1500 --rounds 3 --max-new-tokens 64 -- --model-path ${SWA_MODEL:?set SWA_MODEL} $HI $DET --max-total-tokens 8192 --page-size 16 --chunked-prefill-size 1024" ;;
    swa_seq) echo "--num-prefixes 6 --prefix-len 1500 --rounds 2 --prompts-per-round 24 --max-new-tokens 64 --concurrency 1 -- --model-path ${SWA_MODEL:?set SWA_MODEL} $HI $DET --max-total-tokens 8192 --page-size 16 --chunked-prefill-size 1024" ;;
    swa_small) echo "--num-prefixes 6 --prefix-len 1500 --rounds 3 --max-new-tokens 300 -- --model-path ${SWA_MODEL:?set SWA_MODEL} $HI $DET --max-total-tokens 16384 --swa-full-tokens-ratio 0.25 --page-size 16 --chunked-prefill-size 1024" ;;
    swa_small_seq) echo "--num-prefixes 6 --prefix-len 1500 --rounds 2 --prompts-per-round 24 --max-new-tokens 300 --concurrency 1 -- --model-path ${SWA_MODEL:?set SWA_MODEL} $HI $DET --max-total-tokens 16384 --swa-full-tokens-ratio 0.25 --page-size 16 --chunked-prefill-size 1024" ;;
    tp2dp2)  echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 64 -- --model-path $MODEL $HI $DET --max-total-tokens 8192 --page-size 16 --tp 2 --attn-dp-size 2" ;;
    tp2)     echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 128 -- --model-path $MODEL $HI $DET --max-total-tokens 8192 --page-size 16 --chunked-prefill-size 2048 --tp 2" ;;
    *) echo "unknown case $1" >&2; exit 2 ;;
  esac
}

run_case() {  # name gpu_a gpu_b port_a port_b
  # A "<case>@aa" name runs the flag-off server twice: the noise floor of <case>.
  local name=$1 flag_b=1
  local args
  if [[ $name == *@aa ]]; then flag_b=0; fi
  args=$(case_args "${name%@aa}")
  FLAG_B=$flag_b GPU_A=$2 GPU_B=$3 PORT_A=$4 PORT_B=$5 \
    bash "$SRC/test/manual/borrow_cache/run_pair.sh" "$SRC" "$ROOT/$name" $args \
    > "$ROOT/$name.out" 2>&1
  echo "$name rc=$?"
}

mkdir -p "$ROOT"
if [ $# -eq 1 ] && [[ $1 == tp2* ]]; then
  run_case "$1" 0,1 2,3 30000 30500 &
  P1=$!
else
  run_case "$1" 0 1 30000 30500 &
  P1=$!
  if [ $# -ge 2 ]; then run_case "$2" 2 3 31000 31500 & P2=$!; fi
fi
wait $P1; [ -n "${P2:-}" ] && wait $P2
for n in "$@"; do
  echo "=== $n"
  grep -v "Capturing\|it/s\]" "$ROOT/$n.out" | grep -A40 '"num_prompts"' | grep -E "cached_tokens|mismatched|identical|step|margin"
  grep "Cache service stats" "$ROOT/$n/decoupled.log" | tail -1 | cut -c1-600
  grep -E "Traceback|Error" "$ROOT/$n/decoupled.log" | head -5
  tail -1 "$ROOT/$n.out"
done
