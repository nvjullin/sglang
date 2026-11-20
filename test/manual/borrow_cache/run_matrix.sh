#!/bin/bash
# Run two run_pair.sh comparisons side by side on a 4-GPU node.
#   run_matrix.sh <src-root> <log-root> <case-a> <case-b>
# A case name selects compare/server args from the table below.
set -uo pipefail
SRC=$1; ROOT=$2; shift 2
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
DET="--enable-deterministic-inference --attention-backend triton"
HI="--enable-hierarchical-cache --hicache-ratio 4"

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
    tp2dp2)  echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 64 -- --model-path $MODEL $HI $DET --max-total-tokens 8192 --page-size 16 --tp 2 --dp 2 --enable-dp-attention" ;;
    tp2)     echo "--num-prefixes 8 --prefix-len 1500 --rounds 3 --max-new-tokens 128 -- --model-path $MODEL $HI $DET --max-total-tokens 8192 --page-size 16 --chunked-prefill-size 2048 --tp 2" ;;
    *) echo "unknown case $1" >&2; exit 2 ;;
  esac
}

run_case() {  # name gpu_a gpu_b port_a port_b
  local name=$1
  local args
  args=$(case_args "$name")
  GPU_A=$2 GPU_B=$3 PORT_A=$4 PORT_B=$5 \
    bash "$SRC/test/manual/borrow_cache/run_pair.sh" "$SRC" "$ROOT/$name" $args \
    > "$ROOT/$name.out" 2>&1
  echo "$name rc=$?"
}

mkdir -p "$ROOT"
if [ $# -eq 1 ] && [[ $1 == tp2* ]]; then
  run_case "$1" 0,1 2,3 30000 30001 &
  P1=$!
else
  run_case "$1" 0 1 30000 30001 &
  P1=$!
  if [ $# -ge 2 ]; then run_case "$2" 2 3 30002 30003 & P2=$!; fi
fi
wait $P1; [ -n "${P2:-}" ] && wait $P2
for n in "$@"; do
  echo "=== $n"
  grep -v "Capturing\|it/s\]" "$ROOT/$n.out" | grep -A40 '"num_prompts"' | grep -E "cached_tokens|mismatched|identical|step|margin"
  grep "Cache service stats" "$ROOT/$n/decoupled.log" | tail -1 | cut -c1-600
  grep -E "Traceback|Error" "$ROOT/$n/decoupled.log" | head -5
  tail -1 "$ROOT/$n.out"
done
