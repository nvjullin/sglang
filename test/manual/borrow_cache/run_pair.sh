#!/bin/bash
# Launch a flag-off and a flag-on (SGLANG_ENABLE_DECOUPLED_HICACHE) server on two
# GPUs with the same args, run compare_servers.py against both, then stop them.
#
#   run_pair.sh <src-root> <log-dir> <compare-args...> -- <server-args...>
#
# Env: GPU_A / GPU_B (default 0 / 1), PORT_A / PORT_B (default 30000 / 30001),
#      TP (default 1; GPU_A/GPU_B then name comma lists), READY_SECS (default 1200).
set -uo pipefail
SRC=$1; LOG=$2; shift 2
COMPARE_ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do COMPARE_ARGS+=("$1"); shift; done
shift
SERVER_ARGS=("$@")
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
PORT_A=${PORT_A:-30000}; PORT_B=${PORT_B:-30001}
FLAG_A=${FLAG_A:-0}; FLAG_B=${FLAG_B:-1}
mkdir -p "$LOG"
export PYTHONPATH=$SRC/python:${PYTHONPATH:-}

launch() {  # gpu port flag logname
  # Own session, so its worker processes can be killed as one group.
  CUDA_VISIBLE_DEVICES=$1 SGLANG_ENABLE_DECOUPLED_HICACHE=$3 \
    setsid python3 -m sglang.launch_server --port "$2" "${SERVER_ARGS[@]}" \
    > "$LOG/$4.log" 2>&1 &
  echo $!
}

wait_ready() {  # port pid
  for _ in $(seq 1 $(( ${READY_SECS:-1200} / 2 ))); do
    if curl -sf "http://127.0.0.1:$1/health" > /dev/null; then return 0; fi
    if ! kill -0 "$2" 2>/dev/null; then return 1; fi
    sleep 2
  done
  return 1
}

PA=$(launch "$GPU_A" "$PORT_A" "$FLAG_A" baseline)
PB=$(launch "$GPU_B" "$PORT_B" "$FLAG_B" decoupled)
rc=0
if ! wait_ready "$PORT_A" "$PA"; then echo "baseline failed to start"; tail -50 "$LOG/baseline.log"; rc=2; fi
if ! wait_ready "$PORT_B" "$PB"; then echo "decoupled failed to start"; tail -80 "$LOG/decoupled.log"; rc=2; fi
if [ $rc -eq 0 ]; then
  python3 "$SRC/test/manual/borrow_cache/compare_servers.py" \
    --urls "http://127.0.0.1:$PORT_A" "http://127.0.0.1:$PORT_B" \
    --out "$LOG/compare.json" "${COMPARE_ARGS[@]}"
  rc=$?
fi
kill -- -"$PA" -"$PB" 2>/dev/null
sleep 5
kill -9 -- -"$PA" -"$PB" 2>/dev/null
grep -E "Traceback|Error|error" "$LOG/decoupled.log" | head -20
echo "run_pair rc=$rc"
exit $rc
