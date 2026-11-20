"""Two 1P1D deployments on one 4-GPU node: the prefill of the second runs with
SGLANG_ENABLE_DECOUPLED_HICACHE. Runs compare_servers.py against both routers.

    python run_pd_pair.py --model Qwen/Qwen3-0.6B --log-dir /tmp/pd \\
        --server-args="--enable-hierarchical-cache --max-total-tokens 8192" \\
        -- --num-prefixes 8 --prefix-len 1500
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from sglang.test.server_fixtures.disaggregation_fixture import (
    _maybe_set_roce_gid_index,
    _maybe_set_ucx_net_devices,
    get_rdma_devices_args,
    popen_launch_sgl_router,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_pd_server
from sglang.utils import wait_for_http_ready

HERE = Path(__file__).parent


def launch_side(
    *, model, mode, gpu, port, bootstrap, nccl, flag, args, log_dir, name, backend
):
    transport = ["--disaggregation-transfer-backend", backend]
    if backend == "mooncake":
        transport += ["--disaggregation-ib-device", get_rdma_devices_args([gpu])]
    other = [
        "--disaggregation-mode",
        mode,
        "--disaggregation-bootstrap-port",
        str(bootstrap),
        "--nccl-port",
        str(nccl),
        "--base-gpu-id",
        str(gpu),
        *transport,
        *args,
    ]
    env = {"SGLANG_ENABLE_DECOUPLED_HICACHE": "1" if flag else "0"}
    out = open(log_dir / f"{name}.log", "w")
    proc = popen_launch_pd_server(
        model,
        f"http://127.0.0.1:{port}",
        timeout=1200,
        other_args=other,
        env=env,
        return_stdout_stderr=(out, out),
    )
    return proc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--server-args", default="")
    p.add_argument("--prefill-args", default="")
    # nixl over UCX cuda_ipc needs no RDMA NIC (works on one NVLink node).
    p.add_argument("--transfer-backend", default="nixl")
    p.add_argument("compare_args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if a.transfer_backend == "mooncake":
        os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "true"
        os.environ["MC_MIN_RPC_PORT"] = "61000"
        os.environ["MC_MAX_RPC_PORT"] = "65000"
        _maybe_set_roce_gid_index(get_rdma_devices_args())
        _maybe_set_ucx_net_devices(get_rdma_devices_args())
    else:
        # NIXL >= 1.5 refuses a UCX_TLS without a CUDA transport (some sites
        # inject UCX_TLS=tcp into containers).
        os.environ["UCX_TLS"] = "cuda_copy,cuda_ipc,sm,self,tcp"
    common = shlex.split(a.server_args)
    prefill_only = shlex.split(a.prefill_args)
    procs, routers = [], []
    try:
        for i, flag in enumerate((False, True)):
            name = "decoupled" if flag else "baseline"
            base = 31000 + 1000 * i
            pre = launch_side(
                model=a.model, mode="prefill", gpu=2 * i, port=base + 100,
                bootstrap=base + 500, nccl=base + 300, flag=flag,
                args=common + prefill_only, log_dir=log_dir, name=f"{name}_prefill",
                backend=a.transfer_backend,
            )
            dec = launch_side(
                model=a.model, mode="decode", gpu=2 * i + 1, port=base + 200,
                bootstrap=base + 500, nccl=base + 400, flag=False,
                args=common, log_dir=log_dir, name=f"{name}_decode",
                backend=a.transfer_backend,
            )
            procs += [pre, dec]
        for i in range(2):
            base = 31000 + 1000 * i
            for port, proc in zip((base + 100, base + 200), procs[2 * i : 2 * i + 2]):
                wait_for_http_ready(
                    url=f"http://127.0.0.1:{port}/health", timeout=1200, process=proc
                )
            routers.append(
                popen_launch_sgl_router(
                    a.model,
                    [f"http://127.0.0.1:{base + 100}", f"http://127.0.0.1:{base + 200}"],
                    host="127.0.0.1",
                    port=str(base),
                )
            )
        cmd = [
            sys.executable,
            str(HERE / "compare_servers.py"),
            "--urls",
            "http://127.0.0.1:31000",
            "http://127.0.0.1:32000",
            "--out",
            str(log_dir / "compare.json"),
            *[x for x in a.compare_args if x != "--"],
        ]
        return subprocess.call(cmd)
    finally:
        for proc in routers + procs:
            kill_process_tree(proc.pid)


if __name__ == "__main__":
    sys.exit(main())
