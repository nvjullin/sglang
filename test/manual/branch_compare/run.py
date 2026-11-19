"""branch_compare CLI: record / verify phases.

Usage (run from sglang-source/test/manual/):

    SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \\
    python -m branch_compare.run \\
        --mode record --artifact-dir DIR \\
        --model-path MODEL [--tp-size N] [...any server args...] \\
        --eval-name gpqa --num-examples 16

    SGLANG_ENABLE_FORCED_TOKEN_IDS=1 \\
    python -m branch_compare.run \\
        --mode verify --artifact-dir DIR --record-dir RECORD_DIR \\
        --model-path MODEL [--tp-size N] [...any server args...]

The script registers ServerArgs.add_cli_args so every server flag is
accepted as pass-through and forwarded to popen_launch_server. If
--base-url is set the script attaches to a running server instead of
launching one (server args become advisory and are recorded in meta.json).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from branch_compare import artifacts
from branch_compare import prompts as prompts_mod
from branch_compare import transport as transport_mod

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

DEFAULT_PORT = 30000
TIMEOUT_LAUNCH = 600
LARGE_ARTIFACT_GB = 5.0
DEFAULT_EVAL_EXAMPLES = 16
# Verify takes these from the record unless given on the command line.
INHERITED_FROM_RECORD = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "topk_logprobs",
    "concurrency",
)
DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "min_p": None,
    "seed": None,
    "topk_logprobs": 128,
    "concurrency": 16,
}


# --------------------------- argparse ---------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="branch_compare: record / verify a server's per-step "
        "logprobs for divergence comparison."
    )

    # Mode + I/O
    g = parser.add_argument_group("branch_compare")
    g.add_argument("--mode", choices=["record", "verify"], required=True)
    g.add_argument("--artifact-dir", required=True)
    g.add_argument(
        "--record-dir",
        default=None,
        help="(verify mode) Directory containing main's record artifact.",
    )
    g.add_argument(
        "--base-url",
        default=None,
        help="Attach to a running server instead of launching one.",
    )
    g.add_argument(
        "--transport",
        choices=transport_mod.TRANSPORTS,
        default="sglang",
        help="Request protocol of --base-url: sglang's native /generate (a server "
        "or the sglang router), or Dynamo's native SGLang /generate (see "
        "transport.DynamoTransport for the env vars it needs).",
    )
    g.add_argument(
        "--forced-transport",
        choices=["inline", "path"],
        default="inline",
        help="(verify mode) inline sends the record's ids in the request; path "
        "sends the record file's path, which every prefill and decode worker "
        "must be able to read.",
    )

    # Prompts
    g.add_argument(
        "--eval-name",
        default=None,
        choices=prompts_mod.EVAL_NAMES,
        help="Take prompts from this eval (sgl-eval for most; see prompts.py).",
    )
    g.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Prompts to take: default 16 from --eval-name, all of --prompts-file.",
    )
    g.add_argument(
        "--prompts-file",
        default=None,
        help="Alternative to --eval-name: JSONL with one of input_ids, messages "
        "or prompt per line (see prompts.py).",
    )
    g.add_argument(
        "--api",
        choices=["chat", "completion"],
        default="chat",
        help="chat applies the chat template client-side; completion tokenizes "
        "raw text.",
    )
    g.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help="JSON passed to apply_chat_template, e.g. '{\"enable_thinking\": true}'.",
    )

    # Generation. Defaults are applied in main(): verify inherits the record's.
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--min-p", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument(
        "--topk-logprobs",
        type=int,
        default=None,
        help="K for top_logprobs_num (default 128). PD disaggregation rejects K > 128.",
    )

    g.add_argument(
        "--ack-large-artifacts",
        action="store_true",
        help="Required when projected artifact size exceeds 5 GB.",
    )
    g.add_argument(
        "--concurrency",
        "--num-threads",
        dest="concurrency",
        type=int,
        default=None,
        help="In-flight requests (default 16). Set it to the deployment's "
        "benchmark concurrency so batch shapes match; there must be at least as "
        "many prompts.",
    )

    # Server pass-through. Last so server's own --help text reads naturally.
    ServerArgs.add_cli_args(parser)
    return parser


# --------------------------- helpers ---------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _git_sha(cwd: str = ".") -> Tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True
        ).strip()
        return sha, bool(diff)
    except Exception:
        return "unknown", False


def _server_args_passthrough(args: argparse.Namespace) -> List[str]:
    """Reconstruct the CLI list to forward to popen_launch_server.

    Walk a fresh parser populated only by ServerArgs.add_cli_args and emit
    flags whose value on `args` differs from the parser default. We can't go
    through ServerArgs(...) construction because __post_init__ derives fields
    (model_config, random_seed, kt_*, ...) that don't map back to CLI flags.
    """
    sa_parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(sa_parser)

    out: List[str] = []
    for action in sa_parser._actions:
        dest = action.dest
        if dest in ("help", "model_path"):
            continue  # model_path is passed positionally to popen_launch_server
        if not action.option_strings:
            continue
        if not hasattr(args, dest):
            continue
        cur = getattr(args, dest)
        default = action.default
        if cur == default:
            continue
        cli = action.option_strings[0]
        if isinstance(cur, bool):
            if cur and not default:
                out.append(cli)
            # bool flags rarely have a paired --no-X; leave it alone otherwise
        elif isinstance(cur, list):
            for item in cur:
                out += [cli, str(item)]
        else:
            out += [cli, str(cur)]
    return out


def _project_artifact_size_gb(n_prompts: int, max_new_tokens: int, topk: int) -> float:
    # bf16 logprobs (2B) + int32 token_ids (4B) per (step, k), plus int64 chosen_logprob (negligible)
    bytes_per = (2 + 4) * topk
    total = n_prompts * max_new_tokens * bytes_per
    return total / (1024**3)


def _check_artifact_size(args: argparse.Namespace, n: int, max_new: int) -> None:
    proj_gb = _project_artifact_size_gb(n, max_new, args.topk_logprobs)
    print(
        f"[branch_compare] {n} prompts, projected artifact size <= {proj_gb:.2f} GB",
        flush=True,
    )
    if proj_gb > LARGE_ARTIFACT_GB and not args.ack_large_artifacts:
        raise SystemExit(
            f"Projected artifact size ({proj_gb:.2f} GB) exceeds "
            f"{LARGE_ARTIFACT_GB} GB. Re-run with --ack-large-artifacts to confirm."
        )


def _check_concurrency(args: argparse.Namespace, n: int) -> None:
    if n < args.concurrency:
        raise SystemExit(
            f"{n} prompts cannot keep --concurrency {args.concurrency} requests in "
            "flight; raise --num-examples or lower --concurrency."
        )


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _sampling_meta(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "seed": args.seed,
    }


def _build_sampling_params(args: argparse.Namespace, max_new: int) -> Dict[str, Any]:
    sp: Dict[str, Any] = {
        "max_new_tokens": max_new,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k is not None:
        sp["top_k"] = args.top_k
    if args.min_p is not None:
        sp["min_p"] = args.min_p
    if args.seed is not None:
        sp["sampling_seed"] = args.seed
    return sp


def _common_meta(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    sha, dirty = _git_sha()
    return {
        "phase": phase,
        "git_commit": sha,
        "git_dirty": dirty,
        "model_path": args.model_path,
        "server_args": _server_args_passthrough(args),
        "server_mode": "attached" if args.base_url else "launched",
        "transport": args.transport,
        "topk_logprobs": args.topk_logprobs,
        "max_new_tokens": args.max_new_tokens,
        "concurrency": args.concurrency,
        "sampling": _sampling_meta(args),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


# --------------------------- record / verify ---------------------------


def cmd_record(args: argparse.Namespace, client) -> None:
    # --tokenizer-path is the server's own flag; it also picks the prompt tokenizer.
    tokenizer = _load_tokenizer(args.tokenizer_path or args.model_path)
    prompts_list = prompts_mod.load_prompts(
        eval_name=args.eval_name,
        num_examples=(
            args.num_examples
            if args.num_examples is not None or args.prompts_file is not None
            else DEFAULT_EVAL_EXAMPLES
        ),
        prompts_file=args.prompts_file,
        api=args.api,
        tokenizer=tokenizer,
        chat_template_kwargs=args.chat_template_kwargs,
    )
    n = len(prompts_list)
    _check_artifact_size(args, n, args.max_new_tokens)
    _check_concurrency(args, n)
    os.makedirs(args.artifact_dir, exist_ok=True)
    meta = _common_meta(args, "record")

    def _record_one(i_p: Tuple[int, prompts_mod.Prompt]) -> Dict[str, Any]:
        i, p = i_p
        t0 = time.time()
        res = client.generate(
            p.input_ids,
            _build_sampling_params(args, args.max_new_tokens),
            args.topk_logprobs,
        )
        artifacts.write_prompt_artifact(
            args.artifact_dir,
            i,
            input_ids=p.input_ids,
            output_ids=res.output_ids,
            top_k_token_ids=res.top_k_token_ids,
            top_k_logprobs=res.top_k_logprobs,
            chosen_logprob=res.chosen_logprob,
        )
        print(
            f"  [{i + 1}/{n}] {len(p.input_ids)} prompt tokens, "
            f"{len(res.output_ids)} steps, finish={res.finish_reason}, "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
        return {
            "idx": i,
            "prompt": p.text,
            "messages": p.messages,
            "logprobs_file": artifacts.prompt_filename(i),
            "n_prompt_tokens": len(p.input_ids),
            "n_steps": len(res.output_ids),
            "stop_reason": res.finish_reason,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        prompt_records = list(ex.map(_record_one, enumerate(prompts_list)))

    meta["eval"] = {
        "name": args.eval_name,
        "num_examples": args.num_examples,
        "api": args.api,
        "chat_template_kwargs": args.chat_template_kwargs,
        "prompts_file": args.prompts_file,
    }
    meta["prompts"] = prompt_records
    artifacts.write_meta(args.artifact_dir, meta)
    print(f"[branch_compare] record artifact written to {args.artifact_dir}")


def _first_mismatch(out_ids: List[int], record_ids: List[int]) -> Optional[int]:
    if out_ids == record_ids:
        return None
    return next(
        (k for k, (a, b) in enumerate(zip(out_ids, record_ids)) if a != b),
        min(len(out_ids), len(record_ids)),
    )


def cmd_verify(args: argparse.Namespace, client, record_meta: Dict[str, Any]) -> None:
    record_dir_abs = os.path.abspath(args.record_dir)
    n = len(record_meta["prompts"])
    max_steps = max((p["n_steps"] for p in record_meta["prompts"]), default=0)
    _check_artifact_size(args, n, max_steps)
    _check_concurrency(args, n)
    os.makedirs(args.artifact_dir, exist_ok=True)
    meta = _common_meta(args, "verify")

    def _verify_one(
        entry: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            return _verify_request(entry)
        except Exception as e:
            # One failed request must not discard the others; the verdict fails on it.
            print(f"  [{entry['idx'] + 1}/{n}] REQUEST FAILED: {e!r}", flush=True)
            return None, {"prompt_idx": entry["idx"], "error": repr(e)[:2000]}

    def _verify_request(
        entry: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        idx = entry["idx"]
        record = artifacts.read_prompt_artifact(record_dir_abs, idx)
        record_ids = record["output_ids"].tolist()
        if args.forced_transport == "path":
            forced = transport_mod.Forced(
                path=artifacts.absolute_logprobs_path(record_dir_abs, idx)
            )
        else:
            forced = transport_mod.Forced(ids=record_ids)
        t0 = time.time()
        res = client.generate(
            record["input_ids"].tolist(),
            _build_sampling_params(args, max_new=len(record_ids)),
            args.topk_logprobs,
            forced=forced,
        )
        artifacts.write_prompt_artifact(
            args.artifact_dir,
            idx,
            input_ids=record["input_ids"].tolist(),
            output_ids=res.output_ids,
            top_k_token_ids=res.top_k_token_ids,
            top_k_logprobs=res.top_k_logprobs,
            chosen_logprob=res.chosen_logprob,
        )

        mismatch_step = _first_mismatch(res.output_ids, record_ids)
        mismatch: Optional[Dict[str, Any]] = None
        if mismatch_step is not None:
            mismatch = {
                "prompt_idx": idx,
                "first_mismatch_step": mismatch_step,
                "n_steps": len(res.output_ids),
                "record_n_steps": len(record_ids),
            }
            print(
                f"  [{idx + 1}/{n}] FORCED-TOKEN MISMATCH at step "
                f"{mismatch_step}: forcing did not take.",
                flush=True,
            )
        else:
            print(
                f"  [{idx + 1}/{n}] {len(res.output_ids)} steps OK, "
                f"finish={res.finish_reason}, {time.time() - t0:.1f}s",
                flush=True,
            )
        return {
            "idx": idx,
            "prompt": entry.get("prompt"),
            "messages": entry.get("messages"),
            "logprobs_file": artifacts.prompt_filename(idx),
            "n_prompt_tokens": len(record["input_ids"]),
            "n_steps": len(res.output_ids),
            "stop_reason": res.finish_reason,
        }, mismatch

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(_verify_one, record_meta["prompts"]))
    failed_steps = [m for r, m in results if r is not None and m is not None]
    failed_requests = [m for r, m in results if r is None]

    meta["forced_transport"] = args.forced_transport
    meta["record_dir"] = record_dir_abs
    meta["record_git_commit"] = record_meta.get("git_commit")
    meta["prompts"] = [r for r, _ in results if r is not None]
    meta["failed_steps"] = failed_steps
    meta["failed_requests"] = failed_requests
    artifacts.write_meta(args.artifact_dir, meta)
    print(
        f"[branch_compare] verify artifact written to {args.artifact_dir}; "
        f"{len(failed_steps)} prompts had forced-token mismatches, "
        f"{len(failed_requests)} requests failed."
    )


# --------------------------- main ---------------------------


def _resolve_inherited(
    args: argparse.Namespace, record_meta: Optional[Dict[str, Any]]
) -> None:
    """Fill unset generation settings: from the record in verify mode, else the
    defaults. Explicit command-line values always win."""
    for name in INHERITED_FROM_RECORD:
        if getattr(args, name) is not None:
            continue
        value = DEFAULTS[name]
        if record_meta is not None:
            source = (
                record_meta.get("sampling", {})
                if name in ("temperature", "top_p", "top_k", "min_p", "seed")
                else record_meta
            )
            value = source.get(name, value)
        setattr(args, name, value)


def _launch_server(args: argparse.Namespace):
    port = args.port or _free_port() or DEFAULT_PORT
    base_url = f"http://127.0.0.1:{port}"
    # Pass through every server arg the user set (excluding our own group).
    server_other_args = _server_args_passthrough(args)
    # popen_launch_server passes the model as a positional + appends --port,
    # so strip those if they snuck in.
    server_other_args = [a for a in server_other_args if a not in ("--model-path",)]
    if "--port" in server_other_args:
        i = server_other_args.index("--port")
        del server_other_args[i : i + 2]
    # Surface the env-var requirement so users who forget get a clear error.
    if not envs.SGLANG_ENABLE_FORCED_TOKEN_IDS.get():
        print(
            "[branch_compare] WARNING: SGLANG_ENABLE_FORCED_TOKEN_IDS is "
            "not set in the environment. The server will reject "
            "forced-token requests on the verify phase.",
            file=sys.stderr,
        )
    proc = popen_launch_server(
        args.model_path,
        base_url,
        timeout=TIMEOUT_LAUNCH,
        other_args=server_other_args,
        env=os.environ.copy(),
    )
    print(f"[branch_compare] server launched at {base_url}", flush=True)
    return proc, base_url


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("--model-path is required")

    if args.mode == "record" and args.record_dir is not None:
        raise SystemExit(
            "--record-dir is only valid in --mode verify "
            "(record mode generates a fresh artifact; did you mean --mode verify?)"
        )
    if args.mode == "verify" and args.record_dir is None:
        raise SystemExit("--mode verify requires --record-dir")
    record_meta = (
        artifacts.read_meta(args.record_dir) if args.mode == "verify" else None
    )
    _resolve_inherited(args, record_meta)

    proc = None
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        proc, base_url = _launch_server(args)
    client = transport_mod.make_transport(args.transport, base_url)

    try:
        if args.mode == "record":
            cmd_record(args, client)
        else:
            cmd_verify(args, client, record_meta)
    finally:
        if proc is not None:
            kill_process_tree(proc.pid)


if __name__ == "__main__":
    main()
