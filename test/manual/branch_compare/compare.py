"""Compute per-step divergence metrics from two branch_compare artifacts.

Usage (run from sglang-source/test/manual/):
    python -m branch_compare.compare \\
        --record-dir DIR --branch-dir DIR --out-dir DIR

Per (prompt, step) the comparator finds the intersection of main's top-K
token IDs with branch's top-K, looks up the logprob each side assigned to
each common token, and computes their mean absolute difference (nats).
Steps with fewer than 2 common tokens have no value (NaN) and are left out.

These per-step scalars are aggregated into mean/p50/p90/p99/min/max.
The intersection size (`common_topk_count`) per step is also reported - if
it dips below K, the rank ordering itself shifted.

`verdict.json` holds pass/fail under VERDICT_POLICY: the hard checks always run;
each threshold check runs only when its limit is given.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from branch_compare import artifacts


def _per_step_metrics(
    main_top_ids: np.ndarray,  # int32 [K]
    main_top_lps: np.ndarray,  # float32 [K]
    branch_top_ids: np.ndarray,  # int32 [K]
    branch_top_lps: np.ndarray,  # float32 [K]
) -> Tuple[float, int]:
    """Return (mean_abs, common_count)."""
    branch_lookup: Dict[int, float] = {
        int(t): float(lp) for t, lp in zip(branch_top_ids, branch_top_lps)
    }
    common_main_lps: List[float] = []
    common_branch_lps: List[float] = []
    for tid, lp in zip(main_top_ids, main_top_lps):
        tid_int = int(tid)
        if tid_int in branch_lookup:
            common_main_lps.append(float(lp))
            common_branch_lps.append(branch_lookup[tid_int])

    common = len(common_main_lps)
    if common < 2:
        return float("nan"), common

    m = np.asarray(common_main_lps, dtype=np.float64)
    b = np.asarray(common_branch_lps, dtype=np.float64)
    return float(np.abs(m - b).mean()), common


def _aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


VERDICT_POLICY = "branch_compare.v1"


def _token_mismatch(
    record: Dict[str, torch.Tensor], branch: Dict[str, torch.Tensor]
) -> str | None:
    """Why the two artifacts did not see the same tokens, or None. Their logprobs
    are comparable only when the prompt and every emitted token match."""
    if not torch.equal(record["input_ids"], branch["input_ids"]):
        return "input_ids differ"
    if not torch.equal(record["output_ids"], branch["output_ids"]):
        return "output_ids differ"
    for side, a in (("record", record), ("branch", branch)):
        n = a["output_ids"].numel()
        rows = (
            a["top_k_token_ids"].shape[0],
            a["top_k_logprobs"].shape[0],
            a["chosen_logprob"].numel(),
        )
        if any(r != n for r in rows):
            return f"{side} logprob rows {rows} != {n} output tokens"
    return None


def _check(name: str, ok: bool, value: Any, limit: Any = None) -> Dict[str, Any]:
    return {"name": name, "pass": bool(ok), "value": value, "limit": limit}


def _verdict(
    args: argparse.Namespace,
    record_meta: Dict[str, Any],
    branch_meta: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    record_idx = {p["idx"] for p in record_meta["prompts"]}
    branch_idx = {p["idx"] for p in branch_meta["prompts"]}
    abs_diff = summary["mean_abs_diff"]
    common = summary["common_topk_count"]
    topk = summary["topk"] or 0
    mismatched = {m["idx"] for m in summary["token_mismatches"]} | {
        m["prompt_idx"] for m in summary["failed_steps"]
    }
    checks = [
        # A verify of a record, not a second record (whose sampled tokens may match).
        _check(
            "branch_is_verify",
            branch_meta.get("phase") == "verify",
            branch_meta.get("phase"),
            "verify",
        ),
        # Forcing failed: the two sides saw different prefixes.
        _check("forced_tokens_match", not mismatched, len(mismatched), 0),
        # A failed request leaves its prompt unverified.
        _check(
            "all_prompts_verified",
            record_idx <= branch_idx,
            len(record_idx - branch_idx),
            0,
        ),
        # No step shares two top-K tokens: a different model, not drift.
        _check("measurable_steps", abs_diff["n"] > 0, abs_diff["n"]),
    ]
    finite = abs_diff["n"] > 0
    if args.max_mean_abs_diff is not None:
        checks.append(
            _check(
                "mean_abs_diff",
                finite and abs_diff["mean"] <= args.max_mean_abs_diff,
                abs_diff["mean"],
                args.max_mean_abs_diff,
            )
        )
    if args.max_p99_abs_diff is not None:
        checks.append(
            _check(
                "p99_abs_diff",
                finite and abs_diff["p99"] <= args.max_p99_abs_diff,
                abs_diff["p99"],
                args.max_p99_abs_diff,
            )
        )
    chosen = summary["chosen_abs_diff"]
    if args.max_chosen_abs_diff_mean is not None:
        checks.append(
            _check(
                "chosen_abs_diff_mean",
                chosen["n"] > 0 and chosen["mean"] <= args.max_chosen_abs_diff_mean,
                chosen["mean"],
                args.max_chosen_abs_diff_mean,
            )
        )
    if args.min_common_topk_frac is not None:
        frac = common["mean"] / topk if topk and common["n"] > 0 else float("nan")
        checks.append(
            _check(
                "common_topk_frac",
                frac >= args.min_common_topk_frac,
                frac,
                args.min_common_topk_frac,
            )
        )
    return {
        "policy": VERDICT_POLICY,
        "pass": all(c["pass"] for c in checks),
        "checks": checks,
        "record_dir": os.path.abspath(args.record_dir),
        "branch_dir": os.path.abspath(args.branch_dir),
        "topk": topk,
        "mean_abs_diff": abs_diff,
        "common_topk_count": common,
        "chosen_abs_diff": chosen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--branch-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--max-mean-abs-diff",
        type=float,
        default=None,
        help="Fail when the mean abs logprob diff (nats) exceeds this.",
    )
    parser.add_argument(
        "--max-p99-abs-diff",
        type=float,
        default=None,
        help="Fail when the p99 of the per-step abs diff exceeds this.",
    )
    parser.add_argument(
        "--max-chosen-abs-diff-mean",
        type=float,
        default=None,
        help="Fail when the mean |logprob diff| of the emitted token exceeds this. "
        "Unlike the top-K mean, the low-probability tail does not dominate it.",
    )
    parser.add_argument(
        "--min-common-topk-frac",
        type=float,
        default=None,
        help="Fail when the mean common top-K count / K falls below this.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    record_meta = artifacts.read_meta(args.record_dir)
    branch_meta = artifacts.read_meta(args.branch_dir)

    record_by_idx = {p["idx"]: p for p in record_meta["prompts"]}

    abs_all: List[float] = []
    common_all: List[float] = []
    # |record - verify| logprob of the emitted (forced) token, per step.
    chosen_all: List[float] = []
    per_prompt: List[Dict[str, Any]] = []
    token_mismatches: List[Dict[str, Any]] = []

    for entry in branch_meta["prompts"]:
        idx = entry["idx"]
        rec = record_by_idx.get(idx)
        if rec is None:
            print(f"[branch_compare] no record entry for idx={idx}, skipping")
            continue

        main_a = artifacts.read_prompt_artifact(args.record_dir, idx)
        branch_a = artifacts.read_prompt_artifact(args.branch_dir, idx)

        reason = _token_mismatch(main_a, branch_a)
        if reason is not None:
            print(f"[branch_compare] idx={idx}: {reason}")
            token_mismatches.append({"idx": idx, "reason": reason})

        # On a mismatch, the metrics below still cover the common prefix of rows.
        n_steps = min(
            main_a["top_k_token_ids"].shape[0],
            branch_a["top_k_token_ids"].shape[0],
        )

        # float32 for numpy (artifacts written before float32 storage are bf16).
        main_top_lps_f32 = main_a["top_k_logprobs"].float().numpy()
        branch_top_lps_f32 = branch_a["top_k_logprobs"].float().numpy()
        main_top_ids = main_a["top_k_token_ids"].numpy()
        branch_top_ids = branch_a["top_k_token_ids"].numpy()

        n_ids = min(
            n_steps,
            *(
                a[k].numel()
                for a in (main_a, branch_a)
                for k in ("output_ids", "chosen_logprob")
            ),
        )
        same = main_a["output_ids"][:n_ids] == branch_a["output_ids"][:n_ids]
        chosen = (
            main_a["chosen_logprob"][:n_ids].float()
            - branch_a["chosen_logprob"][:n_ids].float()
        ).abs()
        chosen_all.extend(chosen[same].tolist())

        prompt_abs: List[float] = []
        for s in range(n_steps):
            mabs, common = _per_step_metrics(
                main_top_ids[s],
                main_top_lps_f32[s],
                branch_top_ids[s],
                branch_top_lps_f32[s],
            )
            abs_all.append(mabs)
            common_all.append(float(common))
            prompt_abs.append(mabs)

        per_prompt.append(
            {
                "idx": idx,
                "n_steps": n_steps,
                "abs_diff": _aggregate(prompt_abs),
            }
        )

    summary: Dict[str, Any] = {
        "record_commit": record_meta.get("git_commit"),
        "branch_commit": branch_meta.get("git_commit"),
        "record_server_args": record_meta.get("server_args"),
        "branch_server_args": branch_meta.get("server_args"),
        "topk": record_meta.get("topk_logprobs"),
        "mean_abs_diff": _aggregate(abs_all),
        "common_topk_count": _aggregate(common_all),
        "chosen_abs_diff": _aggregate(chosen_all),
        "token_mismatches": token_mismatches,
        "failed_steps": branch_meta.get("failed_steps", []),
        "failed_requests": branch_meta.get("failed_requests", []),
        "per_prompt": per_prompt,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[branch_compare] wrote {summary_path}")

    verdict = _verdict(args, record_meta, branch_meta, summary)
    verdict_path = os.path.join(args.out_dir, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    failed = [c["name"] for c in verdict["checks"] if not c["pass"]]
    print(
        f"[branch_compare] verdict {'PASS' if verdict['pass'] else 'FAIL'}"
        + (f" ({', '.join(failed)})" if failed else "")
        + f"; wrote {verdict_path}"
    )

    # Terminal histograms (also archived to histograms.txt). The panels
    # share the same width / bin count for easy visual comparison.
    hist_text = _render_histograms(
        [
            ("Mean abs diff", abs_all, summary["mean_abs_diff"]),
            ("Common top-K count", common_all, summary["common_topk_count"]),
        ],
        n_bins=24,
        width=48,
    )
    print(hist_text)
    hist_path = os.path.join(args.out_dir, "histograms.txt")
    with open(hist_path, "w") as f:
        f.write(hist_text)
    print(f"[branch_compare] wrote {hist_path}")


# --- ASCII histogram rendering ----------------------------------------------

_BLOCKS = " ▏▎▍▌▋▊▉█"  # 0/8, 1/8, ..., 8/8 of a cell


def _bar(fraction: float, width: int) -> str:
    """Render a horizontal bar of `fraction` (0..1) using sub-block chars."""
    fraction = max(0.0, min(1.0, fraction))
    total_eighths = int(round(fraction * width * 8))
    full = total_eighths // 8
    rem = total_eighths % 8
    out = "█" * full
    if rem and full < width:
        out += _BLOCKS[rem]
    return out.ljust(width)


def _fmt(v: float, *, integer: bool = False) -> str:
    """Format a value for the bin-edge label column."""
    if not np.isfinite(v):
        return "nan"
    if integer:
        return str(int(round(v)))
    av = abs(v)
    if av == 0.0:
        return "0"
    if av < 1e-3 or av >= 1e4:
        return f"{v:.3e}"
    return f"{v:.6g}"


def _ascii_histogram(
    title: str,
    values: List[float],
    aggregates: Dict[str, float],
    *,
    n_bins: int = 24,
    width: int = 48,
) -> str:
    """Render one labeled histogram + summary line as ASCII."""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    lines: List[str] = []
    lines.append(f"=== {title} (n={clean.size}) ===")

    if clean.size == 0:
        lines.append("  (no data)")
        return "\n".join(lines)

    lo = float(clean.min())
    hi = float(clean.max())
    is_int = bool(np.all(clean == np.floor(clean)))

    if lo == hi or _fmt(lo, integer=is_int) == _fmt(hi, integer=is_int):
        # Range is zero, or so small that every bin edge renders to the
        # same label at display precision. Collapse to one row.
        lines.append(
            f"  {_fmt(lo, integer=is_int):>12s}  {_bar(1.0, width)} {clean.size}"
        )
    elif is_int and (int(hi) - int(lo)) < n_bins:
        # One bin per distinct integer value — no fractional edges.
        ints = np.arange(int(lo), int(hi) + 1, dtype=np.int64)
        counts = np.array([int((clean == v).sum()) for v in ints])
        max_count = int(counts.max()) if counts.size else 0
        for v, c in zip(ints, counts):
            frac = (c / max_count) if max_count else 0.0
            lines.append(
                f"  {_fmt(int(v), integer=True):>12s}  {_bar(frac, width)} {int(c)}"
            )
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(clean, bins=edges)
        max_count = int(counts.max()) if counts.size else 0
        for i, c in enumerate(counts):
            edge_left = edges[i]
            frac = (c / max_count) if max_count else 0.0
            lines.append(
                f"  {_fmt(edge_left, integer=is_int):>12s}  {_bar(frac, width)} {int(c)}"
            )
        lines.append(f"  {_fmt(edges[-1], integer=is_int):>12s}  (upper edge)")

    s = aggregates
    lines.append(
        f"  mean={_fmt(s['mean'])}  p50={_fmt(s['p50'])}  "
        f"p90={_fmt(s['p90'])}  p99={_fmt(s['p99'])}  "
        f"min={_fmt(s['min'], integer=is_int)}  max={_fmt(s['max'], integer=is_int)}"
    )
    return "\n".join(lines)


def _render_histograms(
    panels: List[Tuple[str, List[float], Dict[str, float]]],
    *,
    n_bins: int,
    width: int,
) -> str:
    return "\n\n".join(
        _ascii_histogram(title, vals, aggs, n_bins=n_bins, width=width)
        for title, vals, aggs in panels
    )


if __name__ == "__main__":
    main()
