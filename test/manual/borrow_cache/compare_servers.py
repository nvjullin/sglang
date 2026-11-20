"""Send one shared-prefix workload to several servers and compare their outputs.

Prompts are random token ids built from a few long shared prefixes, sent over
several rounds in shuffled order, so a server with a small device KV pool must
evict prefixes and bring them back. With deterministic inference on every
server, the generated ids must match exactly.

    python compare_servers.py --urls http://127.0.0.1:30000 http://127.0.0.1:30001
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys

import requests


def make_workload(args) -> list[list[int]]:
    rng = random.Random(args.seed)
    vocab = range(args.vocab_lo, args.vocab_hi)
    prefixes = [
        [rng.choice(vocab) for _ in range(args.prefix_len)]
        for _ in range(args.num_prefixes)
    ]
    prompts = []
    for _ in range(args.rounds):
        batch = [
            prefixes[i % args.num_prefixes]
            + [rng.choice(vocab) for _ in range(rng.randrange(1, args.max_suffix_len))]
            for i in range(args.prompts_per_round)
        ]
        rng.shuffle(batch)
        prompts.extend(batch)
    return prompts


def generate(url: str, ids: list[int], max_new_tokens: int) -> dict:
    r = requests.post(
        f"{url}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 2,
        },
        timeout=600,
    )
    r.raise_for_status()
    out = r.json()
    meta = out["meta_info"]
    # Top-2 logprob gap per step: a flip at a near-tie is numerics, not KV.
    margins = [
        (top[0][0] - top[1][0]) if top and len(top) > 1 else None
        for top in meta.get("output_top_logprobs", [])
    ]
    return {
        "output_ids": out["output_ids"],
        "cached_tokens": meta.get("cached_tokens", 0),
        "margins": margins,
    }


def run(url: str, prompts: list[list[int]], args) -> list[dict]:
    results: list = [None] * len(prompts)
    # Rounds stay in order so later rounds can hit what earlier ones cached.
    per_round = args.prompts_per_round
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for start in range(0, len(prompts), per_round):
            futs = {
                pool.submit(generate, url, prompts[i], args.max_new_tokens): i
                for i in range(start, min(start + per_round, len(prompts)))
            }
            for f in concurrent.futures.as_completed(futs):
                results[futs[f]] = f.result()
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-prefixes", type=int, default=8)
    p.add_argument("--prefix-len", type=int, default=1500)
    p.add_argument("--max-suffix-len", type=int, default=200)
    p.add_argument("--prompts-per-round", type=int, default=32)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--vocab-lo", type=int, default=1000)
    p.add_argument("--vocab-hi", type=int, default=30000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    prompts = make_workload(args)
    runs = {}
    with concurrent.futures.ThreadPoolExecutor(len(args.urls)) as pool:
        futs = {pool.submit(run, url, prompts, args): url for url in args.urls}
        for f in concurrent.futures.as_completed(futs):
            runs[futs[f]] = f.result()

    ref_url = args.urls[0]
    summary = {"num_prompts": len(prompts), "servers": {}}
    ok = True
    for url in args.urls:
        res = runs[url]
        mismatches = [
            i for i, (a, b) in enumerate(zip(runs[ref_url], res))
            if a["output_ids"] != b["output_ids"]
        ]
        divergences = []
        for i in mismatches[:10]:
            a, b = runs[ref_url][i]["output_ids"], res[i]["output_ids"]
            step = next(k for k in range(len(a)) if a[k] != b[k])
            divergences.append(
                {
                    "prompt": i,
                    "step": step,
                    "ref_margin": runs[ref_url][i]["margins"][step],
                    "margin": res[i]["margins"][step],
                }
            )
        summary["servers"][url] = {
            "cached_tokens": sum(r["cached_tokens"] for r in res),
            "mismatched_prompts": len(mismatches),
            "divergences": divergences,
        }
        ok &= not mismatches
    summary["identical"] = ok
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "runs": runs}, f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
