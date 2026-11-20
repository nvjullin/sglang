"""Write a branch_compare prompts file whose prompts share long prefixes.

Each prefix is natural English (text files found on the machine: licenses,
Python's own documentation strings), so logit distributions look like real
traffic. Prompts are ordered so that every prefix recurs after others have
pushed it out of a small device KV pool.

    python make_shared_prefix_prompts.py --tokenizer Qwen/Qwen3-8B --out prompts.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pydoc
import random

from transformers import AutoTokenizer


def corpus_texts(min_chars: int) -> list[str]:
    texts = []
    for path in sorted(glob.glob("/usr/share/common-licenses/*")):
        if os.path.isfile(path):
            with open(path, errors="ignore") as f:
                texts.append(f.read())
    for name in ("os", "json", "collections", "itertools", "functools", "re", "argparse", "typing"):
        texts.append(pydoc.render_doc(name, renderer=pydoc.plaintext))
    return [t for t in texts if len(t) >= min_chars]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-prefixes", type=int, default=8)
    p.add_argument("--prefix-tokens", type=int, default=3000)
    p.add_argument("--prompts-per-prefix", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    texts = corpus_texts(min_chars=a.prefix_tokens * 3)
    if len(texts) < a.num_prefixes:
        raise SystemExit(f"only {len(texts)} source texts long enough")
    prefixes = [
        tok.encode(t, add_special_tokens=False)[: a.prefix_tokens]
        for t in texts[: a.num_prefixes]
    ]
    questions = [
        "Summarize the text above in two sentences.",
        "List three key terms used in the text above.",
        "What is the main purpose of the text above?",
        "Rewrite the first paragraph above in plain language.",
        "Which section above would a newcomer find hardest, and why?",
        "Give one example that illustrates the text above.",
        "What would you ask the author of the text above?",
        "Explain the text above to a ten-year-old.",
    ]
    rows = []
    for k in range(a.prompts_per_prefix):
        order = list(range(a.num_prefixes))
        rng.shuffle(order)
        for i in order:
            q = f"\n\nQuestion {k + 1}: {questions[(i + k) % len(questions)]}\nAnswer:"
            rows.append({"input_ids": prefixes[i] + tok.encode(q, add_special_tokens=False)})
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} prompts, {a.num_prefixes} prefixes of {a.prefix_tokens} tokens")


if __name__ == "__main__":
    main()
