"""Prompt loading for branch_compare.

Two paths:
  1. --eval-name NAME: run the eval with a RecordingSampler (returns empty
     text, captures each message list) and use the captured messages as the
     prompts. Most evals come from sgl-eval (the `test` extra of
     python/pyproject.toml); humaneval and mgsm from sglang.test.simple_eval_*.
  2. --prompts-file FILE: JSONL with `{"prompt": "..."}` per line.

For --api chat we render the captured messages through the model's chat
template via HF AutoTokenizer (client-side); the server receives plain
text on /generate. This guarantees byte-identical input across phases.
For --api completion, we send the last message's content unmodified (or
the JSONL "prompt" field).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.test.simple_eval_common import SamplerBase

# Text-only sgl-eval benchmarks.
SGL_EVAL_NAMES = ("gsm8k", "mmlu", "mmlu_pro", "gpqa", "aime24", "aime25", "aime26")
SIMPLE_EVAL_NAMES = ("humaneval", "mgsm")
EVAL_NAMES = SGL_EVAL_NAMES + SIMPLE_EVAL_NAMES


@dataclass
class Prompt:
    text: str  # what we send to /generate (chat-templated if --api=chat)
    messages: Optional[List[Dict[str, Any]]] = (
        None  # original chat messages for traceability
    )


class RecordingSampler(SamplerBase):
    """Captures every message list an eval sends and answers with empty text.

    Speaks both sampler protocols: sglang.test.simple_eval_* call
    `sampler(messages) -> str`; sgl-eval calls `sampler(messages, gen) -> Sample`
    and expects the sampler to prepend `gen.system_message`.
    """

    def __init__(self):
        self.captured: List[List[Dict[str, Any]]] = []

    def __call__(self, message_list, gen=None):
        if gen is None:
            self.captured.append(message_list)
            return ""
        from sgl_eval.types import Sample

        if gen.system_message:
            message_list = [
                {"role": "system", "content": gen.system_message},
                *message_list,
            ]
        self.captured.append(message_list)
        return Sample(text="")


def _simple_eval_runner(eval_name: str, num_examples: int):
    if eval_name == "humaneval":
        from sglang.test.simple_eval_humaneval import HumanEval

        return HumanEval(num_examples, num_threads=1)
    if eval_name == "mgsm":
        from sglang.test.simple_eval_mgsm import MGSMEval

        return MGSMEval(num_examples, num_threads=1)
    raise ValueError(f"Unsupported eval_name: {eval_name}")


def _sgl_eval_runner(eval_name: str, num_examples: int):
    from sgl_eval.registry import get

    spec = get(eval_name)
    # One thread keeps the captured order equal to the dataset order.
    return lambda sampler: spec.run(
        sampler=sampler,
        gen=spec.default_gen,
        n_repeats=1,
        num_examples=num_examples,
        num_threads=1,
    )


def _capture_messages_via_eval(
    eval_name: str, num_examples: int
) -> List[List[Dict[str, Any]]]:
    """Run the eval with a RecordingSampler. Grading runs on dummy responses
    and may print warnings; we only need the captured message_lists."""
    if eval_name in SGL_EVAL_NAMES:
        run_eval = _sgl_eval_runner(eval_name, num_examples)
    else:
        run_eval = _simple_eval_runner(eval_name, num_examples)
    sampler = RecordingSampler()
    try:
        run_eval(sampler)
    except Exception:
        # Some evals try to grade dummy outputs and crash on the empty string;
        # whatever was captured before the crash is still usable.
        pass
    return sampler.captured[:num_examples]


def _apply_chat_template(tokenizer, messages: List[Dict[str, Any]]) -> str:
    """Render messages to a string the same way the server would on
    /v1/chat/completions, so that record/verify see byte-identical inputs.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _last_user_content(messages: List[Dict[str, Any]]) -> str:
    """For --api completion: pull the last user message's content as a
    plain string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # Multimodal content list
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                return "".join(parts)
    raise ValueError(f"No user message in {messages!r}")


def load_prompts(
    *,
    eval_name: Optional[str],
    num_examples: Optional[int],
    prompts_file: Optional[str],
    api: str,
    tokenizer=None,
) -> List[Prompt]:
    if (eval_name is None) == (prompts_file is None):
        raise ValueError("Exactly one of --eval-name or --prompts-file must be set")

    if prompts_file is not None:
        prompts: List[Prompt] = []
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["prompt"]
                prompts.append(Prompt(text=text, messages=None))
        return prompts

    assert eval_name is not None and num_examples is not None
    captured = _capture_messages_via_eval(eval_name, num_examples)
    if not captured:
        raise RuntimeError(
            f"RecordingSampler captured no messages for eval={eval_name!r}; "
            f"the eval may have crashed before invoking the sampler."
        )

    prompts = []
    for messages in captured:
        if api == "chat":
            if tokenizer is None:
                raise ValueError("--api chat requires a tokenizer")
            text = _apply_chat_template(tokenizer, messages)
        elif api == "completion":
            text = _last_user_content(messages)
        else:
            raise ValueError(f"Unsupported --api: {api}")
        prompts.append(Prompt(text=text, messages=messages))
    return prompts
