"""Prompt loading for branch_compare.

Two paths:
  1. --eval-name NAME: run the eval with a RecordingSampler (returns empty
     text, captures each message list) and use the captured messages as the
     prompts. Most evals come from sgl-eval (the `test` extra of
     python/pyproject.toml); humaneval and mgsm from sglang.test.simple_eval_*.
  2. --prompts-file FILE: JSONL, one prompt per line, as `{"input_ids": [...]}`,
     `{"messages": [...]}` (chat template applied, needs --api chat) or
     `{"prompt": "..."}` (raw text, tokenized as is).

Prompts are tokenized on the client, so the server (and any frontend in front of
it) receives token ids. The record stores them and verify replays the same ids.
For --api chat the captured messages go through the model's chat template with
--chat-template-kwargs (e.g. the thinking switch the benchmark sends); for
--api completion the last user message is tokenized as raw text.
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
    input_ids: List[int]
    # For traceability only: the rendered text and the original chat messages.
    text: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


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


def _chat_prompt(
    tokenizer, messages: List[Dict[str, Any]], chat_template_kwargs: Dict[str, Any]
) -> Prompt:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
    )
    # What apply_chat_template(tokenize=True) does; the template adds the specials.
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    return Prompt(input_ids=input_ids, text=text, messages=messages)


def _text_prompt(tokenizer, text: str, messages=None) -> Prompt:
    return Prompt(input_ids=tokenizer.encode(text), text=text, messages=messages)


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


def _prompt_from_jsonl(
    obj: Dict[str, Any], api: str, tokenizer, chat_template_kwargs: Dict[str, Any]
) -> Prompt:
    if "input_ids" in obj:
        return Prompt(input_ids=[int(t) for t in obj["input_ids"]])
    if "messages" in obj:
        if api != "chat":
            raise ValueError('a {"messages": ...} prompt needs --api chat')
        return _chat_prompt(tokenizer, obj["messages"], chat_template_kwargs)
    return _text_prompt(tokenizer, obj["prompt"])


def load_prompts(
    *,
    eval_name: Optional[str],
    num_examples: Optional[int],
    prompts_file: Optional[str],
    api: str,
    tokenizer,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Prompt]:
    if (eval_name is None) == (prompts_file is None):
        raise ValueError("Exactly one of --eval-name or --prompts-file must be set")
    chat_template_kwargs = chat_template_kwargs or {}

    if prompts_file is not None:
        prompts: List[Prompt] = []
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(
                        _prompt_from_jsonl(
                            json.loads(line), api, tokenizer, chat_template_kwargs
                        )
                    )
        return prompts[:num_examples] if num_examples is not None else prompts

    assert eval_name is not None and num_examples is not None
    captured = _capture_messages_via_eval(eval_name, num_examples)
    if not captured:
        raise RuntimeError(
            f"RecordingSampler captured no messages for eval={eval_name!r}; "
            f"the eval may have crashed before invoking the sampler."
        )

    if api == "chat":
        return [_chat_prompt(tokenizer, m, chat_template_kwargs) for m in captured]
    if api == "completion":
        return [_text_prompt(tokenizer, _last_user_content(m), m) for m in captured]
    raise ValueError(f"Unsupported --api: {api}")
