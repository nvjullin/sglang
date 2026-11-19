"""How branch_compare sends one request and reads back per-step logprobs.

Every transport takes token ids in, so record and verify feed the server the
same prompt tokens whatever tokenizer or chat template the frontend runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from sglang.srt.sampling.sampling_params import (
    FORCED_TOKEN_IDS_KEY,
    FORCED_TOKEN_IDS_PATH_KEY,
)

TIMEOUT_REQUEST = 3600
TRANSPORTS = ("sglang", "dynamo")


@dataclass
class Forced:
    """The ids to force, sent inline or as a path the server workers can read."""

    ids: Optional[List[int]] = None
    path: Optional[str] = None

    def custom_params(self) -> Dict[str, Any]:
        if self.path is not None:
            return {FORCED_TOKEN_IDS_PATH_KEY: self.path}
        return {FORCED_TOKEN_IDS_KEY: self.ids}


@dataclass
class Result:
    output_ids: List[int]
    chosen_logprob: List[float]
    top_k_token_ids: List[List[int]]
    top_k_logprobs: List[List[float]]
    finish_reason: str


def _generate_payload(
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    topk_logprobs: int,
    forced: Optional[Forced],
    *,
    stream: bool,
) -> Dict[str, Any]:
    sampling_params = dict(sampling_params)
    if forced is not None:
        sampling_params["custom_params"] = forced.custom_params()
    return {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": topk_logprobs,
        "return_text_in_logprobs": False,
        "stream": stream,
    }


class SglangTransport:
    """sglang's native /generate, directly or through the sglang router."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        topk_logprobs: int,
        forced: Optional[Forced] = None,
    ) -> Result:
        payload = _generate_payload(
            input_ids, sampling_params, topk_logprobs, forced, stream=False
        )
        r = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=TIMEOUT_REQUEST
        )
        if not r.ok:
            raise RuntimeError(f"/generate returned {r.status_code}: {r.text[:2000]}")
        return _parse_sglang_meta(r.json()["meta_info"])


class DynamoTransport:
    """Dynamo's native SGLang /generate. The frontend needs
    DYN_SGLANG_ENABLE_GENERATE=1 and every sglang worker DYN_SGL_ALLOW_TOP_LOGPROBS=1;
    the body reaches prefill and decode unchanged except n and max_new_tokens on
    prefill. Only the streaming (SSE) form is served."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        topk_logprobs: int,
        forced: Optional[Forced] = None,
    ) -> Result:
        payload = _generate_payload(
            input_ids, sampling_params, topk_logprobs, forced, stream=True
        )
        with requests.post(
            f"{self.base_url}/generate",
            json=payload,
            stream=True,
            timeout=TIMEOUT_REQUEST,
        ) as r:
            if not r.ok:
                raise RuntimeError(
                    f"/generate returned {r.status_code}: {r.text[:2000]}"
                )
            meta = _accumulate_sse_meta(r.iter_lines(decode_unicode=True))
        return _parse_sglang_meta(meta)


_LOGPROB_KEYS = ("output_token_logprobs", "output_top_logprobs")


def _accumulate_sse_meta(lines) -> Dict[str, Any]:
    """The final meta_info of a streamed sglang response, with the logprob lists of
    every chunk joined. A worker with --incremental-streaming-output sends each
    chunk's new steps; otherwise every chunk holds all steps so far."""
    meta: Dict[str, Any] = {}
    logprobs: Dict[str, List[Any]] = {key: [] for key in _LOGPROB_KEYS}
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError(f"/generate stream error: {chunk['error']}")
        meta = chunk.get("meta_info") or meta
        completion_tokens = meta.get("completion_tokens")
        for key in _LOGPROB_KEYS:
            steps = meta.get(key) or []
            if len(steps) == completion_tokens:
                logprobs[key] = list(steps)
            else:
                logprobs[key].extend(steps)
    if not meta:
        raise RuntimeError("/generate stream ended without a response")
    return {**meta, **logprobs}


def _parse_sglang_meta(meta: Dict[str, Any]) -> Result:
    output_ids: List[int] = []
    chosen_logprob: List[float] = []
    for lp, tid, _ in meta["output_token_logprobs"]:
        output_ids.append(int(tid))
        chosen_logprob.append(float(lp))
    top_ids = [[int(t[1]) for t in step] for step in meta["output_top_logprobs"]]
    top_lps = [[float(t[0]) for t in step] for step in meta["output_top_logprobs"]]
    finish = (meta.get("finish_reason") or {}).get("type", "unknown")
    return Result(output_ids, chosen_logprob, top_ids, top_lps, finish)


def make_transport(name: str, base_url: str):
    if name == "sglang":
        return SglangTransport(base_url)
    if name == "dynamo":
        return DynamoTransport(base_url)
    raise ValueError(f"Unknown transport {name!r}; choose from {TRANSPORTS}")
