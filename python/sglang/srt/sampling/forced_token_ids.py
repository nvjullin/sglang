"""Debug-only forced decoding (SGLANG_ENABLE_FORCED_TOKEN_IDS): a batch's forced
output ids, looked up on the device by token position.

The token emitted from a logits row at KV position ``p`` lands at ``p + 1``, so its
index into the request's forced ids is ``p + 1 - prompt_len``. Positions are already
on the device for every forward (prefill, decode, speculative verify, and under the
overlap scheduler), so no host-side step counter is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import msgspec
import torch

from sglang.srt.sampling.sampling_params import get_forced_token_ids
from sglang.srt.utils.device import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# Lookup result for a row whose emitted token is not forced.
NOT_FORCED = -1


class ForcedTokenIds(msgspec.Struct, frozen=True):
    """Forced ids of one batch's requests, in batch order.

    Rebuilt (never mutated) when the batch's requests change, so a forward that
    holds an older copy keeps reading consistent tensors.
    """

    # Per request: its forced ids as an int32 view of `ids`, or None.
    segments: tuple[Optional[torch.Tensor], ...]
    # Per request: len(origin_input_ids); 0 for an unforced request.
    prompt_lens: tuple[int, ...]
    # [total] int32: every forced request's ids, back to back.
    ids: torch.Tensor
    # [bs] int64 each; a length of 0 marks an unforced request.
    starts: torch.Tensor
    lengths: torch.Tensor
    prompt_lens_device: torch.Tensor

    @classmethod
    def from_reqs(cls, reqs: Sequence[Req], device: str) -> Optional[ForcedTokenIds]:
        """The batch's table, or None when no request is forced."""
        lists = [
            get_forced_token_ids(r.sampling_params.custom_params) or None for r in reqs
        ]
        if all(ids is None for ids in lists):
            return None
        flat = [token for ids in lists if ids is not None for token in ids]
        flat_device = torch.tensor(
            flat, dtype=torch.int32, pin_memory=is_pin_memory_available(device)
        ).to(device, non_blocking=True)
        segments: List[Optional[torch.Tensor]] = []
        offset = 0
        for ids in lists:
            if ids is None:
                segments.append(None)
                continue
            segments.append(flat_device[offset : offset + len(ids)])
            offset += len(ids)
        return cls._from_segments(
            segments,
            prompt_lens=[
                len(r.origin_input_ids) if ids is not None else 0
                for r, ids in zip(reqs, lists)
            ],
            device=device,
        )

    def filter(self, keep_indices: Sequence[int]) -> Optional[ForcedTokenIds]:
        segments = [self.segments[i] for i in keep_indices]
        if all(s is None for s in segments):
            return None
        return self._from_segments(
            segments,
            prompt_lens=[self.prompt_lens[i] for i in keep_indices],
            device=self.ids.device,
        )

    @classmethod
    def merge(
        cls,
        lhs: Optional[ForcedTokenIds],
        rhs: Optional[ForcedTokenIds],
        *,
        lhs_len: int,
        rhs_len: int,
    ) -> Optional[ForcedTokenIds]:
        if lhs is None and rhs is None:
            return None
        device = (lhs if lhs is not None else rhs).ids.device
        lhs_segments = lhs.segments if lhs is not None else (None,) * lhs_len
        rhs_segments = rhs.segments if rhs is not None else (None,) * rhs_len
        lhs_prompt_lens = lhs.prompt_lens if lhs is not None else (0,) * lhs_len
        rhs_prompt_lens = rhs.prompt_lens if rhs is not None else (0,) * rhs_len
        return cls._from_segments(
            lhs_segments + rhs_segments,
            prompt_lens=lhs_prompt_lens + rhs_prompt_lens,
            device=device,
        )

    @classmethod
    def _from_segments(
        cls,
        segments: Sequence[Optional[torch.Tensor]],
        *,
        prompt_lens: Sequence[int],
        device,
    ) -> ForcedTokenIds:
        present = [s for s in segments if s is not None]
        ids = present[0] if len(present) == 1 else torch.cat(present)
        lengths = [0 if s is None else s.numel() for s in segments]
        starts = []
        offset = 0
        for length in lengths:
            starts.append(offset)
            offset += length
        # Re-slice from the new `ids`, so the buffers the old segments viewed can be freed.
        new_segments = tuple(
            None if s is None else ids[start : start + length]
            for s, start, length in zip(segments, starts, lengths)
        )
        meta = torch.tensor(
            [starts, lengths, list(prompt_lens)],
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(device),
        ).to(device, non_blocking=True)
        return cls(
            segments=new_segments,
            prompt_lens=tuple(prompt_lens),
            ids=ids,
            starts=meta[0],
            lengths=meta[1],
            prompt_lens_device=meta[2],
        )

    def __len__(self) -> int:
        return len(self.segments)

    def lookup(self, positions: torch.Tensor, rows_per_req: int) -> torch.Tensor:
        """int32 [len(self) * rows_per_req]: the forced id of the token emitted from
        each logits row, or NOT_FORCED. ``positions`` holds each row's KV position,
        request-major; rows past the batch (DP padding) are ignored."""
        bs = len(self)
        num_rows = bs * rows_per_req
        row_positions = positions[:num_rows].reshape(bs, rows_per_req).to(torch.int64)
        forced_index = row_positions + 1 - self.prompt_lens_device.unsqueeze(1)
        is_forced = (forced_index >= 0) & (forced_index < self.lengths.unsqueeze(1))
        flat_index = (self.starts.unsqueeze(1) + forced_index).clamp_(
            0, self.ids.numel() - 1
        )
        return torch.where(is_forced, self.ids[flat_index], NOT_FORCED).reshape(
            num_rows
        )


def force_tokens(tokens: torch.Tensor, forced: torch.Tensor) -> torch.Tensor:
    """``tokens`` with every row that has a forced id replaced by it; ``forced`` is
    flat and ``tokens`` any shape of the same numel (e.g. [bs, draft_token_num])."""
    forced = forced.view(tokens.shape)
    return torch.where(forced.ge(0), forced.to(tokens.dtype), tokens)


def force_probs_(probs: torch.Tensor, forced: torch.Tensor) -> None:
    """Make each row of ``probs`` [rows, vocab] that has a forced id one-hot at it."""
    is_forced = forced.ge(0).unsqueeze(1)
    probs.masked_fill_(is_forced, 0.0)
    probs.scatter_add_(
        1, forced.clamp(min=0).to(torch.int64).unsqueeze(1), is_forced.to(probs.dtype)
    )
