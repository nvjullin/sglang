"""EAGLE verify under SGLANG_ENABLE_FORCED_TOKEN_IDS: a request with forced ids
accepts a draft only while it matches the forced sequence and emits the forced
token as its bonus, whatever the target model and the drafts propose. Greedy
verify forces the argmax; sampled verify forces through one-hot target
probabilities, which the tree kernel must turn into exactly the forced token."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.forced_token_ids import ForcedTokenIds
from sglang.srt.sampling.sampling_params import FORCED_TOKEN_IDS_KEY
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

TOKENS = 5
VOCAB = 97
BATCH = 16


def _req(prompt_len, forced):
    custom_params = None if forced is None else {FORCED_TOKEN_IDS_KEY: forced}
    return SimpleNamespace(
        origin_input_ids=[0] * prompt_len,
        sampling_params=SimpleNamespace(custom_params=custom_params),
    )


def _verify_round(*, greedy, generator):
    """A chain verify round in which request i follows its forced sequence for a
    random number of drafts and then proposes random tokens. Even rows are forced;
    odd rows are not, and run the ordinary greedy or sampled verify."""
    device = "cuda"
    prompt_lens = torch.randint(3, 50, (BATCH,), generator=generator)
    # Tokens already generated: the next forced index is seq_len + 1 - prompt_len.
    generated = torch.randint(0, 20, (BATCH,), generator=generator)
    seq_lens = prompt_lens + generated
    forced = {
        i: torch.randint(
            0, VOCAB, (int(generated[i]) + TOKENS + 3,), generator=generator
        )
        for i in range(0, BATCH, 2)
    }
    keep = torch.randint(0, TOKENS, (BATCH,), generator=generator)
    drafts = torch.randint(0, VOCAB, (BATCH, TOKENS), generator=generator)
    for i, ids in forced.items():
        start = int(generated[i]) + 1  # forced index of the root's prediction
        n = int(keep[i])
        drafts[i, 1 : 1 + n] = ids[start : start + n]
        if 1 + n < TOKENS:  # the first draft past the prefix must miss
            drafts[i, 1 + n] = (ids[start + n] + 1) % VOCAB
    reqs = [
        _req(int(prompt_lens[i]), forced[i].tolist() if i in forced else None)
        for i in range(BATCH)
    ]
    table = ForcedTokenIds.from_reqs(reqs, device)
    positions = seq_lens[:, None] + torch.arange(TOKENS)
    next_token = torch.arange(1, TOKENS + 1)
    next_token[-1] = -1
    verify_input = SimpleNamespace(
        topk=1,
        tree_topk=1,
        draft_probs=None,
        draft_token=drafts.flatten().to(device),
        draft_token_num=TOKENS,
        max_tree_depth=TOKENS,
        positions=positions.flatten().to(device),
        retrieve_index=torch.arange(BATCH * TOKENS, device=device).view(BATCH, TOKENS),
        retrieve_next_token=next_token.expand(BATCH, TOKENS).contiguous().to(device),
        retrieve_next_sibling=torch.full((BATCH, TOKENS), -1, device=device),
    )
    sampling_info = SimpleNamespace(
        is_all_greedy=greedy,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        step_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        watermark=None,
        forced_token_ids=table,
        temperatures=torch.ones((BATCH, 1), device=device),
        top_ks=None,
        top_ps=None,
    )
    batch = SimpleNamespace(
        device=device,
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=seq_lens.to(device),
        req_pool_indices=torch.arange(BATCH, device=device),
        sampling_info=sampling_info,
        return_logprob=False,
    )
    logits = torch.randn(BATCH * TOKENS, VOCAB, generator=generator).to(device)
    return batch, verify_input, logits, forced, generated, keep


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestEagleForcedTokens(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        stack = ExitStack()
        self.addCleanup(stack.close)
        # One rank: sampled verify broadcasts its result over the TP group.
        stack.enter_context(
            get_parallel().override(tp_group=SimpleNamespace(world_size=1))
        )

    def _check(self, greedy):
        generator = torch.Generator().manual_seed(0)
        for _ in range(8):
            batch, verify_input, logits, forced, generated, keep = _verify_round(
                greedy=greedy, generator=generator
            )
            predict, accept_lens, accept_index = eagle_sample(
                verify_input,
                batch,
                SimpleNamespace(
                    next_token_logits=logits, next_token_logits_vocab_shard=None
                ),
            )
            accept_lens = accept_lens.cpu()
            emitted = predict[accept_index.clamp(min=0)].cpu()
            for i, ids in forced.items():
                start = int(generated[i]) + 1
                n = int(accept_lens[i])
                # Matching drafts are accepted; the first mismatch ends the chain.
                self.assertEqual(n, int(keep[i]) + 1, f"request {i}")
                self.assertEqual(
                    emitted[i, :n].tolist(), ids[start : start + n].tolist()
                )

    def test_greedy_verify_emits_the_forced_tokens(self):
        self._check(greedy=True)

    def test_sampled_verify_emits_the_forced_tokens(self):
        self._check(greedy=False)


if __name__ == "__main__":
    unittest.main()
