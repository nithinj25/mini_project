"""Engine tests: tiny random Qwen2 models, no downloads. Runs on CUDA when available, else CPU.

Run: python test_specdec.py   (force CPU: python test_specdec.py --cpu)
"""
import copy
import sys
import time

import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from specdec import (FixedGamma, GammaController, assert_compatible, baseline_generate, best_gamma,
                     expected_tokens, predicted_speedup, rollback, speculative_generate, to_probs,
                     verify)
from winperf import disable_power_throttling

DEVICE = torch.device("cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu")
VOCAB = 32
PROMPT = [1, 5, 9, 2, 7]


def tiny_model(seed: int, vocab: int = VOCAB, init: float = 0.5) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      initializer_range=init, tie_word_embeddings=False)
    # float64 so batched verification and one-at-a-time decoding agree to the last bit that matters
    return Qwen2ForCausalLM(cfg).double().to(DEVICE).eval()


def noisy_copy(model, noise: float, seed: int):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    with torch.no_grad():
        for w in m.parameters():
            w.add_(noise * torch.randn_like(w))
    return m


def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float((a - b).abs().sum())


# --------------------------------------------------------------------------- tests

def test_verify_lossless():
    """Output of one verify step ~ p exactly, whatever q is. Mutation (residual -> p) gives TV ≈ 0.19."""
    torch.manual_seed(0)
    V, N = 8, 40_000
    p = torch.softmax(torch.randn(V) * 1.5, -1).to(DEVICE)
    q = torch.softmax(torch.randn(V) * 1.5, -1).to(DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    counts = torch.zeros(V, device=DEVICE)
    xs = torch.multinomial(q, N, replacement=True, generator=gen).tolist()
    for x in xs:
        _, tok = verify([x], q[None], torch.stack([p, p]), gen)  # row 1 (bonus) unused on rejection
        counts[tok if _ == 0 else x] += 1
    d = tv(counts / N, p)
    print(f"  verify TV(out, p) = {d:.4f}   TV(q, p) = {tv(q, p):.3f}")
    assert d < 0.015, d


def test_greedy_matches_baseline():
    """Greedy speculative output == greedy baseline, for a bad draft and a good draft."""
    target = tiny_model(0)
    bad = tiny_model(123)
    good = noisy_copy(target, 0.02, seed=7)
    ref, _ = baseline_generate(target, PROMPT, 60)
    alphas = {}
    for name, draft in [("bad", bad), ("good", good)]:
        for g in (1, 3, 5):
            out, st = speculative_generate(target, draft, PROMPT, 60, FixedGamma(g))
            assert out == ref, (name, g)
            alphas[(name, g)] = st.alpha
    a_bad = sum(v for k, v in alphas.items() if k[0] == "bad") / 3
    a_good = sum(v for k, v in alphas.items() if k[0] == "good") / 3
    print(f"  alpha bad draft ~ {a_bad:.2f}   alpha good draft ~ {a_good:.2f}")
    assert a_bad < 0.3 and a_good > 0.5, (a_bad, a_good)


def test_draft_equals_target():
    """Draft = target: every guess is kept (α = 1 greedy, ≈ 1 sampling) and a round commits γ+1 tokens."""
    target = tiny_model(0)
    out, st = speculative_generate(target, target, PROMPT, 41, FixedGamma(4))
    assert st.alpha == 1.0 and st.target_passes == 41 // 5 + (41 % 5 > 0), (st.alpha, st.target_passes)
    _, st = speculative_generate(target, target, PROMPT, 60, FixedGamma(4), temperature=1.0, seed=3)
    print(f"  greedy alpha = 1.0, sampling alpha = {st.alpha:.4f}")
    assert st.alpha > 0.99, st.alpha


def test_two_token_joint():
    """Sampling at T=1: joint law of the first two tokens matches exact enumeration from the target.

    3 tokens are generated so γ is not capped below 2 (a round commits up to γ+1 tokens).
    γ=1 exercises the cross-round rollback path; γ=2 exercises within-round commits.
    """
    V = 6
    target = tiny_model(0, vocab=V, init=0.4)
    draft = tiny_model(99, vocab=V, init=0.4)
    with torch.no_grad():
        p1 = to_probs(target(torch.tensor([[1, 5, 2]], device=DEVICE)).logits[0, -1], 1.0)
        exact = torch.stack([p1[a] * to_probs(target(torch.tensor([[1, 5, 2] + [a]], device=DEVICE)).logits[0, -1], 1.0)
                             for a in range(V)])
    N = 3000
    for g in (1, 2):
        counts = torch.zeros(V, V, device=DEVICE)
        for s in range(N):
            out, _ = speculative_generate(target, draft, [1, 5, 2], 3, FixedGamma(g), temperature=1.0, seed=s)
            counts[out[0], out[1]] += 1
        d = tv(counts / N, exact)
        print(f"  gamma={g}: TV(joint, exact) = {d:.4f}")
        assert d < 0.06, (g, d)


def test_eos_and_length():
    target = tiny_model(0)
    draft = noisy_copy(target, 0.02, seed=7)
    for g in (1, 4, 8):
        for n in (1, 2, 5, 13):
            out, st = speculative_generate(target, draft, PROMPT, n, FixedGamma(g))
            assert len(out) == n == st.new_tokens, (g, n, len(out))
    ref, _ = baseline_generate(target, PROMPT, 40)
    eos = ref[len(ref) // 2]
    stop = ref.index(eos) + 1
    for g in (1, 4, 8):
        out, _ = speculative_generate(target, draft, PROMPT, 40, FixedGamma(g), eos_token_id=[eos, 10_000])
        assert out == ref[:stop], g
    base, _ = baseline_generate(target, PROMPT, 40, eos_token_id=eos)
    assert base == ref[:stop]


def test_formulas_controller_and_cache():
    assert abs(expected_tokens(0.8, 4) - 3.3616) < 1e-9
    assert abs(predicted_speedup(0.8, 4, 0.3) - 3.3616 / 2.2) < 1e-9
    assert expected_tokens(1.0, 4) == 5
    # break-even at γ=1: speculation pays only if α > c
    assert predicted_speedup(0.5, 1, 0.4) > 1 > predicted_speedup(0.3, 1, 0.4)
    assert best_gamma(0.9, 0.05) > best_gamma(0.6, 0.05) >= best_gamma(0.6, 0.4) == 1
    ctl = GammaController(c=0.1)
    for _ in range(30):
        ctl.update(8, 8, 8)
    hi = ctl.next_gamma()
    for _ in range(100):  # α̂ weights by tokens tested, so failures need more rounds
        ctl.update(0, 1, 8)
    assert hi == 8 and ctl.next_gamma() == 1, (hi, ctl.next_gamma())
    ctl.reset()
    assert ctl.alpha_hat == 0.7
    # rollback: negative crop, and a no-op (not a wipe) when nothing is extra
    m = tiny_model(0)
    cache = DynamicCache()
    m(input_ids=torch.tensor([PROMPT], device=DEVICE), past_key_values=cache, use_cache=True)
    rollback(cache, 5)
    assert cache.get_seq_length() == 5
    rollback(cache, 3)
    assert cache.get_seq_length() == 3


def test_tokenizer_mismatch_refused():
    class Tok:
        def __init__(self, v):
            self.v = v

        def get_vocab(self):
            return self.v

    m = tiny_model(0)
    assert_compatible(Tok({"a": 0}), Tok({"a": 0}), m, m)
    for a, b, other in [(Tok({"a": 0}), Tok({"b": 0}), m), (Tok({"a": 0}), Tok({"a": 0}), tiny_model(0, vocab=40))]:
        try:
            assert_compatible(a, b, m, other)
        except ValueError:
            continue
        raise AssertionError("incompatible pair accepted")


def test_on_round_hook():
    """The display hook sees exactly the returned tokens, does not change them, and is not timed."""
    target = tiny_model(0)
    draft = noisy_copy(target, 0.02, seed=7)
    for temp in (0.0, 1.0):
        ref, _ = speculative_generate(target, draft, PROMPT, 40, FixedGamma(3), temperature=temp, seed=5)
        seen, from_draft = [], 0

        def hook(committed, n_draft, rejected):
            nonlocal from_draft
            assert 0 <= n_draft <= len(committed) and (rejected is None or isinstance(rejected, int))
            seen.extend(committed)
            from_draft += n_draft
            time.sleep(0.01)

        out, st = speculative_generate(target, draft, PROMPT, 40, FixedGamma(3), temperature=temp, seed=5,
                                       on_round=hook)
        assert out == ref == seen, temp
        # no EOS and γ is capped to the remaining budget, so no round is truncated
        assert from_draft == st.accepted and st.callback_s >= 0.01 * st.rounds
        base, _ = baseline_generate(target, PROMPT, 40, temperature=temp, seed=5)
        seen, from_draft = [], 0
        out, st = baseline_generate(target, PROMPT, 40, temperature=temp, seed=5, on_round=hook)
        assert out == base == seen and from_draft == 0
        # 40 hook calls x 10 ms would dominate if they were timed
        assert st.seconds < st.callback_s, (st.seconds, st.callback_s)


if __name__ == "__main__":
    disable_power_throttling()
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if DEVICE.type == "cuda" else ""))
    t_all = time.perf_counter()
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t0 = time.perf_counter()
        print(f"{t.__name__} ...")
        t()
        print(f"  ok ({time.perf_counter() - t0:.1f}s)")
    print(f"ALL TESTS PASSED ({len(tests)} tests, {time.perf_counter() - t_all:.0f}s)")
