"""Speculative decoding engine (Leviathan et al. 2023; Chen et al. 2023).

Batch size 1. A single accept/reject path serves greedy and sampling: temperature 0
turns every distribution into a one-hot at the argmax (SPEC §5.4).

Cache invariant (SPEC §5.1): each model's KV cache always holds a prefix of the
committed sequence. Before a model runs it is fed `seq[cache.get_seq_length():]`,
so there is no separate prefill path.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


# --------------------------------------------------------------------------- distributions

def to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Logits [..., V] -> float32 probabilities. Temperature 0 gives a one-hot at the argmax."""
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits / temperature, dim=-1)


def sample(probs: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probs, 1, generator=generator))


def make_generator(device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


# --------------------------------------------------------------------------- verification

def verify(draft_tokens: list[int], q: torch.Tensor, p: torch.Tensor,
           generator: torch.Generator) -> tuple[int, int]:
    """Modified rejection sampling (SPEC §3 steps 3-4).

    draft_tokens: γ guesses; q: [γ, V] draft distributions they were sampled from;
    p: [γ+1, V] target distributions (row γ is the bonus distribution).

    Returns (n_accepted, extra_token). Tokens *tested* = n_accepted + (n_accepted < γ).
    """
    gamma = len(draft_tokens)
    if gamma:
        idx = torch.tensor(draft_tokens, device=p.device)
        rows = torch.arange(gamma, device=p.device)
        r = torch.rand(gamma, generator=generator, device=p.device)
        rejected = (r * q[rows, idx] >= p[rows, idx]).tolist()
        for j in range(gamma):
            if rejected[j]:
                residual = torch.clamp(p[j] - q[j], min=0)
                if float(residual.sum()) <= 0:  # only reachable through rounding when p ≈ q
                    residual = p[j]
                return j, sample(residual, generator)
    return gamma, sample(p[gamma], generator)


# --------------------------------------------------------------------------- caches

def rollback(cache: DynamicCache, length: int) -> None:
    """Crop `cache` to `length` tokens; no-op if already that short.

    Uses negative crop (SPEC §5.3). Note crop(-0) would be crop(0) and wipe the cache,
    hence the guard.
    """
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


@torch.inference_mode()
def _forward(model, cache: DynamicCache, seq: list[int], n_keep: int) -> torch.Tensor:
    """Feed the part of `seq` the cache has not seen; return the last `n_keep` logit rows."""
    start = cache.get_seq_length()
    ids = torch.tensor([seq[start:]], device=model.device)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=n_keep)
    return out.logits[0]


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- stats

@dataclass
class Stats:
    new_tokens: int = 0
    seconds: float = 0.0
    ttft_s: float = float("nan")
    target_passes: int = 0
    draft_passes: int = 0
    rounds: int = 0
    tested: int = 0
    accepted: int = 0
    pos_tested: list[int] = field(default_factory=list)
    pos_accepted: list[int] = field(default_factory=list)
    gammas: list[int] = field(default_factory=list)
    callback_s: float = 0.0  # time spent in on_round, excluded from `seconds`

    def record_round(self, gamma: int, n_accepted: int) -> None:
        n_tested = n_accepted + (n_accepted < gamma)
        self.rounds += 1
        self.target_passes += 1
        self.gammas.append(gamma)
        self.tested += n_tested
        self.accepted += n_accepted
        while len(self.pos_tested) < n_tested:
            self.pos_tested.append(0)
            self.pos_accepted.append(0)
        for i in range(n_tested):
            self.pos_tested[i] += 1
            self.pos_accepted[i] += i < n_accepted

    @property
    def alpha(self) -> float:
        """Kept ÷ tested, never kept ÷ proposed (SPEC §5.5)."""
        return self.accepted / self.tested if self.tested else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.seconds if self.seconds else float("nan")

    @property
    def target_passes_per_token(self) -> float:
        return self.target_passes / self.new_tokens if self.new_tokens else float("nan")

    @property
    def mean_gamma(self) -> float:
        return sum(self.gammas) / len(self.gammas) if self.gammas else float("nan")


def _eos_set(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return set(eos_token_id)


def _commit(seq: list[int], new: list[int], eos: set[int], limit: int) -> bool:
    """Append `new` to `seq`, stopping at EOS or once `limit` total tokens exist. Returns done."""
    for tok in new:
        seq.append(tok)
        if tok in eos or len(seq) >= limit:
            return True
    return False


def _notify(on_round, stats: Stats, committed: list[int], n_draft: int, rejected) -> None:
    """Call the display hook and keep its time out of the measurement."""
    if on_round is not None:
        t = time.perf_counter()
        on_round(committed, n_draft, rejected)
        stats.callback_s += time.perf_counter() - t


# --------------------------------------------------------------------------- decoders

@torch.inference_mode()
def baseline_generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.0,
                      eos_token_id=None, seed: int = 0, on_round=None) -> tuple[list[int], Stats]:
    """Plain autoregressive decoding: one target pass per token.

    on_round(committed, n_draft, rejected) is called after every commit (n_draft is always 0 here);
    its time is excluded from stats.seconds so a live display does not distort the timing.
    """
    device = model.device
    gen = make_generator(device, seed)
    eos, limit = _eos_set(eos_token_id), len(prompt_ids) + max_new_tokens
    seq, cache, stats = list(prompt_ids), DynamicCache(), Stats()

    _sync(device)
    t0 = time.perf_counter()
    done = max_new_tokens <= 0
    while not done:
        p = to_probs(_forward(model, cache, seq, 1)[-1], temperature)
        stats.target_passes += 1
        before = len(seq)
        done = _commit(seq, [sample(p, gen)], eos, limit)
        if stats.target_passes == 1:
            _sync(device)
            stats.ttft_s = time.perf_counter() - t0
        _notify(on_round, stats, seq[before:], 0, None)
    _sync(device)
    stats.seconds = time.perf_counter() - t0 - stats.callback_s
    new = seq[len(prompt_ids):]
    stats.new_tokens = len(new)
    return new, stats


@torch.inference_mode()
def speculative_generate(target, draft, prompt_ids: list[int], max_new_tokens: int, controller,
                         temperature: float = 0.0, eos_token_id=None,
                         seed: int = 0, on_round=None) -> tuple[list[int], Stats]:
    """Speculative decoding (SPEC §3). `controller` picks γ each round (FixedGamma, GammaController, ...).

    on_round(committed, n_draft, rejected) is called after every round: the tokens committed, how many
    of them (from the front) are accepted draft guesses, and the rejected guess (None if all were kept).
    Its time is excluded from stats.seconds.
    """
    device = target.device
    gen = make_generator(device, seed)
    eos, limit = _eos_set(eos_token_id), len(prompt_ids) + max_new_tokens
    seq, t_cache, d_cache, stats = list(prompt_ids), DynamicCache(), DynamicCache(), Stats()
    controller.reset()

    _sync(device)
    t0 = time.perf_counter()
    done = max_new_tokens <= 0
    while not done:
        remaining = limit - len(seq)
        gamma_max = min(controller.next_gamma(), remaining - 1)  # a round commits up to γ+1 tokens

        # 1. draft: γ autoregressive guesses, keeping the full distribution each time
        drafts, qs = [], []
        for i in range(gamma_max):
            logits = _forward(draft, d_cache, seq + drafts, 1)[-1]
            q = to_probs(logits, temperature)
            stats.draft_passes += 1
            if controller.stop_before(q, i, logits):
                break
            drafts.append(sample(q, gen))
            qs.append(q)
        gamma = len(drafts)

        # 2. target: one pass over (unseen committed tokens + drafts), γ+1 rows kept
        p = to_probs(_forward(target, t_cache, seq + drafts, gamma + 1), temperature)
        q = torch.stack(qs) if qs else p[:0]

        # 3-4. accept/reject, then one replacement or bonus token
        n_acc, extra = verify(drafts, q, p, gen)
        stats.record_round(gamma, n_acc)
        controller.update(n_acc, n_acc + (n_acc < gamma), gamma)

        # 5. commit, then roll both caches back to len(seq) - 1
        before = len(seq)
        done = _commit(seq, drafts[:n_acc] + [extra], eos, limit)
        rollback(t_cache, len(seq) - 1)
        rollback(d_cache, len(seq) - 1)
        if stats.rounds == 1:
            _sync(device)
            stats.ttft_s = time.perf_counter() - t0
        committed = seq[before:]
        _notify(on_round, stats, committed, min(n_acc, len(committed)), drafts[n_acc] if n_acc < gamma else None)
    _sync(device)
    stats.seconds = time.perf_counter() - t0 - stats.callback_s
    new = seq[len(prompt_ids):]
    stats.new_tokens = len(new)
    return new, stats


# --------------------------------------------------------------------------- speed formulas

def expected_tokens(alpha: float, gamma: int) -> float:
    """E = (1 - α^(γ+1)) / (1 - α): expected tokens committed per round."""
    if alpha >= 1:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def predicted_speedup(alpha: float, gamma: int, c: float) -> float:
    """S = E / (1 + γc)."""
    return expected_tokens(alpha, gamma) / (1 + gamma * c)


def best_gamma(alpha: float, c: float, gamma_max: int = 8, gamma_min: int = 1) -> int:
    return max(range(gamma_min, gamma_max + 1), key=lambda g: predicted_speedup(alpha, g, c))


# --------------------------------------------------------------------------- γ controllers

class FixedGamma:
    def __init__(self, gamma: int):
        self.gamma = gamma

    def reset(self) -> None:
        pass

    def next_gamma(self) -> int:
        return self.gamma

    def stop_before(self, q: torch.Tensor, i: int, logits: torch.Tensor = None) -> bool:
        """Called with the draft distribution q (and raw logits) for guess i, before it is sampled."""
        return False

    def update(self, accepted: int, tested: int, gamma: int) -> None:
        pass


class GammaController(FixedGamma):
    """Cost-ratio-aware adaptive γ: each round pick the γ maximising S(α̂, γ, c).

    α̂ is a decayed kept/tested ratio with `prior_weight` pseudo-observations at `alpha0`,
    so the first rounds are not decided by one or two coin flips.
    """

    def __init__(self, c: float, gamma_max: int = 8, gamma_min: int = 1, alpha0: float = 0.7,
                 prior_weight: float = 4.0, decay: float = 0.95):
        self.c, self.gamma_max, self.gamma_min = c, gamma_max, gamma_min
        self.alpha0, self.prior_weight, self.decay = alpha0, prior_weight, decay
        self.reset()

    def reset(self) -> None:
        self.kept = self.alpha0 * self.prior_weight
        self.seen = self.prior_weight

    @property
    def alpha_hat(self) -> float:
        return self.kept / self.seen

    def next_gamma(self) -> int:
        return best_gamma(self.alpha_hat, self.c, self.gamma_max, self.gamma_min)

    def update(self, accepted: int, tested: int, gamma: int) -> None:
        self.kept = self.decay * self.kept + accepted
        self.seen = self.decay * self.seen + tested


class AdaEDL(FixedGamma):
    """AdaEDL: entropy-based early draft stopping (Agrawal, Jeon, Lee, NeurIPS ENLSP 2024, arXiv:2410.18351).

    Stopping rule (paper §3, Fig. 1, App. A): before drafting the next token, with H = entropy of the
    draft distribution p_DM, stop drafting if
        1 - sqrt(entropy_factor * H) < λ
    (1 - sqrt(γ_ent·H) is an approximate lower bound on that token's acceptance probability).

    Threshold update (paper Algorithm 1), after every round with n_drafted guesses and n_acc accepted:
        AR  <- β1·AR + (1-β1)·n_acc/n_drafted
        λ'  =  λ + ε  if AR < α;   λ - ε  elif n_acc != L;   λ  otherwise
        λ   <- β2·λ + (1-β2)·λ'
    Defaults are the paper's (§3.1): γ_ent = 0.2, β1 = 0.5, β2 = 0.9, ε = 0.01, α = 0.9. The paper does not
    fix the initial λ (it sweeps 0.3-0.9), so lam0 is a parameter.

    Choices the paper leaves open, made here:
      * Greedy (T = 0): q is one-hot (entropy 0), which would never stop, so H is taken from the draft's
        unscaled softmax. For T > 0, H is the entropy of the distribution the draft samples from.
      * min_draft = 1: the first guess of a round is always drafted. Checking it too can give a round with
        n_drafted = 0, where Algorithm 1's n_acc/n_drafted is 0/0 and λ can stay too high to ever recover.
      * Entropy in nats; the moving average AR starts at the target α.
      * AR follows the paper (kept ÷ drafted); the reported Stats.alpha stays kept ÷ tested (SPEC §5.5).
    """

    def __init__(self, max_draft: int = 8, lam0: float = 0.7, temperature: float = 1.0,
                 entropy_factor: float = 0.2, target_ar: float = 0.9, eps: float = 0.01,
                 beta1: float = 0.5, beta2: float = 0.9, min_draft: int = 1, log_all: bool = False):
        self.max_draft, self.lam0, self.temperature = max_draft, lam0, temperature
        self.log_all = log_all  # also log entropy at positions < min_draft (analysis only; costs time)
        self.entropy_factor, self.target_ar, self.eps = entropy_factor, target_ar, eps
        self.beta1, self.beta2, self.min_draft = beta1, beta2, min_draft
        self.reset()

    def reset(self) -> None:
        self.lam, self.ar = self.lam0, self.target_ar
        self.steps = []   # (round, position, entropy, stopped) for every checked draft position
        self.rounds = []  # (n_drafted, n_acc, λ after the update) per round

    def next_gamma(self) -> int:
        return self.max_draft

    def entropy(self, logits: torch.Tensor) -> float:
        t = self.temperature if self.temperature > 0 else 1.0
        p = torch.softmax(logits.float() / t, dim=-1)
        return float(-torch.special.xlogy(p, p).sum())  # xlogy: 0·log 0 = 0, so masked tokens are safe

    def lower_bound(self, h: float) -> float:
        return 1.0 - math.sqrt(self.entropy_factor * h)

    def stop_before(self, q: torch.Tensor, i: int, logits: torch.Tensor = None) -> bool:
        if i < self.min_draft and not self.log_all:
            return False
        h = self.entropy(logits if logits is not None else torch.log(q.clamp_min(1e-30)))
        stop = i >= self.min_draft and self.lower_bound(h) < self.lam
        self.steps.append((len(self.rounds), i, h, stop))
        return stop

    def update(self, accepted: int, tested: int, gamma: int) -> None:
        if gamma > 0:
            self.ar = self.beta1 * self.ar + (1 - self.beta1) * accepted / gamma
        if self.ar < self.target_ar:
            lam_p = self.lam + self.eps
        elif accepted != self.max_draft:
            lam_p = self.lam - self.eps
        else:
            lam_p = self.lam
        self.lam = self.beta2 * self.lam + (1 - self.beta2) * lam_p
        self.rounds.append((gamma, accepted, self.lam))


# --------------------------------------------------------------------------- pair checks + cost ratio

def assert_compatible(target_tok, draft_tok, target_model, draft_model) -> None:
    """Refuse to run on a pair that does not share a tokenizer (SPEC §5.7)."""
    if target_tok.get_vocab() != draft_tok.get_vocab():
        raise ValueError("target and draft tokenizers have different vocabularies")
    rows_t = target_model.get_output_embeddings().weight.shape[0]
    rows_d = draft_model.get_output_embeddings().weight.shape[0]
    if rows_t != rows_d:
        raise ValueError(f"output layers differ: target {rows_t} rows, draft {rows_d} rows")


@torch.inference_mode()
def time_decode_step(model, device, context_len: int = 256, steps: int = 20, warmup: int = 3,
                     seed: int = 0) -> float:
    """Mean seconds for one single-token decode step on top of a `context_len` KV cache."""
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(0, model.config.vocab_size, (1, context_len), generator=g).to(device)
    tok = ctx[:, -1:]
    cache = DynamicCache()
    model(input_ids=ctx, past_key_values=cache, use_cache=True, logits_to_keep=1)
    for _ in range(warmup):
        model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        model(input_ids=tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
    _sync(device)
    return (time.perf_counter() - t0) / steps


def measure_cost_ratio(target, draft, device, context_len: int = 256, steps: int = 20,
                       warmup: int = 3) -> dict:
    """c = draft step time / target step time."""
    t_target = time_decode_step(target, device, context_len, steps, warmup)
    t_draft = time_decode_step(draft, device, context_len, steps, warmup)
    return {"c": t_draft / t_target, "target_step_ms": 1e3 * t_target, "draft_step_ms": 1e3 * t_draft}


def mean_std(xs) -> tuple[float, float]:
    xs = list(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd
