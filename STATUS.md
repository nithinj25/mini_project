# Project status

Speculative decoding with Qwen2.5-0.5B-Instruct (draft) and Qwen2.5-3B-Instruct (target), batch size 1. See [SPEC.md](SPEC.md) for the plan.

_Last updated: 2026-10-10 (P6 complete)_

| Phase | What | Status |
| --- | --- | --- |
| P1 | Environment, real-model baseline, measured cost ratio `c` | ✅ Done |
| P2 | Speculative decoding verified on the real models | ✅ Done |
| P3 | Fixed-γ sweep | ✅ Done |
| P4 | AdaEDL baseline | ✅ Done |
| P5 | Adaptive-γ comparison (the O2 claim), eager draft | ✅ Done: **O2 not supported** (ties the best fixed γ, beats AdaEDL) |
| P5b | Same comparison with the CUDA-graph draft (SPEC §7's static-cache route) | ✅ Done: **O2 not supported** (ties the best fixed γ, beats AdaEDL); speed-ups 2.1–3.5× |
| P6 | Plots and report | ✅ Done: [REPORT.md](REPORT.md), `results/figures/`, `results/results.csv` |

## Setup

- **Hardware:** RTX 4060 Laptop GPU, 8 GB, compute capability 8.9. The spec assumes a T4 with 16 GB, so results describe this GPU, not a T4.
- **Software:** Python 3.12, torch 2.11.0+cu128, transformers 5.17.0. Everything runs in fp16 on the GPU, and both models together use 6.7 GB.
- **Starter code:** the bundle described in SPEC §4 was not available, so the engine, tests and harness were written from the spec:

| File | Purpose |
| --- | --- |
| `specdec.py` | Engine: baseline and speculative decoders, `verify` (modified rejection sampling), KV rollback with negative crop, `Stats`, `FixedGamma` / `GammaController`, `measure_cost_ratio`, speed formulas |
| `test_specdec.py` | 7 tests on tiny random models, running on the GPU by default (`--cpu` to force CPU) |
| `bench.py` | Benchmark harness: CSV output, `--tiny` smoke mode, resumes by skipping finished cells |
| `phase1.py`, `phase2.py` | Phase 1 and Phase 2 scripts |
| `summarize.py` | Aggregates `results/phase3.csv` per (task, temperature, method) |
| `winperf.py` | Opts the process out of Windows power throttling (see below) |
| `phase4.py`, `phase5.py`, `phase5b.py` | AdaEDL analysis; the O2 comparison with the eager draft and with the CUDA-graph draft (pre-registered) |
| `make_results.py`, `plots.py` | Build `results/results.csv` (all phases) and `results/figures/` |
| `REPORT.md` | The final report (Phase 6) |
| `demo/`, `run_demo.bat` | Live presentation demo |
| `prompts.py` | 6 prompts each for code, maths and chat |

### Windows power throttling

The first timings were 4–10× too slow. Windows 11 classed the Python process as background and moved it onto efficiency cores, so each GPU kernel launch took 50–90 µs instead of ~16 µs. Batch-1 decoding is bound by these launches. `winperf.disable_power_throttling()` switches this off for the current process only and is called at the start of every script. All numbers below were measured with it on.

## Tests

`ALL TESTS PASSED`: 13 tests on the GPU; on the CPU the 11 non-graph tests pass and the 2 CUDA-graph tests skip (`results/tests_cpu.log`), about 170 s on the GPU.

- **Lossless `verify`:** TV(output, p) = 0.0027 over 40k samples.
- **Mutation check (SPEC §8):** replacing the residual with p raises that TV to 0.171, and the two-token joint test's TV from 0.023 to 0.119. Both tests catch the bug.
- **Greedy equality:** speculative output equals the baseline with a bad draft (α ≈ 0.05) and a good one (α ≈ 0.63).
- **Draft = target:** α = 1.0.
- **Two-token joint distribution:** matches exact enumeration at γ = 1 and γ = 2.
- **Other:** EOS and length limits, formulas, controller, rollback, and refusal of mismatched tokenizers.

## Phase 1 results

Files: `results/phase1.json`, `results/phase1_baseline.csv`, `results/phase1_predictions.csv`, `results/phase1.log`.

- **Sanity text** (fp16, greedy) is coherent: *"A hash table is a data structure that uses a hash function to map keys to values…"*
- **Pair compatibility** is asserted in code: identical vocabularies and 151,936 output rows. Architecture matches the spec: 3B = 36 layers / 2 KV heads / head dim 128, 0.5B = 24 / 2 / 64.

| Measurement | Result |
| --- | --- |
| Cost ratio `c` (3 runs, context 256) | **0.693 ± 0.029** (draft ≈ 38 ms, target ≈ 54 ms per step) |
| Baseline throughput (3 tasks × 3 prompts, 128 tokens, greedy) | **17.51 ± 0.37 tokens/s** |
| Time to first token | 62–69 ms |
| Peak GPU memory | 6.71 GB |

**Predicted speed-ups, recorded before any speculative run:** S = E / (1 + γc) with c = 0.693.

| α | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
| --- | --- | --- | --- | --- | --- |
| Best γ | 1 | 1 | 1 | 1 | 2 |
| Best S | 0.89 | 0.95 | 1.00 | 1.06 | 1.14 |

`c` is above 0.5, so the achievable speed-up is small. Both models are bound by kernel-launch overhead, which is why the 6× smaller draft is only ~1.5× faster per step. As SPEC §7 says, P5 should include the `torch.compile` / static-cache route for the draft.

## Phase 2 results

Files: `results/phase2.csv`, `results/phase2.json`, `results/phase2.log`. The run covered 18 prompts at γ ∈ {1, 4, 8}, greedy, 128 tokens.

- **49 / 54 runs match the baseline exactly.**
- **All 5 mismatches are fp16 near-ties.** They come from 2 chat prompts. At the first differing token, the target's top two logits differ by only 0.016–0.031, one or two fp16 steps, and re-scoring ranks the two tokens 1st and 2nd. For example, *"…manageable blocks. **Break** your study"* versus *"…manageable blocks **with** short breaks"*. One of the two prompts matched exactly at γ = 8, which is consistent with numerics and not a logic error.
- **Deviation from SPEC §9:** the spec says to rerun mismatches in float32, but the 3B target needs ~12 GB in fp32 and this GPU has 8 GB. The near-tie logit-gap analysis replaces that check.

| Task | Mean α |
| --- | --- |
| Code | 0.90 |
| Maths | 0.94 |
| Chat | 0.72 |

## Phase 3 results

Files: `results/phase3.csv` (36 cells, 252 rows; the P3 rows of `results/results.csv`), `results/phase3_summary.csv`, `results/phase3.log`. The grid is γ ∈ {1, 2, 3, 4, 6, 8} × 3 tasks × 2 temperatures × 6 prompts, with a warm-up and 3 repeats per (cell, method), 128 new tokens. The run took two sessions (2026-09-30 and 2026-10-06), and bench.py resumed by skipping saved cells.

**Speed-up versus baseline** (mean ± std across 6 prompts; predicted S(α, γ, c) from the measured α in brackets):

| Task | T | γ = 1 | γ = 2 | γ = 3 | γ = 4 | γ = 6 | γ = 8 | α |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Code | 0 | **1.20 ± 0.06** (1.09) | 1.17 ± 0.12 (1.11) | 1.14 (1.08) | 1.10 (1.05) | 1.06 (1.00) | 1.01 (0.93) | 0.89–0.91 |
| Code | 0.7 | **1.21 ± 0.09** (1.10) | 1.19 ± 0.10 (1.09) | 1.16 (1.04) | 1.09 (1.00) | 1.02 (0.95) | 0.91 (0.82) | 0.88–0.90 |
| Maths | 0 | 1.20 ± 0.05 (1.11) | **1.26 ± 0.10** (1.16) | 1.17 (1.15) | 1.13 (1.13) | 1.20 (1.10) | 1.15 (1.06) | 0.93–0.95 |
| Maths | 0.7 | 1.27 ± 0.13 (1.12) | **1.28 ± 0.10** (1.17) | 1.21 (1.14) | 1.18 (1.15) | 1.17 (1.10) | 1.07 (1.02) | 0.94–0.96 |
| Chat | 0 | **1.15 ± 0.13** (0.98) | 1.04 ± 0.14 (0.91) | 0.92 (0.82) | 0.83 (0.72) | 0.67 (0.57) | 0.58 (0.50) | 0.71–0.74 |
| Chat | 0.7 | **1.02 ± 0.10** (0.98) | 0.91 ± 0.15 (0.89) | 0.79 (0.81) | 0.68 (0.71) | 0.61 (0.59) | 0.46 (0.46) | 0.71–0.72 |

**Best fixed γ:** 1 for code and chat, 2 for maths. This matches the Phase 1 prediction made before any speculative run (γ = 1 up to α = 0.8, γ = 2 at α = 0.9).

**Findings:**

- **Speed-up exists, but it is modest:** about 1.2× on code, 1.26–1.28× on maths, and 1.0–1.15× on chat. This is what c ≈ 0.7 allows.
- **Acceptance rate depends on the task, not the temperature:** α ≈ 0.90 on code, 0.95 on maths and 0.72 on chat, nearly identical at T = 0 and T = 0.7.
- **The formula ranks γ correctly but underestimates speed-up** by about 0.05–0.17 at small γ, most on code and maths at T = 0. The likely cause is the baseline's fixed per-token overhead (sampling over 152k vocabulary entries, one GPU sync per token), which speculation spreads over several tokens and which the isolated step timing behind `c` does not capture. Chat at T = 0.7 matches the formula closely.
- **Large γ hurts on chat:** γ = 8 runs at 0.46–0.58× the baseline speed, because at α ≈ 0.72 most of a long draft is thrown away.
- **Greedy equality:** 99 of 108 speculative greedy runs match the baseline. All 9 mismatches are on chat prompts 1 and 5, the two prompts Phase 2 diagnosed as fp16 near-ties (top-2 logit gap 0.016 and 0.031). Which tied token wins changes with γ, as before.

## Phase 4 results: AdaEDL

**Implementation.** The `AdaEDL` controller in `specdec.py` is a drop-in alternative to `GammaController`, selected in bench.py with `--methods adaedl[:λ0]` and `--max-draft L`. It was implemented from the paper's LaTeX source (arXiv:2410.18351):

- **Stopping rule** (§3, Fig. 1, App. A): before drafting the next token, stop if 1 − √(0.2 · H) < λ, where H is the draft distribution's entropy. 1 − √(0.2 · H) approximates a lower bound on that token's acceptance probability.
- **Threshold update** (Algorithm 1), after each round: AR ← 0.5 · AR + 0.5 · n_acc / n_drafted. Then λ' = λ + 0.01 if AR < 0.9, else λ − 0.01 if n_acc ≠ L, else λ. Then λ ← 0.9 · λ + 0.1 · λ'. All values are the paper's.

**Choices the paper leaves open:**

- **Greedy mode:** the paper only evaluates sampling. At T = 0 our draft distribution is one-hot (entropy 0), so H is taken from the draft's unscaled softmax.
- **First guess of a round:** always drafted (min_draft = 1). Checking it too allows rounds with n_drafted = 0, where Algorithm 1's n_acc / n_drafted is 0/0 and λ can lock above the level where drafting resumes.
- **Other details:** entropy is in nats, and AR starts at the target 0.9.
- **Initial λ:** the paper sweeps 0.3–0.9 without fixing one value, so λ0 is a parameter; we use 0.7 (see the sweep below).

**Tests:** 3 new tests, all mutation-checked:

- **Stopping rule:** exact on both sides of the threshold, using uniform distributions with known entropy.
- **λ update:** Algorithm 1 worked by hand over 4 rounds.
- **Losslessness:** with AdaEDL, greedy output equals the baseline and the sampled two-token joint distribution matches exact enumeration (TV 0.026), with draft lengths that really vary.

Planted bugs that the tests catch: a flipped inequality, an ignored min_draft, swapped ±ε, and AR computed from tested instead of drafted.

**Run.** `phase4.py` ran 18 prompts at T = 0 and 0.7 with λ0 = 0.7 and L = 8, plus a λ0 sweep. Files: `results/phase4_runs.csv`, `results/phase4_steps.csv` (every entropy check), `results/phase4.json`, `results/phase4.log`.

**Done criterion met: AdaEDL drafts less when the draft model's entropy is high.**

| Task | T | Mean entropy H (nats) | Mean draft length | α |
| --- | --- | --- | --- | --- |
| Maths | 0 | 0.19 | 4.75 | 0.98 |
| Maths | 0.7 | 0.12 | 5.26 | 0.98 |
| Code | 0 | 0.41 | 3.46 | 0.95 |
| Code | 0.7 | 0.26 | 4.01 | 0.92 |
| Chat | 0 | 1.31 | 1.47 | 0.77 |
| Chat | 0.7 | 0.77 | 1.72 | 0.75 |

- **Correlation with draft length:** prompt mean entropy vs mean draft length r = −0.89 (36 runs); per round r = −0.58.
- **Stop rate of the check by entropy:** 0% below 0.25 nats, 19% at 0.25–0.5, and 100% above 0.5 (λ ≈ 0.7 corresponds to H ≈ 0.45).

**The paper's premise holds on this pair.** Acceptance of a drafted token falls with its entropy, and the bound 1 − √(0.2 · H) stays below the measured acceptance in every bin:

| Entropy bin (nats) | Tested tokens | Measured acceptance | Mean bound |
| --- | --- | --- | --- |
| 0–0.25 | 2560 | 0.99 | 0.95 |
| 0.25–0.5 | 311 | 0.90 | 0.74 |
| 0.5–1 | 238 | 0.68 | 0.62 |
| 1–2 | 272 | 0.56 | 0.46 |
| 2–4 | 152 | 0.36 | 0.27 |

**Greedy equality:** 16 / 18 match. The 2 mismatches are the known fp16 near-ties: chat prompt 1 at token 33 (*"blocks. Break"* vs *"blocks with short"*) and chat prompt 5 at token 19 (*"some key"* vs *"several factors"*). These are the same positions and words as in Phases 2 and 3.

**Initial-λ sweep** (T = 0.7, 2 prompts per task, single runs):

| λ0 | 0.3 | 0.5 | 0.7 | 0.9 |
| --- | --- | --- | --- | --- |
| Mean draft length | 7.57 | 5.79 | 4.04 | 2.28 |
| α | 0.85 | 0.88 | 0.90 | 0.89 |
| Speed-up (indicative) | 0.83× | 0.87× | **1.02×** | 0.94× |

λ0 = 0.7 is used for Phase 5.

**Early speed signal** (single runs, with entropy logged at every position, so a little pessimistic): AdaEDL reached ~1.06× on code, ~1.10× on maths and ~0.75–0.80× on chat. That is below the best fixed γ from Phase 3 (1.20×, 1.26–1.28×, 1.02–1.15×). The likely reason is that AdaEDL targets an acceptance rate but ignores what a draft step costs. With c ≈ 0.7 on this GPU, its long drafts on easy text (5–6 tokens on maths) are expensive even at α ≈ 0.98. The timed comparison is Phase 5.

## Phase 5 results: the O2 claim

**The claim:** cost-ratio-aware adaptive γ gives higher tokens/s than (a) the best fixed γ and (b) AdaEDL.

**Verdict: O2 is not supported on this pair and GPU.** Adaptive γ shows **no detectable difference from the best fixed γ**, and it **clearly beats AdaEDL**. The decision rule was committed to git before any Phase 5 measurement (commit `26dc6f9`, see the `phase5.py` docstring), and the adaptive controller was used exactly as written in Phase 1, with no tuning.

**Protocol:**

- **Cells:** 36 (3 tasks × 6 prompts × T ∈ {0, 0.7}), 128 new tokens.
- **Methods:** baseline, fixed γ = 1 and 2, adaptive, and AdaEDL (λ0 = 0.7).
- **c:** measured in every cell (mean 0.70, sd 0.08).
- **Timing:** every method warmed up, then 3 repeats with the method order rotated between repeats. Speed-ups are paired by repeat.
- **Files:** `results/phase5.csv`, `results/phase5_summary.json`, `results/phase5.log`.

**Paired comparison**, d = speed-up(adaptive) − speed-up(opponent) per cell, with a 95% t-interval:

| Adaptive vs | Cells | Mean d | 95% CI | Adaptive wins | Verdict |
| --- | --- | --- | --- | --- | --- |
| **(a) Best fixed γ, chosen a priori** (1 code/chat, 2 maths) | 36 | −0.011 | [−0.039, +0.018] | 17 / 36 | no detectable difference |
| **(b) AdaEDL** (λ0 = 0.7) | 36 | **+0.178** | **[+0.134, +0.223]** | 33 / 36 | **beats** |
| Per-cell oracle best of fixed 1/2 (chosen after the fact) | 36 | −0.027 | [−0.054, +0.001] | 14 / 36 | no detectable difference |

By task, adaptive vs AdaEDL: code +0.15, maths +0.08, chat +0.30, each with a CI above 0. Adaptive vs the a-priori best fixed γ has CIs containing 0 for every task.

**Speed-up by method** (mean across prompts):

| Task | T | Fixed γ=1 | Fixed γ=2 | Adaptive (mean γ) | AdaEDL (mean draft) |
| --- | --- | --- | --- | --- | --- |
| Code | 0 | 1.22 | 1.22 | 1.24 (2.2) | 1.08 (3.5) |
| Code | 0.7 | 1.20 | 1.16 | 1.18 (1.6) | 1.03 (4.0) |
| Maths | 0 | 1.26 | 1.25 | 1.25 (2.4) | 1.11 (4.8) |
| Maths | 0.7 | 1.22 | 1.25 | 1.20 (2.3) | 1.18 (5.4) |
| Chat | 0 | 1.09 | 0.99 | 1.09 (1.0) | 0.76 (1.5) |
| Chat | 0.7 | 1.09 | 0.97 | 1.07 (1.0) | 0.80 (1.7) |

Greedy output matches the baseline in 34–35 of 36 cells per method. The misses are the known fp16 near-ties on chat prompts 1 and 5.

**Why adaptive γ does not beat the best fixed γ:**

1. **The speed-up curve is flat near its peak.** With c ≈ 0.7 and α ≈ 0.9–0.95, the formula gives γ = 1, 2 and 3 within ~3–4% of each other (Phase 1 table, Phase 3 sweep). Adaptive γ settles in that flat region: about 1.0 on chat, where it correctly drops to the minimum, and 1.6–2.4 on code and maths. So it lands on essentially the same speed as the best fixed γ. The gains adaptivity can find are smaller than run-to-run timing noise on a laptop (per-cell paired sd ≈ 0.05–0.1).
2. **The fixed opponent had an advantage.** Its γ was picked per task from the Phase 3 sweep, after seeing the data. Adaptive γ has to discover the right value online in each 128-token generation, starting from a prior of α = 0.7.
3. **Adaptivity does help where prompts differ a lot.** On easy prompts it raises γ: code prompt 1 went from 1.50× (γ = 1) to 1.60× (γ ≈ 3.5). It also drops to γ = 1 on chat. On average these per-prompt gains are cancelled by noise and the cost of learning α̂ during each generation.

**Why AdaEDL loses** (consistent with Phase 4):

1. **It ignores the cost of drafting.** It drafts long whenever the draft model is confident (4.8–5.4 tokens on maths at α ≈ 0.98), which is too long when a draft step costs ~70% of a target step.
2. **Stopping wastes a draft pass.** To decide whether to stop, it runs the draft model for the next position and then discards that pass. From the Phase 4 logs, 97% of chat rounds end in an early stop, so **38% of all draft passes on chat are wasted** (19% on code, 10% on maths). This is why it falls below 1× on chat even though its drafts are short.
3. **The formula overpredicts it** (predicted 1.20–1.32× vs measured 1.03–1.18×), because S = E / (1 + γc) counts neither the discarded pass nor the entropy computation.

**When the claim could hold:** a cheaper draft would make the curve steeper and the best γ larger and more prompt-dependent, which is where online adaptation pays. A lower c could come from a compiled or static-cache draft, or from a larger target. See the open issues.

## Phase 5b results and Phase 6

Full write-up: **[REPORT.md](REPORT.md)**, sections 5.6–6.

- **Compiled draft (SPEC §7), done natively on Windows.** `torch.compile` needs Triton, which is unavailable here, so `GraphDraft` captures the draft's decode step as a raw CUDA graph over a `StaticCache`. In fp64 it is bit-identical to the eager draft, and speculative decoding with it gives identical outputs and acceptance. The draft step falls from 53 ms to 7 ms, and c from 0.68 to 0.09–0.12.
- **Phase 5b, pre-registered in `15f882b`.** Speed-ups are 2.1–3.5×.
  - Adaptive vs the best fixed γ per task (picked after the fact: chat 4, code 8, maths 8): −0.074, CI [−0.225, +0.076], no detectable difference.
  - Adaptive vs AdaEDL: +0.199, CI [+0.066, +0.331], beats it.
  - **O2 is not supported in either cost regime.**
- **Exploratory:** adaptive γ matches the best single fixed γ for the whole workload in both regimes (2.974× vs 2.979× in P5b), without a sweep.
- **Phase 6 outputs:**
  - `results/results.csv`: every row of P3, P5 and P5b, keyed by phase, task, prompt, temperature and method. Look one up with `python make_results.py --show P5b code 1 0.0 adaptive`.
  - `results/figures/F1–F5`, made by `plots.py`.
  - The demo's Results page now includes the O2 comparison.

## Open issues

1. **Laptop timing noise** grows with the speed-up (paired sd ≈ 0.3–0.6 per cell at 3×), so few-percent effects cannot be resolved with 36 cells.
2. **The fp32 recheck is impossible on 8 GB.** Every greedy mismatch is an fp16 tie with logit gap 0–0.031, analysed in the report.
3. **The cap L = 8 binds** for code and maths at c ≈ 0.09.

## Next steps (optional)

1. A 7B/14B target on a 24–40 GB GPU (c ≈ 0.1–0.2, no binding cap), where adaptive γ has the most room. This needs the 152,064 vs 151,936 output-row fix for larger Qwen models.
2. Longer generations and a larger L on this pair.

## Presentation demo

`run_demo.bat` (or `.venv\Scripts\python -m streamlit run demo\app.py`) opens a local app at http://localhost:8501:

1. **Base model:** the 3B model alone, streamed live with tokens/s.
2. **Speculative decoding:** the same prompt streamed live. Guesses from the 0.5B model that the 3B accepted are highlighted green, and rejected guesses can be shown struck through. The page compares speed and output with page 1.
3. **Results:** the Phase 3 charts (speed-up vs γ, acceptance by draft position, predicted vs measured) and the data table, read from `results/`.

Display time is excluded from all timings by the `on_round` hook in `specdec.py` (covered by `test_on_round_hook`).

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python test_specdec.py     # ALL TESTS PASSED
.venv\Scripts\python bench.py --tiny     # smoke test; speed-up < 1x is expected (c ≈ 1)
.venv\Scripts\python phase1.py
.venv\Scripts\python phase2.py
.venv\Scripts\python bench.py            # Phase 3 sweep
.venv\Scripts\python phase4.py           # Phase 4: AdaEDL
.venv\Scripts\python phase5.py           # Phase 5: O2 comparison (eager draft)
.venv\Scripts\python phase5b.py          # Phase 5b: O2 comparison (CUDA-graph draft)
.venv\Scripts\python make_results.py     # results/results.csv
.venv\Scripts\python plots.py            # results/figures/
.venv\Scripts\python summarize.py
```
