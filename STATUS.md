# Project status

Speculative decoding with Qwen2.5-0.5B-Instruct (draft) and Qwen2.5-3B-Instruct (target), batch size 1. See [SPEC.md](SPEC.md) for the plan.

_Last updated: 2026-10-01_

| Phase | What | Status |
| --- | --- | --- |
| P1 | Environment, real-model baseline, measured cost ratio `c` | ✅ Done |
| P2 | Speculative decoding verified on the real models | ✅ Done |
| P3 | Fixed-γ sweep | ⏸ Paused at 22 / 36 cells |
| P4 | AdaEDL baseline | Not started |
| P5 | Adaptive-γ comparison (the O2 claim) | Not started |
| P6 | Plots and report | Not started |

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
| `summarize.py` | Aggregates `results/results.csv` per (task, temperature, method) |
| `winperf.py` | Opts the process out of Windows power throttling (see below) |
| `prompts.py` | 6 prompts each for code, maths and chat |

### Windows power throttling

The first timings were 4–10× too slow. Windows 11 classed the Python process as background and moved it onto efficiency cores, so each GPU kernel launch took 50–90 µs instead of ~16 µs. Batch-1 decoding is bound by these launches. `winperf.disable_power_throttling()` switches this off for the current process only and is called at the start of every script. All numbers below were measured with it on.

## Tests

`ALL TESTS PASSED`: 7 tests, about 170 s on the GPU.

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

## Phase 3 (partial)

Files: `results/results.csv`, `results/phase3.log`. The full grid is γ ∈ {1, 2, 3, 4, 6, 8} × 3 tasks × 2 temperatures × 6 prompts, with a warm-up and 3 repeats per cell. Saved so far: all 12 code cells and maths prompts 0–4 (22 / 36 cells).

**Speed-up versus baseline** (mean ± std across prompts, predicted value in brackets):

| Task | T | γ = 1 | γ = 2 | γ = 4 | γ = 8 | α |
| --- | --- | --- | --- | --- | --- | --- |
| Code | 0 | **1.20 ± 0.06** (1.09) | 1.17 ± 0.12 (1.11) | 1.10 ± 0.14 (1.05) | 1.01 ± 0.18 (0.93) | 0.89–0.91 |
| Code | 0.7 | **1.21 ± 0.09** (1.10) | 1.19 ± 0.10 (1.09) | 1.09 ± 0.13 (1.00) | 0.91 ± 0.19 (0.82) | 0.88–0.90 |
| Maths* | 0 | 1.19 ± 0.06 (1.11) | **1.23 ± 0.08** (1.15) | 1.13 ± 0.11 (1.12) | 1.15 ± 0.30 (1.04) | 0.92–0.94 |
| Maths* | 0.7 | 1.22 ± 0.05 (1.12) | **1.25 ± 0.09** (1.16) | 1.17 ± 0.13 (1.15) | 0.99 ± 0.09 (0.99) | 0.93–0.95 |

\* Maths covers prompts 0–4 only. Chat has not been run yet.

Every greedy cell matches the baseline so far. Findings so far:

- **Best fixed γ:** 1 for code and 2 for maths, in line with the Phase 1 prediction (γ = 1 up to α = 0.8, γ = 2 at α = 0.9).
- **Measured speed-ups beat the formula** by ~0.05–0.1 at small γ. The likely reason is the fixed per-token overhead of the baseline (sampling over 152k vocabulary entries, one GPU sync per token), which speculation spreads over several tokens and `c` does not capture.

## Open issues

1. **Throughput drifted during the P3 run.** Baseline speed rose from ~17 to ~22–26 tokens/s from maths prompt 1 onward, presumably because the machine's background load changed. Speed-up ratios remain valid, since each cell runs its methods back to back. But `c` (0.734) was measured once at the start, so predictions for later cells may be off. Re-measure `c` per cell, or at least per task, before the final report.
2. **The fp32 recheck is impossible on 8 GB.** Phase 2 uses the near-tie analysis described above instead.
3. **High `c` limits the headroom.** Compiling the draft (P5) is the main lever.

## Next steps

1. Finish P3: run `.venv\Scripts\python bench.py >> results\phase3.log` (it skips saved cells, ~50 min left), then `python summarize.py`.
2. P4: implement AdaEDL (arXiv:2410.18351) from the paper as a drop-in controller.
3. P5: adaptive-γ comparison, including the compiled / static-cache draft.
4. P6: plots and report, every number traced to `results.csv`.

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python test_specdec.py     # ALL TESTS PASSED
.venv\Scripts\python bench.py --tiny     # smoke test; speed-up < 1x is expected (c ≈ 1)
.venv\Scripts\python phase1.py
.venv\Scripts\python phase2.py
.venv\Scripts\python bench.py            # Phase 3 sweep
.venv\Scripts\python summarize.py
```
