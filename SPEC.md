# Speculative Decoding — Project Spec

BAI506 mini project · Qwen2.5-0.5B draft + Qwen2.5-3B target · single GPU, batch size 1

> **For Claude Code:** read this file before touching anything. Work one phase at a time, run the tests after every change, and stop at the end of a phase to report numbers. Do not skip ahead to later phases. Do not "improve" the accept/reject rule or the cache logic without a failing test to justify it.

---

## 1. What this project is

Make a 3B language model generate text faster while producing **exactly** the same output distribution, by having a 0.5B model guess tokens that the 3B model verifies in one forward pass.

**We are not inventing the algorithm.** Speculative decoding and its correctness proof come from Leviathan et al. (2023) and Chen et al. (2023). Our work is the implementation, the instrumentation, and one comparison that can come out negative.

### Non-goals

- No training of any model (no distillation, no draft heads).
- No batch sizes above 1.
- No tree-shaped drafts (SpecInfer / Medusa style).
- No claim of a novel algorithm.

---

## 2. Objectives

**O1 — Build.** Implement a lossless speculative decoding engine in PyTorch: draft-token proposal, single-pass parallel verification, modified rejection sampling, KV-cache rollback, adaptive draft-length control.

**O2 — Evaluate.** Test whether adapting the draft length from the measured cost ratio `c` and the running acceptance rate improves throughput over:

- (a) the best **fixed** draft length γ, and
- (b) an **entropy-based early-stopping** baseline (AdaEDL, arXiv:2410.18351),

across code, mathematics and general-text workloads.

### The claim being tested

> On a Qwen2.5-0.5B/3B pair on a T4-class GPU, cost-ratio-aware adaptive γ gives higher tokens/s than the best fixed γ and than entropy-based early stopping.

**This claim can fail.** "Adaptive γ did not beat the best fixed γ on this pair, and here is why" is an acceptable, reportable outcome. Do not tune the experiment until it passes; report what happens.

---

## 3. The algorithm (the contract the code must satisfy)

Per round, with draft length γ:

1. Draft model generates x₁…x_γ autoregressively, storing the **full** distribution q_i at each step.
2. Target model runs **one** forward pass over (unseen committed tokens + x₁…x_γ), keeping γ+1 logit rows → p₁…p_{γ+1}.
3. For i = 1…γ in order: draw r ~ U[0,1); keep x_i if `r * q_i(x_i) < p_i(x_i)`. Stop at the first rejection.
4. At the first rejection j: sample the replacement from `max(0, p_j − q_j)` normalised. If all γ kept: sample a bonus token from p_{γ+1}.
5. Commit kept tokens + that one extra token. Roll both KV caches back to `len(committed) − 1`.

### Index alignment

The target's input ends with `…, last_committed, x1, x2, x3`. Output at position k predicts token k+1, so:

| Target output taken at | Predicts | Used as |
| --- | --- | --- |
| last_committed | x1 | p₁ vs q₁ |
| x1 | x2 | p₂ vs q₂ |
| x2 | x3 | p₃ vs q₃ |
| x3 | next token | p₄ = bonus distribution |

γ guesses → γ+1 target rows → `logits_to_keep = γ + 1`.

### Formulas

- Acceptance rate: α = Σ min(p, q) = 1 − TV(p, q).
- Expected tokens per round: `E = (1 − α^(γ+1)) / (1 − α)` — check: α=0.8, γ=4 → 3.3616.
- Cost ratio: `c = draft step time / target step time`.
- Predicted speed-up: `S = E / (1 + γc)` — check: 3.3616 / 2.2 = 1.53.
- Break-even at γ=1: speculation pays only if α > c.

### Losslessness (why it works)

P(output = x) = min(p, q) + max(0, p − q) = p(x), because Σ max(0, p − q) = 1 − Σ min(p, q) exactly cancels the rejection mass.

---

## 4. What already exists

Files from the starter bundle, all tested on CPU with tiny random models:

| File | Status |
| --- | --- |
| `specdec.py` | Engine: baseline decoder, speculative decoder, `verify`, `rollback`, `Stats`, `GammaController`, `measure_cost_ratio`, speed formulas |
| `test_specdec.py` | 6 tests, all passing, about 1 minute on CPU |
| `bench.py` | Benchmark harness, CSV output, `--tiny` smoke mode |
| `requirements.txt` | torch ≥ 2.3, transformers ≥ 4.56 (tested on torch 2.5.1 / transformers 5.17.0) |

**Not yet done:** anything on real Qwen models, AdaEDL baseline, plots, report.

---

## 5. Invariants — do not break these

1. **Cache invariant.** Each model's KV cache holds a prefix of the committed sequence. Before a model runs, feed it `seq[cache.get_seq_length():]` (plus the drafts, for the target). This removes the need for separate prefill code and handles the case where the draft never saw its own last guess.
2. **Rollback both caches every round**, to `len(seq) − 1`. A stale entry causes silently wrong output, never an exception.
3. **Use negative crop**: `cache.crop(-k)` removes k tokens. Positive arguments are deprecated in transformers 5.17 and scheduled for removal; negative works on 4.x and 5.x.
4. **Greedy is one-hot sampling.** Temperature 0 returns a one-hot distribution at the argmax, so one accept/reject path serves greedy and sampling.
5. **α = kept ÷ tested**, never kept ÷ proposed. Guesses after a rejection are never tested and must not be counted.
6. **q must be the exact distribution the draft sampled from**, at the same temperature. Same for p and the target.
7. **Both models must share a tokenizer.** Refuse to run otherwise.

---

## 6. Phase plan

| Phase | Deliverable | Done when |
| --- | --- | --- |
| **P1** | Environment + real-model baseline + measured `c` | `c` recorded, baseline tokens/s recorded, sample output is sensible text |
| P2 | Speculative decoding verified on real models | Greedy output matches baseline (or mismatches explained as FP16 numerics) |
| P3 | Fixed-γ sweep | tokens/s, α, passes/token for γ ∈ {1,2,3,4,6,8} × 3 tasks × 2 temperatures |
| P4 | AdaEDL baseline implemented | Reproduces the paper's qualitative behaviour: shorter drafts at high entropy |
| P5 | Adaptive-γ comparison (O2) | Claim in §2 tested; result stated either way |
| P6 | Plots + report | Predicted vs measured speed-up plot; all numbers traceable to `results.csv` |

---

## 7. Phase 1 — implement this now

**Goal:** get real numbers for the two quantities that decide everything downstream: baseline throughput, and `c`.

### Tasks

1. **Environment.**
   - GPU with ≥ 16 GB VRAM (Colab T4, Kaggle, or lab machine).
   - `pip install -U transformers nvidia-ml-py`
   - Run `python test_specdec.py` → must end with `ALL TESTS PASSED`.
   - Run `python bench.py --tiny` → must complete; it will report a speed-up **below 1×** because the toy draft is the same size as the toy target (c ≈ 1). Expected, not a bug.

2. **Load the real models and sanity-check text.**
   - `Qwen/Qwen2.5-3B-Instruct` (target) and `Qwen/Qwen2.5-0.5B-Instruct` (draft), FP16 on a T4 (BF16 only on compute capability ≥ 8.0).
   - Generate 64 tokens from one prompt with the **baseline** decoder and print the text. It must be coherent. If it is repeated punctuation or `!!!!`, stop and retry in float32 before going further.

3. **Confirm the pair is compatible.**
   - Assert the two tokenizers have identical vocabularies.
   - Assert both output layers have 151,936 rows.
   - Record layers / KV heads / head dim for the report: 3B = 36 / 2 / 128, 0.5B = 24 / 2 / 64.

4. **Measure `c`.**
   - `measure_cost_ratio(target, draft, device)` with context length 256, 20 timed steps after 3 warm-ups, `torch.cuda.synchronize()` around timers.
   - Repeat 3 times and record mean and spread. `c` varies with GPU load, so a single number is not enough.

5. **Record the baseline.**
   - Baseline tokens/s and time-to-first-token for 3 prompts × 3 tasks, 128 new tokens, greedy.
   - Warm-up run first, then 3 repeats, report mean ± standard deviation.

6. **Compute the predictions before running any speculative decoding.**
   - Using the measured `c`, print `S(α, γ, c)` for α ∈ {0.5, 0.6, 0.7, 0.8, 0.9} and γ ∈ {1…8}.
   - Record the γ the formula says is best at each α. This is the prediction P3 and P5 will be tested against — write it down **before** measuring, so it is a real prediction and not hindsight.

### Phase 1 acceptance criteria

- [ ] `ALL TESTS PASSED` on the GPU machine
- [ ] Baseline text is coherent
- [ ] Tokenizer compatibility asserted in code, not just by eye
- [ ] `c` measured 3 times, mean and spread recorded
- [ ] Baseline tokens/s and TTFT recorded with standard deviations
- [ ] Predicted speed-up table saved to a file

### What to report back at the end of Phase 1

The measured `c`, the baseline tokens/s, and the best γ the formula predicts at α = 0.7. If `c` is above ~0.5, say so explicitly: it means the achievable speed-up is small, and P5 should include the `torch.compile` / static-cache route for the draft model.

---

## 8. Testing policy

- Tests run before and after every change to the engine. A change that breaks a test gets reverted, not "fixed" by relaxing the test.
- New feature → new test in the same style (tiny random models, CPU, no downloads).
- Existing coverage: lossless `verify` (40k samples, TV 0.0066), greedy equality vs baseline with a bad draft (α ≈ 0.07) and a good draft (α ≈ 0.74), draft = target giving α = 1.0, two-token joint distribution vs exact enumeration, EOS and length limits, formulas and controller.
- **Mutation check:** replacing the residual with p makes the `verify` test's TV jump from 0.0066 to 0.19. Any new test must fail for its corresponding bug; verify this once when writing it.

---

## 9. Benchmark protocol

- Warm-up runs before timing; `torch.cuda.synchronize()` around every timer.
- Same seed per prompt across methods.
- 3 repeats minimum; report mean ± standard deviation.
- Workloads: code, maths, general chat (6 prompts each). Temperatures 0 and 0.7.
- One CSV row per (task, prompt, temperature, method) with: `new_tokens`, `seconds`, `tokens_per_s`, `ttft_ms`, `target_passes_per_token`, `alpha`, `peak_mem_gb`, `gpu_util`, `matches_baseline`.
- Always log the measured `c` and the predicted speed-up alongside the measured one.

### Known measurement traps

- **Greedy mismatch in FP16 is not automatically a bug.** Verification scores several positions per pass, the baseline one at a time, so rounding can flip a near-tie argmax. Rerun that prompt in float32: if the mismatch disappears, it was numerics.
- GPU utilisation is a busy-percentage, not efficiency.
- Never time a single run on a shared GPU.

---

## 10. AdaEDL baseline (Phase 4)

Read the paper before implementing: **AdaEDL**, arXiv:2410.18351 (Agrawal, Jeon, Lee — NeurIPS ENLSP 2024). It stops drafting early using an entropy-based lower bound on the acceptance probability of the next draft token; it is training-free and parameter-light.

Do not implement it from memory or from a guessed formula — take the stopping criterion from the paper. Implement it as a drop-in alternative to `GammaController` so the two can be swapped in `bench.py` with a flag.

---

## 11. Definition of done for the whole project

- Engine passes all tests on GPU and CPU.
- Greedy outputs match the baseline, with any mismatches explained.
- `results.csv` covers every (task × temperature × method) cell.
- The O2 claim is answered with numbers, in whichever direction it came out.
- Plots: tokens/s vs γ per task; acceptance rate by draft position; predicted vs measured speed-up.
- Report ties every number back to a row in `results.csv`.

---

## 12. References

- Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192
- C. Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* arXiv:2302.01318
- S. Agrawal, W. Jeon, M. Lee. *AdaEDL: Early Draft Stopping for Speculative Decoding…* NeurIPS ENLSP 2024. arXiv:2410.18351
- Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115
- H. Xia et al. *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding.* Findings of ACL 2024. arXiv:2401.07851
