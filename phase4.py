"""Phase 4 (SPEC §6, §10): AdaEDL on the real Qwen pair.

Done-criterion: reproduce the paper's qualitative behaviour, shorter drafts at high entropy.
Measured here:
  1. stop rate of the AdaEDL check vs draft entropy H (should rise with H)
  2. draft length vs entropy, per round and per prompt (should fall as H rises; chat < maths)
  3. the paper's premise: acceptance of a drafted token vs its entropy, against the bound 1 - sqrt(0.2 H)
  4. greedy output still equals the baseline
  5. initial-λ sensitivity (λ0 ∈ {0.3, 0.5, 0.7, 0.9}), to choose λ0 for Phase 5
Speeds here are single indicative runs; the timed comparison is Phase 5.

  python phase4.py [--prompts 6] [--max-new 128]

Writes results/phase4_runs.csv, results/phase4_steps.csv, results/phase4.json.
"""
import argparse
import csv
import json
import math
from collections import defaultdict

from bench import DRAFT, TARGET, encode_prompt, eos_ids, load_pair, pick_device
from prompts import PROMPTS
from specdec import AdaEDL, FixedGamma, baseline_generate, mean_std, speculative_generate
from winperf import disable_power_throttling

TASKS = ("code", "math", "chat")
BINS = [0, 0.25, 0.5, 1.0, 2.0, 4.0, 99]
SEED = 17


def bin_of(h):
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= h < hi:
            return f"{lo}-{hi if hi < 99 else 'inf'}"


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sx * sy) if sx and sy else float("nan")


def run_adaedl(target, draft, ids, max_new, temp, eos, lam0):
    ctl = AdaEDL(max_draft=8, lam0=lam0, temperature=temp, log_all=True)
    out, st = speculative_generate(target, draft, ids, max_new, ctl, temp, eos, SEED)
    return out, st, ctl


def step_rows(task, pid, temp, lam0, ctl):
    """One row per checked draft position, with what happened to the token (if it was drafted)."""
    rows = []
    for rnd, i, h, stop in ctl.steps:
        n_drafted, n_acc, _ = ctl.rounds[rnd]
        if stop:
            outcome = "stopped"           # AdaEDL declined to draft this position
        elif i < n_acc:
            outcome = "accepted"
        elif i == n_acc:
            outcome = "rejected"
        else:
            outcome = "untested"          # drafted after an earlier rejection
        rows.append({"task": task, "prompt_id": pid, "temperature": temp, "lam0": lam0, "round": rnd,
                     "position": i, "entropy": round(h, 5), "stopped": stop, "outcome": outcome,
                     "round_draft_len": n_drafted})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    disable_power_throttling()
    device = pick_device()
    tok, target, draft = load_pair(TARGET, DRAFT, "fp16", device)
    eos = eos_ids(target)
    w = encode_prompt(tok, "Hello")
    baseline_generate(target, w, 16, 0.0, eos)
    speculative_generate(target, draft, w, 16, FixedGamma(2), 0.0, eos)

    runs, steps = [], []
    configs = [(t, 0.7) for t in (0.0, 0.7)] + [(0.7, l0) for l0 in (0.3, 0.5, 0.9)]
    for task in TASKS:
        for pid in range(a.prompts):
            ids = encode_prompt(tok, PROMPTS[task][pid])
            ref, bst = baseline_generate(target, ids, a.max_new, 0.0, eos, SEED)
            for temp, lam0 in configs:
                if lam0 != 0.7 and pid >= 2:  # λ0 sweep on 2 prompts per task
                    continue
                out, st, ctl = run_adaedl(target, draft, ids, a.max_new, temp, eos, lam0)
                hs = [h for _, _, h, _ in ctl.steps]
                runs.append({"task": task, "prompt_id": pid, "temperature": temp, "lam0": lam0,
                             "new_tokens": st.new_tokens, "tokens_per_s": round(st.tokens_per_s, 2),
                             "baseline_tokens_per_s": round(bst.tokens_per_s, 2),
                             "speedup_indicative": round(st.tokens_per_s / bst.tokens_per_s, 3),
                             "alpha": round(st.alpha, 4), "mean_draft_len": round(st.mean_gamma, 3),
                             "rounds": st.rounds, "early_stops": sum(1 for *_, s in ctl.steps if s),
                             "mean_entropy": round(sum(hs) / len(hs), 4), "final_lam": round(ctl.lam, 4),
                             "matches_baseline": (out == ref) if temp == 0 else ""})
                steps += step_rows(task, pid, temp, lam0, ctl)
                r = runs[-1]
                print(f"{task:4s} p{pid} T={temp} λ0={lam0}: draft len {r['mean_draft_len']:.2f}  "
                      f"H̄ {r['mean_entropy']:.2f}  α {r['alpha']:.3f}  λ→{r['final_lam']:.3f}  "
                      f"x{r['speedup_indicative']:.2f}  match={r['matches_baseline']}", flush=True)

    for name, rows in (("results/phase4_runs.csv", runs), ("results/phase4_steps.csv", steps)):
        with open(name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    # ---------------- analysis on the main configuration (λ0 = 0.7, both temperatures)
    main_steps = [s for s in steps if s["lam0"] == 0.7]
    main_runs = [r for r in runs if r["lam0"] == 0.7]
    summary = {"lam0_main": 0.7, "max_draft": 8, "entropy_factor": 0.2}

    # 1. stop rate by entropy (checks at positions >= 1, where the rule applies)
    by_bin = defaultdict(lambda: [0, 0])
    for s in main_steps:
        if s["position"] >= 1:
            b = by_bin[bin_of(s["entropy"])]
            b[0] += s["stopped"]
            b[1] += 1
    summary["stop_rate_by_entropy"] = {k: {"stop_rate": round(v[0] / v[1], 4), "checks": v[1]}
                                       for k, v in sorted(by_bin.items(), key=lambda kv: float(kv[0].split("-")[0]))}

    # 2. draft length vs entropy: per round (mean H of the round's checks) and per prompt
    rounds = defaultdict(list)
    for s in main_steps:
        rounds[(s["task"], s["prompt_id"], s["temperature"], s["round"])].append(s)
    r_h = [sum(x["entropy"] for x in v) / len(v) for v in rounds.values()]
    r_len = [v[0]["round_draft_len"] for v in rounds.values()]
    summary["corr_round_entropy_vs_draft_len"] = round(pearson(r_h, r_len), 4)
    summary["corr_prompt_entropy_vs_draft_len"] = round(
        pearson([r["mean_entropy"] for r in main_runs], [r["mean_draft_len"] for r in main_runs]), 4)
    summary["by_task"] = {}
    for task in TASKS:
        for temp in (0.0, 0.7):
            rs = [r for r in main_runs if r["task"] == task and r["temperature"] == temp]
            summary["by_task"][f"{task} T={temp}"] = {
                "mean_entropy": round(mean_std(r["mean_entropy"] for r in rs)[0], 4),
                "mean_draft_len": round(mean_std(r["mean_draft_len"] for r in rs)[0], 3),
                "alpha": round(mean_std(r["alpha"] for r in rs)[0], 4),
                "speedup_indicative": round(mean_std(r["speedup_indicative"] for r in rs)[0], 3)}

    # 3. premise: acceptance of drafted tokens vs entropy, against the bound 1 - sqrt(0.2 H)
    acc = defaultdict(lambda: [0, 0, 0.0])
    for s in main_steps:
        if s["outcome"] in ("accepted", "rejected"):
            b = acc[bin_of(s["entropy"])]
            b[0] += s["outcome"] == "accepted"
            b[1] += 1
            b[2] += 1 - math.sqrt(0.2 * s["entropy"])
    summary["acceptance_by_entropy"] = {k: {"acceptance": round(v[0] / v[1], 4), "tested": v[1],
                                            "mean_bound": round(v[2] / v[1], 4)}
                                        for k, v in sorted(acc.items(), key=lambda kv: float(kv[0].split("-")[0]))}

    # 4. greedy equality
    greedy = [r for r in main_runs if r["temperature"] == 0.0]
    summary["greedy_matches"] = f"{sum(r['matches_baseline'] for r in greedy)}/{len(greedy)}"

    # 5. λ0 sensitivity (T = 0.7, prompts 0-1 of each task)
    summary["lam0_sweep"] = {}
    for l0 in (0.3, 0.5, 0.7, 0.9):
        rs = [r for r in runs if r["lam0"] == l0 and r["temperature"] == 0.7 and r["prompt_id"] < 2]
        summary["lam0_sweep"][str(l0)] = {
            "mean_draft_len": round(mean_std(r["mean_draft_len"] for r in rs)[0], 3),
            "alpha": round(mean_std(r["alpha"] for r in rs)[0], 4),
            "speedup_indicative": round(mean_std(r["speedup_indicative"] for r in rs)[0], 3)}

    with open("results/phase4.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
