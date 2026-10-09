"""Phase 5 (SPEC §2 O2, §6): does cost-ratio-aware adaptive γ beat (a) the best fixed γ and (b) AdaEDL?

PRE-REGISTERED (committed to git before any Phase 5 run; do not edit after seeing results):

  Methods     baseline | fixed:1 | fixed:2 | adaptive (GammaController exactly as written in Phase 1:
              α0 0.7, prior weight 4, decay 0.95, γ ∈ 1..8, c measured in the same cell) | adaedl:0.7
              (AdaEDL, λ0 0.7 chosen in Phase 4, L = 8, paper hyperparameters).
  Opponent a  best fixed γ chosen a priori from Phase 3: γ = 1 for code and chat, γ = 2 for maths.
  Cells       3 tasks × 6 prompts × T ∈ {0, 0.7} = 36; 128 new tokens; same seed per prompt.
  Protocol    per cell: measure c; warm up every method once; then 3 repeats, each running every method
              once in an order rotated by one position per repeat. A method's speed-up in a cell is the
              mean over repeats of (its tokens/s ÷ the baseline's tokens/s in the same repeat).
  Decision    for each opponent X, d = speed-up(adaptive) − speed-up(X) per cell, over all 36 cells:
                beats X      mean d > 0 and the 95% t-interval of d lies above 0
                loses to X   the 95% t-interval lies below 0
                no detectable difference otherwise
              O2 holds only if adaptive beats both opponent a and AdaEDL.
  Secondary   per-task breakdown; the per-cell oracle best of fixed:1/fixed:2 (picked after the fact,
              so a stricter opponent); predicted vs measured speed-up for the adaptive γ.
  Expectation (from Phases 3-4, written beforehand): beats AdaEDL; roughly ties the best fixed γ.

  python phase5.py            # resumable: finished cells in results/phase5.csv are skipped
  python phase5.py --analyse  # only recompute results/phase5_summary.json from the CSV
"""
import argparse
import csv
import json
import math
import os

import torch

from bench import (DRAFT, FIELDS, TARGET, GpuMonitor, encode_prompt, eos_ids, load_pair, make_controller,
                   pick_device)
from prompts import PROMPTS
from specdec import (baseline_generate, mean_std, measure_cost_ratio, predicted_speedup,
                     speculative_generate)
from winperf import disable_power_throttling

TASKS = ("code", "math", "chat")
TEMPS = (0.0, 0.7)
METHODS = ["baseline", "fixed:1", "fixed:2", "adaptive", "adaedl:0.7"]
BEST_FIXED = {"code": "fixed:1", "math": "fixed:2", "chat": "fixed:1"}  # a priori, from Phase 3
REPEATS, MAX_NEW, OUT = 3, 128, "results/phase5.csv"
P5_FIELDS = FIELDS + ["speedup_sd_repeats", "mean_gamma", "order_positions", "phase"]
T_CRIT = {5: 2.571, 11: 2.201, 35: 2.030}  # two-sided 95% t quantiles for the n - 1 used below


def run(method, target, draft, ids, temp, eos, seed, c):
    if method == "baseline":
        return baseline_generate(target, ids, MAX_NEW, temp, eos, seed)
    return speculative_generate(target, draft, ids, MAX_NEW, make_controller(method, c, temp), temp, eos, seed)


def run_cell(task, pid, temp, ids, target, draft, eos, device, gpu):
    seed = 1000 * pid + 17
    c = measure_cost_ratio(target, draft, device)["c"]
    for m in METHODS:                      # warm-up, untimed
        run(m, target, draft, ids, temp, eos, seed, c)
    torch.cuda.reset_peak_memory_stats()
    stats = {m: [] for m in METHODS}
    outs, positions = {}, {m: [] for m in METHODS}
    with GpuMonitor() as mon:
        for r in range(REPEATS):
            order = METHODS[r % len(METHODS):] + METHODS[:r % len(METHODS)]
            for pos, m in enumerate(order):
                out, st = run(m, target, draft, ids, temp, eos, seed, c)
                stats[m].append(st)
                outs[m] = out
                positions[m].append(pos)
    peak = torch.cuda.max_memory_allocated() / 2**30
    base = stats["baseline"]
    rows = []
    for m in METHODS:
        runs = stats[m]
        tps, tps_sd = mean_std(s.tokens_per_s for s in runs)
        ttft, ttft_sd = mean_std(1e3 * s.ttft_s for s in runs)
        ratios = [s.tokens_per_s / b.tokens_per_s for s, b in zip(runs, base)]  # paired by repeat
        sp, sp_sd = mean_std(ratios)
        spec = m != "baseline"
        alpha = sum(s.accepted for s in runs) / max(1, sum(s.tested for s in runs)) if spec else float("nan")
        g = mean_std(s.mean_gamma for s in runs)[0] if spec else 0.0
        last = runs[-1]
        rows.append({
            "task": task, "prompt_id": pid, "temperature": temp, "method": m,
            "gamma_mean": round(g, 3) if spec else "", "repeats": len(runs), "new_tokens": last.new_tokens,
            "seconds": round(mean_std(s.seconds for s in runs)[0], 4),
            "tokens_per_s": round(tps, 3), "tokens_per_s_std": round(tps_sd, 3),
            "ttft_ms": round(ttft, 3), "ttft_ms_std": round(ttft_sd, 3),
            "target_passes_per_token": round(mean_std(s.target_passes_per_token for s in runs)[0], 4),
            "alpha": round(alpha, 4) if spec else "", "peak_mem_gb": round(peak, 3), "gpu_util": round(mon.mean, 1),
            "matches_baseline": (outs[m] == outs["baseline"]) if temp == 0 else "",
            "c": round(c, 4), "predicted_speedup": round(predicted_speedup(alpha, g, c), 4) if spec else 1.0,
            "measured_speedup": round(sp, 4),
            "pos_tested": ";".join(map(str, last.pos_tested)), "pos_accepted": ";".join(map(str, last.pos_accepted)),
            "dtype": "fp16", "gpu": gpu,
            "speedup_sd_repeats": round(sp_sd, 4), "mean_gamma": round(g, 3),
            "order_positions": ";".join(map(str, positions[m])), "phase": "P5"})
    return rows


def paired(rows, a, b_of):
    """Per-cell differences speed-up(a) − speed-up(b), where b_of(task, cell_rows) names the opponent."""
    cells = {}
    for r in rows:
        cells.setdefault((r["task"], r["prompt_id"], r["temperature"]), {})[r["method"]] = float(r["measured_speedup"])
    return [(k[0], ms[a] - ms[b_of(k[0], ms)]) for k, ms in cells.items()]


def verdict(ds):
    n = len(ds)
    m, sd = mean_std(ds)
    half = T_CRIT[n - 1] * sd / math.sqrt(n)
    lo, hi = m - half, m + half
    word = "beats" if lo > 0 else "loses to" if hi < 0 else "no detectable difference vs"
    return {"n": n, "mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "wins": sum(d > 0 for d in ds), "verdict": word}


def analyse():
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    opponents = {
        "best_fixed_a_priori": lambda task, ms: BEST_FIXED[task],
        "adaedl": lambda task, ms: "adaedl:0.7",
        "oracle_best_fixed_per_cell": lambda task, ms: max(("fixed:1", "fixed:2"), key=lambda k: ms[k]),
    }
    out = {"cells": len({(r["task"], r["prompt_id"], r["temperature"]) for r in rows}), "comparisons": {}}
    for name, opp in opponents.items():
        ds = paired(rows, "adaptive", opp)
        out["comparisons"][name] = {"all": verdict([d for _, d in ds])}
        for task in TASKS:
            out["comparisons"][name][task] = verdict([d for t, d in ds if t == task])
    a, e = out["comparisons"]["best_fixed_a_priori"]["all"], out["comparisons"]["adaedl"]["all"]
    out["O2_holds"] = a["verdict"] == "beats" and e["verdict"] == "beats"

    table = {}
    for task in TASKS:
        for temp in TEMPS:
            rs = [r for r in rows if r["task"] == task and float(r["temperature"]) == temp]
            for m in METHODS:
                mr = [r for r in rs if r["method"] == m]
                if not mr:
                    continue
                sp, sd = mean_std(float(r["measured_speedup"]) for r in mr)
                entry = {"speedup": round(sp, 4), "sd_prompts": round(sd, 4),
                         "tokens_per_s": round(mean_std(float(r["tokens_per_s"]) for r in mr)[0], 2)}
                if m != "baseline":
                    entry["alpha"] = round(mean_std(float(r["alpha"]) for r in mr)[0], 4)
                    entry["mean_gamma"] = round(mean_std(float(r["mean_gamma"]) for r in mr)[0], 3)
                    entry["predicted"] = round(mean_std(float(r["predicted_speedup"]) for r in mr)[0], 4)
                if temp == 0 and m != "baseline":
                    entry["greedy_matches"] = f"{sum(r['matches_baseline'] == 'True' for r in mr)}/{len(mr)}"
                table[f"{task} T={temp} {m}"] = entry
    out["table"] = table
    out["c_per_cell"] = dict(zip(("mean", "sd"), (round(x, 4) for x in mean_std(
        float(r["c"]) for r in rows if r["method"] == "baseline"))))
    with open("results/phase5_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    disable_power_throttling()
    device = pick_device()
    gpu = torch.cuda.get_device_name(device)
    tok, target, draft = load_pair(TARGET, DRAFT, "fp16", device)
    eos = eos_ids(target)
    done = set()
    if os.path.exists(OUT):
        done = {(r["task"], int(r["prompt_id"]), float(r["temperature"])) for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    for task in TASKS:
        for pid in range(6):
            ids = encode_prompt(tok, PROMPTS[task][pid])
            for temp in TEMPS:
                if (task, pid, temp) in done:
                    print(f"skip {task} p{pid} T={temp}")
                    continue
                rows = run_cell(task, pid, temp, ids, target, draft, eos, device, gpu)
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=P5_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                print(f"{task:4s} p{pid} T={temp} c={rows[0]['c']}: " + "  ".join(
                    f"{r['method']} x{r['measured_speedup']:.3f}" + (f"(γ̄{r['mean_gamma']:.1f})" if r["method"] != "baseline" else "")
                    for r in rows), flush=True)
    analyse()


if __name__ == "__main__":
    main()
