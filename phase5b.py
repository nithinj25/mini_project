"""Phase 5b: the O2 comparison again, with the static-cache CUDA-graph draft (SPEC §7's route when c > 0.5).

PRE-REGISTERED (committed to git before any Phase 5b run; do not edit after seeing results):

  Draft       GraphDraft(Qwen2.5-0.5B): static KV cache + CUDA-graph decode step. Measured on this laptop
              before registration: c = 0.121 ± 0.014 (vs 0.680 eager); the target stays eager for every
              method, including the baseline. Max draft length L = 8 for every method.
  Methods     baseline | fixed:2 | fixed:4 | fixed:6 | fixed:8 | adaptive (GammaController unchanged from
              Phase 1, c measured in the cell) | adaedl:0.7 (unchanged from Phase 4/5).
  Opponent a  best fixed γ PER TASK chosen AFTER THE FACT from this run (highest mean speed-up over that
              task's 12 cells). This favours the opponent. Secondary: the formula's a-priori pick at
              c = 0.12 (chat γ = 4, code γ = 8, maths γ = 8), and the per-cell oracle.
  Cells       3 tasks × 6 prompts × T ∈ {0, 0.7} = 36; 128 new tokens; same seed per prompt.
  Protocol    as Phase 5: per cell measure c; warm up every method; 3 repeats with the method order rotated
              by one per repeat; speed-up = mean over repeats of tokens/s ÷ the same repeat's baseline.
  Decision    as Phase 5: d = speed-up(adaptive) − speed-up(X) per cell over all 36 cells; "beats" iff the
              95% t-interval of d lies above 0, "loses" iff below, else no detectable difference.
              O2 (graph draft) holds only if adaptive beats both opponent a and AdaEDL.
  Expectation (written beforehand): beats AdaEDL; vs the after-the-fact best fixed γ, a tie or a small edge.

  python phase5b.py            # resumable
  python phase5b.py --analyse
"""
import argparse
import csv
import json
import math
import os

import torch

import phase5
from bench import DRAFT, TARGET, encode_prompt, eos_ids, load_pair, pick_device
from prompts import PROMPTS
from specdec import GraphDraft, mean_std
from winperf import disable_power_throttling

METHODS = ["baseline", "fixed:2", "fixed:4", "fixed:6", "fixed:8", "adaptive", "adaedl:0.7"]
FIXED = [m for m in METHODS if m.startswith("fixed:")]
A_PRIORI = {"chat": "fixed:4", "code": "fixed:8", "math": "fixed:8"}
OUT = "results/phase5b.csv"


def analyse():
    rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    by_task = {}
    for task in phase5.TASKS:
        rs = [r for r in rows if r["task"] == task]
        by_task[task] = max(FIXED, key=lambda m: mean_std(float(r["measured_speedup"]) for r in rs if r["method"] == m)[0])
    opponents = {
        "best_fixed_per_task_after_the_fact": lambda task, ms: by_task[task],
        "adaedl": lambda task, ms: "adaedl:0.7",
        "formula_a_priori_fixed": lambda task, ms: A_PRIORI[task],
        "oracle_best_fixed_per_cell": lambda task, ms: max(FIXED, key=lambda k: ms[k]),
    }
    out = {"cells": len({(r["task"], r["prompt_id"], r["temperature"]) for r in rows}),
           "best_fixed_per_task": by_task, "comparisons": {}}
    for name, opp in opponents.items():
        ds = phase5.paired(rows, "adaptive", opp)
        out["comparisons"][name] = {"all": phase5.verdict([d for _, d in ds])}
        for task in phase5.TASKS:
            out["comparisons"][name][task] = phase5.verdict([d for t, d in ds if t == task])
    a = out["comparisons"]["best_fixed_per_task_after_the_fact"]["all"]
    e = out["comparisons"]["adaedl"]["all"]
    out["O2_graph_draft_holds"] = a["verdict"] == "beats" and e["verdict"] == "beats"
    table = {}
    for task in phase5.TASKS:
        for temp in phase5.TEMPS:
            for m in METHODS:
                mr = [r for r in rows if r["task"] == task and float(r["temperature"]) == temp and r["method"] == m]
                if not mr:
                    continue
                sp, sd = mean_std(float(r["measured_speedup"]) for r in mr)
                entry = {"speedup": round(sp, 4), "sd_prompts": round(sd, 4),
                         "tokens_per_s": round(mean_std(float(r["tokens_per_s"]) for r in mr)[0], 2)}
                if m != "baseline":
                    entry.update(alpha=round(mean_std(float(r["alpha"]) for r in mr)[0], 4),
                                 mean_gamma=round(mean_std(float(r["mean_gamma"]) for r in mr)[0], 3),
                                 predicted=round(mean_std(float(r["predicted_speedup"]) for r in mr)[0], 4))
                    if temp == 0:
                        entry["greedy_matches"] = f"{sum(r['matches_baseline'] == 'True' for r in mr)}/{len(mr)}"
                table[f"{task} T={temp} {m}"] = entry
    out["table"] = table
    out["c_per_cell"] = dict(zip(("mean", "sd"), (round(x, 4) for x in mean_std(
        float(r["c"]) for r in rows if r["method"] == "baseline"))))
    with open("results/phase5b_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=2))


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
    gd = GraphDraft(draft, max_len=1024)
    eos = eos_ids(target)
    phase5.METHODS = METHODS  # run_cell reads the module-level method list
    done = set()
    if os.path.exists(OUT):
        done = {(r["task"], int(r["prompt_id"]), float(r["temperature"])) for r in csv.DictReader(open(OUT, encoding="utf-8"))}
    for task in phase5.TASKS:
        for pid in range(6):
            ids = encode_prompt(tok, PROMPTS[task][pid])
            for temp in phase5.TEMPS:
                if (task, pid, temp) in done:
                    print(f"skip {task} p{pid} T={temp}")
                    continue
                rows = phase5.run_cell(task, pid, temp, ids, target, gd, eos, device, gpu)
                for r in rows:
                    r["phase"] = "P5b"
                new = not os.path.exists(OUT)
                with open(OUT, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=phase5.P5_FIELDS)
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                print(f"{task:4s} p{pid} T={temp} c={rows[0]['c']}: " + "  ".join(
                    f"{r['method']} x{r['measured_speedup']:.2f}" + (f"(γ̄{r['mean_gamma']:.1f})" if r["method"] != "baseline" else "")
                    for r in rows), flush=True)
    analyse()


if __name__ == "__main__":
    main()
