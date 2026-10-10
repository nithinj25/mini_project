"""Phase 6 figures (SPEC §11) from the result CSVs. Writes results/figures/*.png.

  F1 speed-up vs γ per task: eager draft (P3, c≈0.7) and CUDA-graph draft (P5b, c≈0.12)
  F2 acceptance rate by draft position (P3, γ = 8)
  F3 predicted vs measured speed-up (P3 + P5 eager; P5b graph)
  F4 the O2 test: per-cell paired differences, adaptive − opponent, with 95% CIs (P5, P5b)
  F5 AdaEDL: acceptance vs draft entropy against the paper's bound; draft length vs entropy (P4)
"""
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from specdec import mean_std  # noqa: E402

OUT = "results/figures"
TASKS = ["code", "math", "chat"]
NAMES = {"code": "Code", "math": "Maths", "chat": "Chat"}
COLOR = {"code": "#2a78d6", "math": "#eb6834", "chat": "#1baf7a"}  # validated categorical slots 1-3
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
T_CRIT = {5: 2.571, 11: 2.201, 35: 2.030}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white"})


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def gamma(method):
    return int(method.split(":")[1])


def speedups(rows, task, temp, method):
    return [float(r["measured_speedup"]) for r in rows
            if r["task"] == task and float(r["temperature"]) == temp and r["method"] == method]


def baseline_line(ax):
    ax.axhline(1.0, color=NEUTRAL, lw=1.2, ls="--", zorder=1)


def label_ends(ax, x, ends, min_gap_frac=0.045):
    """Direct labels at line ends, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    placed = []
    for task, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(NAMES[task], (x, ends[task]), xytext=(x + 0.18, y), textcoords="data", va="center",
                    color=MUTED, fontsize=9)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ------------------------------------------------------------------ F1
def fig_speedup_vs_gamma(p3, p5b):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    def c_of(rows):
        return mean_std(float(r["c"]) for r in rows if r["method"] == "baseline")[0]

    for row, (rows, label) in enumerate([(p3, f"Eager draft (P3, c ≈ {c_of(p3):.2f})"),
                                         (p5b, f"CUDA-graph draft (P5b, c ≈ {c_of(p5b):.2f})")]):
        fixed = sorted({r["method"] for r in rows if r["method"].startswith("fixed:")}, key=gamma)
        for col, temp in enumerate((0.0, 0.7)):
            ax = axes[row][col]
            baseline_line(ax)
            ends = {}
            for task in TASKS:
                xs = [gamma(m) for m in fixed]
                ms = [mean_std(speedups(rows, task, temp, m)) for m in fixed]
                ax.errorbar(xs, [m for m, _ in ms], yerr=[s for _, s in ms], color=COLOR[task], lw=2,
                            marker="o", ms=6, capsize=3, elinewidth=1, label=NAMES[task], zorder=3)
                ends[task] = ms[-1][0]
            label_ends(ax, xs[-1], ends)
            ax.set_title(f"{label} · T = {temp:g}", loc="left")
            ax.set_ylabel("speed-up vs base model (×)")
            ax.set_xticks(range(1, 9))
            ax.set_xlim(0.6, 9.2)
    for ax in axes[1]:
        ax.set_xlabel("γ (tokens drafted per round, fixed)")
    axes[0][0].legend(loc="lower left", ncols=3)
    fig.suptitle("Speed-up vs draft length γ, by task (mean ± sd over 6 prompts; dashed = base model)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "F1_speedup_vs_gamma.png")


# ------------------------------------------------------------------ F2
def fig_acceptance_by_position(p3):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, temp in zip(axes, (0.0, 0.7)):
        for task in TASKS:
            tested, accepted = [0] * 8, [0] * 8
            for r in p3:
                if r["task"] == task and float(r["temperature"]) == temp and r["method"] == "fixed:8":
                    for i, (t, a) in enumerate(zip(r["pos_tested"].split(";"), r["pos_accepted"].split(";"))):
                        tested[i] += int(t)
                        accepted[i] += int(a)
            rate = [a / t for a, t in zip(accepted, tested)]
            ax.plot(range(1, 9), rate, color=COLOR[task], lw=2, marker="o", ms=6, label=NAMES[task])
            if task == "chat":
                for i in (6, 7):
                    ax.annotate(f"n={tested[i]}", (i + 1, rate[i]), xytext=(0, -14), textcoords="offset points",
                                ha="center", fontsize=8, color=MUTED)
            ax.annotate(NAMES[task], (8, rate[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        color=MUTED, fontsize=9)
        ax.set_title(f"T = {temp:g}", loc="left")
        ax.set_xlabel("position of the guess within the draft")
        ax.set_xlim(0.6, 9.2)
        ax.set_xticks(range(1, 9))
    axes[0].set_ylabel("acceptance rate (kept ÷ tested)")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(loc="lower left", ncols=3)
    fig.suptitle("Acceptance rate by draft position (P3, γ = 8, all 6 prompts per task; "
                 "n = guesses tested at sparse chat positions)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "F2_acceptance_by_position.png")


# ------------------------------------------------------------------ F3
def fig_predicted_vs_measured(p3, p5, p5b):
    marker = {"fixed": "o", "adaptive": "^", "adaedl": "s"}

    def family(m):
        return "fixed" if m.startswith("fixed:") else "adaptive" if m == "adaptive" else "adaedl"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [([("P3", p3), ("P5", [r for r in p5 if not r["method"].startswith("fixed:")])], "Eager draft (P3 + P5)"),
              ([("P5b", p5b)], "CUDA-graph draft (P5b)")]
    for ax, (sets, title) in zip(axes, panels):
        pts = []
        for _, rows in sets:
            groups = defaultdict(list)
            for r in rows:
                if r["method"] != "baseline":
                    groups[(r["task"], float(r["temperature"]), r["method"])].append(r)
            for (task, temp, m), rs in groups.items():
                x = mean_std(float(r["predicted_speedup"]) for r in rs)[0]
                y = mean_std(float(r["measured_speedup"]) for r in rs)[0]
                pts.append((x, y))
                ax.scatter(x, y, s=46, marker=marker[family(m)], color=COLOR[task], edgecolor="white",
                           linewidth=1.2, zorder=3)
        lo = min(min(p) for p in pts) - 0.1
        hi = max(max(p) for p in pts) + 0.1
        ax.plot([lo, hi], [lo, hi], color=NEUTRAL, ls="--", lw=1.2, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left")
        ax.set_xlabel("predicted  S = E / (1 + γ̄c)  (×)")
        ax.set_ylabel("measured speed-up (×)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=COLOR[t], ms=7, label=NAMES[t]) for t in TASKS]
    handles += [plt.Line2D([], [], ls="", marker=marker[f], color=NEUTRAL, ms=7, label=lab)
                for f, lab in (("fixed", "fixed γ"), ("adaptive", "adaptive γ"), ("adaedl", "AdaEDL"))]
    handles += [plt.Line2D([], [], ls="--", color=NEUTRAL, label="prediction = measurement")]
    fig.legend(handles=handles, loc="lower center", ncols=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Predicted vs measured speed-up (each point: one task × temperature × method, mean over prompts)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "F3_predicted_vs_measured.png")


# ------------------------------------------------------------------ F4
def paired(rows, opponent):
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["task"], r["prompt_id"], r["temperature"])][r["method"]] = float(r["measured_speedup"])
    return [ms["adaptive"] - ms[opponent(task, ms)] for (task, _, _), ms in cells.items()]


def fig_o2(p5, p5b, s5, s5b):
    best5b = s5b["best_fixed_per_task"]
    comps = [
        ("P5 eager · vs best fixed γ (a priori)",
         paired(p5, lambda t, ms: {"code": "fixed:1", "math": "fixed:2", "chat": "fixed:1"}[t]),
         s5["comparisons"]["best_fixed_a_priori"]["all"]),
        ("P5 eager · vs AdaEDL", paired(p5, lambda t, ms: "adaedl:0.7"), s5["comparisons"]["adaedl"]["all"]),
        ("P5b graph · vs best fixed γ (after the fact)", paired(p5b, lambda t, ms: best5b[t]),
         s5b["comparisons"]["best_fixed_per_task_after_the_fact"]["all"]),
        ("P5b graph · vs AdaEDL", paired(p5b, lambda t, ms: "adaedl:0.7"), s5b["comparisons"]["adaedl"]["all"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axvline(0, color=NEUTRAL, lw=1.2, ls="--", zorder=1)
    for i, (label, ds, summ) in enumerate(comps):
        y = len(comps) - 1 - i
        jit = [((k * 37) % 11 - 5) / 40 for k in range(len(ds))]
        ax.scatter(ds, [y + j for j in jit], s=14, color="#9ec5f4", edgecolor="none", zorder=2)
        m, (lo, hi) = summ["mean_diff"], summ["ci95"]
        assert abs(m - mean_std(ds)[0]) < 1e-3, "plot data and summary disagree"
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=7, capsize=5, lw=2, zorder=4)
        verdict = summ["verdict"].replace(" vs", "")
        ax.annotate(f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]" + chr(10) + f"→ {verdict}", (1.01, y),
                    xycoords=("axes fraction", "data"), va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([c[0] for c in reversed(comps)])
    ax.set_xlabel("speed-up(adaptive γ) − speed-up(opponent), per cell  (dots: 36 cells; bar: mean, 95% CI)")
    ax.grid(axis="y", visible=False)
    lo = min(min(c[1]) for c in comps)
    hi = max(max(c[1]) for c in comps)
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.set_title("The O2 test: does cost-ratio-aware adaptive γ beat its opponents?", loc="left")
    fig.tight_layout()
    save(fig, "F4_o2_paired_differences.png")


# ------------------------------------------------------------------ F5
def fig_adaedl(p4_summary, p4_runs):
    acc = p4_summary["acceptance_by_entropy"]
    bins = list(acc)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    xs = range(len(bins))
    ax.plot(xs, [acc[b]["acceptance"] for b in bins], color=COLOR["code"], lw=2, marker="o", ms=6, label="measured acceptance")
    ax.plot(xs, [acc[b]["mean_bound"] for b in bins], color=NEUTRAL, lw=2, ls="--", marker="s", ms=5,
            label="paper's bound 1 − √(0.2·H)")
    for x, b in zip(xs, bins):
        ax.annotate(f"n={acc[b]['tested']}", (x, acc[b]["acceptance"]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b.replace("-inf", "+") for b in bins])
    ax.set_xlabel("entropy of the draft distribution (nats)")
    ax.set_ylabel("acceptance of the drafted token")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    ax.set_title("Acceptance falls with entropy; the bound holds", loc="left")
    ax = axes[1]
    for task in TASKS:
        rs = [r for r in p4_runs if r["task"] == task and r["lam0"] == "0.7"]
        ax.scatter([float(r["mean_entropy"]) for r in rs], [float(r["mean_draft_len"]) for r in rs], s=46,
                   color=COLOR[task], edgecolor="white", linewidth=1.2, label=NAMES[task], zorder=3)
    ax.set_xlabel("mean draft entropy of the run (nats)")
    ax.set_ylabel("mean draft length (tokens)")
    ax.legend(loc="upper right")
    ax.set_title("AdaEDL drafts less when the draft is unsure", loc="left")
    fig.suptitle("AdaEDL on the Qwen pair (P4, λ0 = 0.7, L = 8, 18 prompts × 2 temperatures)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "F5_adaedl_entropy.png")


def main():
    p3, p5, p5b = load("results/phase3.csv"), load("results/phase5.csv"), load("results/phase5b.csv")
    fig_speedup_vs_gamma(p3, p5b)
    fig_acceptance_by_position(p3)
    fig_predicted_vs_measured(p3, p5, p5b)
    fig_o2(p5, p5b, json.load(open("results/phase5_summary.json", encoding="utf-8")),
           json.load(open("results/phase5b_summary.json", encoding="utf-8")))
    fig_adaedl(json.load(open("results/phase4.json", encoding="utf-8")), load("results/phase4_runs.csv"))


if __name__ == "__main__":
    main()
