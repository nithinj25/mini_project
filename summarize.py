"""Aggregate results/results.csv per (task, temperature, method): the Phase 3 table (SPEC §6).

  python summarize.py [--csv results/results.csv] [--out results/phase3_summary.csv]

Speed-ups are per prompt (method tokens/s ÷ that prompt's baseline tokens/s), then averaged,
so a slow prompt does not dominate. Every number here traces back to rows of results.csv.
"""
import argparse
import csv
from collections import defaultdict

from specdec import mean_std, predicted_speedup


def gamma_of(method: str):
    return int(method.split(":")[1]) if method.startswith("fixed:") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--out", default="results/phase3_summary.csv")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], float(r["temperature"]), r["method"])].append(r)

    out = []
    for (task, temp, method), rs in groups.items():
        f = lambda k: [float(r[k]) for r in rs if r[k] != ""]
        tps, tps_sd = mean_std(f("tokens_per_s"))
        sp, sp_sd = mean_std(f("measured_speedup"))
        alpha = mean_std(f("alpha"))[0] if f("alpha") else float("nan")
        c = mean_std(f("c"))[0]
        g = gamma_of(method)
        matches = [r["matches_baseline"] for r in rs if r["matches_baseline"] != ""]
        out.append({
            "task": task, "temperature": temp, "method": method, "gamma": g if g is not None else "",
            "prompts": len(rs), "tokens_per_s": round(tps, 2), "tokens_per_s_sd_across_prompts": round(tps_sd, 2),
            "measured_speedup": round(sp, 3), "speedup_sd_across_prompts": round(sp_sd, 3),
            "alpha": round(alpha, 4) if alpha == alpha else "",
            "target_passes_per_token": round(mean_std(f("target_passes_per_token"))[0], 4),
            # prediction from the pooled α of this group, so it is comparable to the pooled measurement
            "predicted_speedup": round(predicted_speedup(alpha, g, c), 3) if g else 1.0,
            "greedy_matches": f"{sum(m == 'True' for m in matches)}/{len(matches)}" if matches else "",
            "c": round(c, 4),
        })
    order = {"code": 0, "math": 1, "chat": 2}
    out.sort(key=lambda r: (order.get(r["task"], 9), r["temperature"], r["gamma"] if r["gamma"] != "" else 0))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"{'task':5s} {'T':>4s} {'method':9s} {'tok/s':>7s} {'speedup':>13s} {'pred':>6s} {'alpha':>6s} {'pass/tok':>8s} match")
    for r in out:
        print(f"{r['task']:5s} {r['temperature']:4.1f} {r['method']:9s} {r['tokens_per_s']:7.2f} "
              f"{r['measured_speedup']:6.3f}±{r['speedup_sd_across_prompts']:.3f} {r['predicted_speedup']:6.3f} "
              f"{str(r['alpha']):>6s} {r['target_passes_per_token']:8.3f} {r['greedy_matches']}")

    print("\nbest fixed gamma per (task, T):")
    for task in order:
        for temp in sorted({r["temperature"] for r in out}):
            fx = [r for r in out if r["task"] == task and r["temperature"] == temp and r["gamma"] != ""]
            if fx:
                b = max(fx, key=lambda r: r["measured_speedup"])
                print(f"  {task:5s} T={temp}: gamma={b['gamma']}  x{b['measured_speedup']:.3f}  (alpha {b['alpha']})")


if __name__ == "__main__":
    main()
