"""Phase 1 (SPEC §7): real-model baseline, pair compatibility, measured c, predicted speed-ups.

  python phase1.py [--dtype fp16|bf16]

Writes results/phase1.json, results/phase1_baseline.csv, results/phase1_predictions.csv.
"""
import argparse
import csv
import json
import os

import torch

from bench import DRAFT, TARGET, arch, encode_prompt, eos_ids, load_pair, pick_device
from prompts import PROMPTS
from winperf import disable_power_throttling
from specdec import baseline_generate, best_gamma, mean_std, measure_cost_ratio, predicted_speedup

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
GAMMAS = range(1, 9)
SANITY_PROMPT = "Explain in two sentences what a hash table is."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    disable_power_throttling()
    device = pick_device()
    out = {"gpu": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
           "dtype": a.dtype, "torch": torch.__version__}
    import transformers
    out["transformers"] = transformers.__version__

    # 2-3. load, compatibility (asserted inside load_pair), architecture
    tok, target, draft = load_pair(TARGET, DRAFT, a.dtype, device)
    out["arch"] = {"target": arch(target), "draft": arch(draft)}
    assert out["arch"]["target"]["vocab_rows"] == out["arch"]["draft"]["vocab_rows"] == 151936
    out["weights_gb"] = torch.cuda.memory_allocated(device) / 2**30
    print(f"{out['gpu']} {a.dtype}  weights {out['weights_gb']:.2f} GB  arch {out['arch']}")

    eos = eos_ids(target)
    ids = encode_prompt(tok, SANITY_PROMPT)
    text, _ = baseline_generate(target, ids, 64, 0.0, eos)
    out["sanity_text"] = tok.decode(text, skip_special_tokens=True)
    print("\n--- sanity text (baseline, greedy, 64 tokens) ---\n" + out["sanity_text"] + "\n---")

    # 4. cost ratio, 3 repeats
    cs = [measure_cost_ratio(target, draft, device, context_len=256, steps=20, warmup=3) for _ in range(3)]
    c_mean, c_sd = mean_std(x["c"] for x in cs)
    out["c_runs"], out["c_mean"], out["c_std"] = cs, c_mean, c_sd
    print(f"c = {c_mean:.4f} ± {c_sd:.4f}   runs: " +
          ", ".join(f"{x['c']:.4f} ({x['draft_step_ms']:.2f}/{x['target_step_ms']:.2f} ms)" for x in cs))

    # 5. baseline: 3 prompts x 3 tasks, 128 new tokens, greedy, warm-up + repeats
    rows = []
    baseline_generate(target, ids, 32, 0.0, eos)
    for task in ("code", "math", "chat"):
        for pid in range(3):
            p_ids = encode_prompt(tok, PROMPTS[task][pid])
            baseline_generate(target, p_ids, 128, 0.0, eos)
            runs = [baseline_generate(target, p_ids, 128, 0.0, eos)[1] for _ in range(a.repeats)]
            tps, tps_sd = mean_std(r.tokens_per_s for r in runs)
            ttft, ttft_sd = mean_std(1e3 * r.ttft_s for r in runs)
            rows.append({"task": task, "prompt_id": pid, "prompt_tokens": len(p_ids), "new_tokens": runs[0].new_tokens,
                         "tokens_per_s": round(tps, 3), "tokens_per_s_std": round(tps_sd, 3),
                         "ttft_ms": round(ttft, 2), "ttft_ms_std": round(ttft_sd, 2)})
            print(f"baseline {task:4s} p{pid}: {tps:6.2f} ± {tps_sd:.2f} tok/s   TTFT {ttft:6.1f} ± {ttft_sd:.1f} ms"
                  f"   ({runs[0].new_tokens} tokens)")
    with open("results/phase1_baseline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    all_tps = [r["tokens_per_s"] for r in rows]
    out["baseline_tokens_per_s_mean"], out["baseline_tokens_per_s_std"] = mean_std(all_tps)
    out["baseline_ttft_ms_mean"], out["baseline_ttft_ms_std"] = mean_std(r["ttft_ms"] for r in rows)
    out["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30

    # 6. predictions, written before any speculative run
    with open("results/phase1_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha"] + [f"S_gamma{g}" for g in GAMMAS] + ["best_gamma", "best_S"])
        print(f"\npredicted S(alpha, gamma, c={c_mean:.4f})")
        print("alpha " + " ".join(f"  g={g} " for g in GAMMAS) + "  best")
        out["predicted_best_gamma"] = {}
        for al in ALPHAS:
            s = [predicted_speedup(al, g, c_mean) for g in GAMMAS]
            bg = best_gamma(al, c_mean)
            out["predicted_best_gamma"][str(al)] = bg
            w.writerow([al] + [round(x, 4) for x in s] + [bg, round(max(s), 4)])
            print(f"{al:4.1f}  " + " ".join(f"{x:6.3f}" for x in s) + f"   {bg}")

    with open("results/phase1.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nc = {c_mean:.4f} ± {c_sd:.4f} | baseline {out['baseline_tokens_per_s_mean']:.2f} ± "
          f"{out['baseline_tokens_per_s_std']:.2f} tok/s | best gamma @ alpha=0.7: {out['predicted_best_gamma']['0.7']}"
          f" | peak mem {out['peak_mem_gb']:.2f} GB")
    if c_mean > 0.5:
        print("WARNING: c > 0.5 -> achievable speed-up is small; P5 should include torch.compile / static cache for the draft.")


if __name__ == "__main__":
    main()
