"""Phase 2 (SPEC §6): speculative decoding verified on the real Qwen pair.

Greedy speculative output must equal the greedy baseline, or each mismatch must be explained
as FP16 numerics. SPEC §9 says to rerun a mismatch in float32, but the 3B target needs ~12 GB
in fp32 and this GPU has 8 GB. Instead, each mismatch gets a near-tie diagnosis: re-score the
common prefix with the target in one pass and report the logit gap between its top-2 tokens at
the first divergence. A gap within FP16 noise, with both tokens in the top 2, means a rounding
flip; a large gap would mean an engine bug.

  python phase2.py [--gammas 1,4,8] [--prompts 6] [--max-new 128]

Writes results/phase2.csv and results/phase2.json.
"""
import argparse
import csv
import json

import torch
from transformers import DynamicCache

from bench import DRAFT, TARGET, encode_prompt, eos_ids, load_pair, pick_device
from prompts import PROMPTS
from specdec import FixedGamma, baseline_generate, speculative_generate
from winperf import disable_power_throttling

NEAR_TIE = 0.25  # logits; FP16 accumulated error over 36 layers is of this order


@torch.inference_mode()
def divergence(target, prompt_ids, ref, out):
    """Index of the first differing token, plus the target's view of that position."""
    k = next((i for i, (a, b) in enumerate(zip(ref, out)) if a != b), min(len(ref), len(out)))
    if k == min(len(ref), len(out)):  # one is a prefix of the other (EOS placement)
        return {"first_div": k, "gap": None, "ref_rank": None, "spec_rank": None}
    ids = torch.tensor([prompt_ids + ref[:k]], device=target.device)
    logits = target(input_ids=ids, past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1).logits[0, -1].float()
    order = logits.argsort(descending=True)
    rank = {int(t): r for r, t in enumerate(order[:50].tolist())}
    top2 = logits[order[:2]]
    return {"first_div": k, "gap": round(float(top2[0] - top2[1]), 4),
            "ref_rank": rank.get(ref[k]), "spec_rank": rank.get(out[k])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--gammas", default="1,4,8")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=128)
    a = ap.parse_args()
    disable_power_throttling()
    device = pick_device()
    tok, target, draft = load_pair(TARGET, DRAFT, a.dtype, device)
    eos = eos_ids(target)
    gammas = [int(g) for g in a.gammas.split(",")]

    # warm-up so the first cell is not charged for CUDA/cuBLAS initialisation
    w = encode_prompt(tok, "Hello")
    baseline_generate(target, w, 16, 0.0, eos)
    speculative_generate(target, draft, w, 16, FixedGamma(4), 0.0, eos)

    rows = []
    for task in ("code", "math", "chat"):
        for pid in range(a.prompts):
            ids = encode_prompt(tok, PROMPTS[task][pid])
            ref, bst = baseline_generate(target, ids, a.max_new, 0.0, eos)
            for g in gammas:
                out, st = speculative_generate(target, draft, ids, a.max_new, FixedGamma(g), 0.0, eos)
                row = {"task": task, "prompt_id": pid, "gamma": g, "match": out == ref,
                       "new_tokens": len(out), "baseline_tokens": len(ref), "alpha": round(st.alpha, 4),
                       "target_passes_per_token": round(st.target_passes_per_token, 4),
                       "tokens_per_s": round(st.tokens_per_s, 2), "baseline_tokens_per_s": round(bst.tokens_per_s, 2),
                       "first_div": "", "gap": "", "ref_rank": "", "spec_rank": "", "ref_text": "", "spec_text": ""}
                if out != ref:
                    d = divergence(target, ids, ref, out)
                    k = d["first_div"]
                    row.update(d, ref_text=tok.decode(ref[max(0, k - 8):k + 4]),
                               spec_text=tok.decode(out[max(0, k - 8):k + 4]))
                rows.append(row)
                tag = "MATCH" if row["match"] else f"DIFF@{row['first_div']} gap={row['gap']} ranks={row['ref_rank']}/{row['spec_rank']}"
                print(f"{task:4s} p{pid} gamma={g}  alpha={row['alpha']:.3f}  {row['tokens_per_s']:6.2f} tok/s "
                      f"(base {row['baseline_tokens_per_s']:.2f})  {tag}", flush=True)
            if pid == 0:
                print(f"  sample [{task}]: {tok.decode(ref, skip_special_tokens=True)[:160]!r}", flush=True)

    with open("results/phase2.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = {"dtype": a.dtype, "gammas": gammas, "runs": len(rows), "matches": sum(r["match"] for r in rows),
               "near_tie_threshold": NEAR_TIE, "alpha_by_task": {}, "mismatches": []}
    for task in ("code", "math", "chat"):
        rs = [r for r in rows if r["task"] == task]
        summary["alpha_by_task"][task] = round(sum(r["alpha"] for r in rs) / len(rs), 4)
    for r in rows:
        if not r["match"]:
            numerics = r["gap"] is not None and r["gap"] <= NEAR_TIE and {r["ref_rank"], r["spec_rank"]} == {0, 1}
            summary["mismatches"].append({k: r[k] for k in ("task", "prompt_id", "gamma", "first_div", "gap",
                                                            "ref_rank", "spec_rank")} | {"near_tie": numerics})
    summary["unexplained"] = sum(not m["near_tie"] for m in summary["mismatches"])
    with open("results/phase2.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{summary['matches']}/{summary['runs']} greedy runs match baseline; "
          f"{len(summary['mismatches'])} mismatches, {summary['unexplained']} not explained as near-ties")
    print("alpha by task:", summary["alpha_by_task"])


if __name__ == "__main__":
    main()
