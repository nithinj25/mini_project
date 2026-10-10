"""Benchmark harness (SPEC §9). One CSV row per (task, prompt, temperature, method).

  python bench.py --tiny                                  # smoke test: random toy models (c ≈ 1)
  python bench.py --methods baseline,fixed:4,adaptive     # real Qwen2.5 pair
"""
from __future__ import annotations

import argparse
import csv
import os
import threading
import time

import torch

from prompts import PROMPTS
from winperf import disable_power_throttling
from specdec import (AdaEDL, FixedGamma, GammaController, assert_compatible, baseline_generate,
                     mean_std, measure_cost_ratio, predicted_speedup, speculative_generate)

TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

FIELDS = ["task", "prompt_id", "temperature", "method", "gamma_mean", "repeats", "new_tokens", "seconds",
          "tokens_per_s", "tokens_per_s_std", "ttft_ms", "ttft_ms_std", "target_passes_per_token", "alpha",
          "peak_mem_gb", "gpu_util", "matches_baseline", "c", "predicted_speedup", "measured_speedup",
          "pos_tested", "pos_accepted", "dtype", "gpu"]


# --------------------------------------------------------------------------- loading

def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible; this project runs on the GPU.")
    return torch.device("cuda")


def load_pair(target_name=TARGET, draft_name=DRAFT, dtype="fp16", device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(target_name)
    draft_tok = AutoTokenizer.from_pretrained(draft_name)
    kw = dict(dtype=DTYPES[dtype], device_map={"": device.index or 0})
    target = AutoModelForCausalLM.from_pretrained(target_name, **kw).eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_name, **kw).eval()
    assert_compatible(tok, draft_tok, target, draft)
    return tok, target, draft


def tiny_pair(device):
    """Two random toy Qwen2 models of the same size: c ≈ 1, so speculation must lose."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    torch.manual_seed(0)
    target = Qwen2ForCausalLM(cfg).to(device).eval()
    torch.manual_seed(1)
    draft = Qwen2ForCausalLM(cfg).to(device).eval()
    return target, draft


def encode_prompt(tok, text: str) -> list[int]:
    chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   tokenize=False)
    return tok(chat, add_special_tokens=False)["input_ids"]


def eos_ids(model):
    return model.generation_config.eos_token_id if model.generation_config is not None else None


def arch(model) -> dict:
    cfg = model.config
    return {"layers": cfg.num_hidden_layers, "kv_heads": cfg.num_key_value_heads,
            "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
            "vocab_rows": model.get_output_embeddings().weight.shape[0]}


# --------------------------------------------------------------------------- measurement

class GpuMonitor:
    """Samples NVML busy-% in a thread. Busy-percentage, not efficiency (SPEC §9)."""

    def __init__(self, interval=0.05):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._h, self._nv = pynvml.nvmlDeviceGetHandleByIndex(0), pynvml
        except Exception:
            self._h = None

    def __enter__(self):
        if self._h is not None:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self._nv.nvmlDeviceGetUtilizationRates(self._h).gpu)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._h is not None:
            self._t.join()

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")


MAX_DRAFT = 8  # L for the adaptive methods; set by --max-draft


def make_controller(method: str, c: float, temp: float = 0.0):
    """fixed:γ | adaptive (cost-ratio-aware, GammaController) | adaedl[:λ0] (entropy early stopping, AdaEDL)."""
    if method.startswith("fixed:"):
        return FixedGamma(int(method.split(":")[1]))
    if method == "adaptive":
        return GammaController(c=c, gamma_max=MAX_DRAFT)
    if method == "adaedl" or method.startswith("adaedl:"):
        lam0 = float(method.split(":")[1]) if ":" in method else 0.7
        return AdaEDL(max_draft=MAX_DRAFT, lam0=lam0, temperature=temp)
    raise ValueError(f"unknown method {method!r}")


def run_method(method, target, draft, ids, max_new, temp, eos, seed, c):
    if method == "baseline":
        return baseline_generate(target, ids, max_new, temp, eos, seed)
    return speculative_generate(target, draft, ids, max_new, make_controller(method, c, temp), temp, eos, seed)


def run_cell(method, target, draft, ids, max_new, temp, eos, seed, c, repeats):
    """Warm-up, then `repeats` timed runs with the same seed. Returns (output, list[Stats], gpu_util, peak_gb)."""
    run_method(method, target, draft, ids, max_new, temp, eos, seed, c)
    torch.cuda.reset_peak_memory_stats()
    runs = []
    with GpuMonitor() as mon:
        for _ in range(repeats):
            out, st = run_method(method, target, draft, ids, max_new, temp, eos, seed, c)
            runs.append(st)
    return out, runs, mon.mean, torch.cuda.max_memory_allocated() / 2**30


def row_for(task, pid, temp, method, out, runs, util, peak, ref_out, base_tps, c, dtype, gpu):
    tps, tps_sd = mean_std(r.tokens_per_s for r in runs)
    ttft, ttft_sd = mean_std(1e3 * r.ttft_s for r in runs)
    st = runs[-1]
    spec = method != "baseline"
    g = st.mean_gamma if spec else 0
    return {
        "task": task, "prompt_id": pid, "temperature": temp, "method": method,
        "gamma_mean": round(g, 3) if spec else "", "repeats": len(runs), "new_tokens": st.new_tokens,
        "seconds": round(sum(r.seconds for r in runs) / len(runs), 4),
        "tokens_per_s": round(tps, 3), "tokens_per_s_std": round(tps_sd, 3),
        "ttft_ms": round(ttft, 3), "ttft_ms_std": round(ttft_sd, 3),
        "target_passes_per_token": round(st.target_passes_per_token, 4),
        "alpha": round(st.alpha, 4) if spec else "", "peak_mem_gb": round(peak, 3), "gpu_util": round(util, 1),
        # at T>0 the random streams differ between methods, so equality is only defined for greedy
        "matches_baseline": (out == ref_out) if temp == 0 else "",
        "c": round(c, 4), "predicted_speedup": round(predicted_speedup(st.alpha, g, c), 4) if spec else 1.0,
        "measured_speedup": round(tps / base_tps, 4),
        "pos_tested": ";".join(map(str, st.pos_tested)), "pos_accepted": ";".join(map(str, st.pos_accepted)),
        "dtype": dtype, "gpu": gpu,
    }


def append_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="smoke test on random toy models")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--methods", default="baseline,fixed:1,fixed:2,fixed:3,fixed:4,fixed:6,fixed:8",
                    help="comma list of baseline, fixed:γ, adaptive, adaedl[:λ0]")
    ap.add_argument("--max-draft", type=int, default=8, help="L, max draft length for adaptive/adaedl")
    ap.add_argument("--tasks", default="code,math,chat")
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--prompts", type=int, default=6, help="prompts per task")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--c", type=float, default=None, help="cost ratio; measured if omitted")
    ap.add_argument("--out", default="results/phase3.csv")
    a = ap.parse_args()
    global MAX_DRAFT
    MAX_DRAFT = a.max_draft

    disable_power_throttling()
    device = pick_device()
    gpu = torch.cuda.get_device_name(device)
    methods = [m for m in a.methods.split(",") if m != "baseline"]
    temps = [float(t) for t in a.temps.split(",")]

    if a.tiny:
        target, draft = tiny_pair(device)
        g = torch.Generator().manual_seed(0)
        cells = [("tiny", 0, torch.randint(0, 1000, (32,), generator=g).tolist())]
        eos, a.out, a.repeats, a.max_new, dtype = None, "results/tiny.csv", 2, 48, "fp32"
        methods, temps = ["fixed:4", "adaptive", "adaedl"], [0.0]
    else:
        tok, target, draft = load_pair(a.target, a.draft, a.dtype, device)
        eos, dtype = eos_ids(target), a.dtype
        cells = [(task, i, encode_prompt(tok, PROMPTS[task][i]))
                 for task in a.tasks.split(",") for i in range(a.prompts)]

    c = a.c if a.c is not None else measure_cost_ratio(target, draft, device)["c"]
    print(f"gpu={gpu} dtype={dtype} c={c:.4f} -> {a.out}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # a cell's rows are appended together, so any row for (task, prompt, temp) means it is complete
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            done = {(r["task"], int(r["prompt_id"]), float(r["temperature"])) for r in csv.DictReader(f)}

    for task, pid, ids in cells:
        for temp in temps:
            if (task, pid, temp) in done:
                print(f"skip {task} p{pid} T={temp} (already in {a.out})")
                continue
            seed = 1000 * pid + 17
            ref, runs, util, peak = run_cell("baseline", target, draft, ids, a.max_new, temp, eos, seed, c, a.repeats)
            base_tps = mean_std(r.tokens_per_s for r in runs)[0]
            rows = [row_for(task, pid, temp, "baseline", ref, runs, util, peak, ref, base_tps, c, dtype, gpu)]
            for m in methods:
                out, runs, util, peak = run_cell(m, target, draft, ids, a.max_new, temp, eos, seed, c, a.repeats)
                rows.append(row_for(task, pid, temp, m, out, runs, util, peak, ref, base_tps, c, dtype, gpu))
            append_rows(a.out, rows)
            for r in rows:
                print(f"{task:5s} p{pid} T={temp:<4} {r['method']:9s} {r['tokens_per_s']:8.1f} tok/s "
                      f"x{r['measured_speedup']:.2f} (pred {r['predicted_speedup']}) alpha={r['alpha']} "
                      f"match={r['matches_baseline']}")


if __name__ == "__main__":
    main()
