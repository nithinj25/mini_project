"""One command for a fresh GPU machine (Kaggle, Colab, a cloud VM): tests, phases, then the report artefacts.

  SPECDEC_RESULTS=results/T4 python run_all.py                       # every step
  SPECDEC_RESULTS=results/T4 python run_all.py --steps tests,p1,p5b  # a subset
  python run_all.py --list

Each step's output goes to <results>/logs/<step>.log (and its key lines to the screen). A step that
finishes is recorded in <results>/logs/done.txt and skipped next time unless --force; p3/p5/p5b also
resume cell by cell, so an interrupted session can simply be re-run. A failing step stops the run.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from paths import R, RESULTS

PY = sys.executable
# name: (commands, rough minutes on a T4 / RTX 4060-class GPU, what it does)
STEPS = {
    "tests":   ([["test_specdec.py"]], 7, "13 correctness tests (2 need CUDA)"),
    "p1":      ([["phase1.py"]], 10, "baseline speed, cost ratio c, predictions"),
    "p2":      ([["phase2.py"]], 15, "greedy output equals the base model"),
    "p3":      ([["bench.py"], ["summarize.py"]], 150, "fixed-γ sweep (eager draft)"),
    "p4":      ([["phase4.py"]], 12, "AdaEDL behaviour vs entropy"),
    "p5":      ([["phase5.py"]], 70, "O2 test, eager draft (pre-registered)"),
    "p5b":     ([["phase5b.py"]], 80, "O2 test, CUDA-graph draft (pre-registered)"),
    "results": ([["make_results.py"]], 1, "results.csv with every row"),
    "plots":   ([["plots.py"]], 1, "figures F1-F5"),
}
KEY = ("ALL TESTS PASSED", "FAILED", "Error", "Traceback", "c = ", "baseline ", "greedy runs match", "O2",
       "best gamma", "gpu=", "verdict", "wrote ", "rows", "skipped")


def environment() -> dict:
    import torch
    import transformers
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                   vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    return env


def run_step(name: str, cmds, log_path: str) -> bool:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "a", encoding="utf-8") as log:
        for cmd in cmds:
            log.write(f"\n$ python {' '.join(cmd)}\n")
            proc = subprocess.Popen([PY, "-W", "ignore", *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", env=env)
            for line in proc.stdout:
                log.write(line)
                if any(k in line for k in KEY) and "Loading weights" not in line:
                    print(f"  [{name}] {line.rstrip()[:160]}", flush=True)
            if proc.wait() != 0:
                return False
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS), help="comma list from: " + ", ".join(STEPS))
    ap.add_argument("--force", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--zip", default=None, help="write a zip of the results directory here at the end")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, (_, mins, what) in STEPS.items():
            print(f"{k:8s} ~{mins:>3d} min  {what}")
        return
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        sys.exit(f"unknown steps: {unknown}; choose from {list(STEPS)}")

    env = environment()
    print(f"results -> {RESULTS}   {env.get('gpu', 'NO GPU')}   torch {env['torch']}   transformers {env['transformers']}")
    if not env["cuda_available"]:
        sys.exit("No CUDA GPU visible. On Kaggle: Settings -> Accelerator -> GPU T4.")
    with open(R("logs/environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    done_path = R("logs/done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    total = sum(STEPS[s][1] for s in steps if a.force or s not in done)
    print(f"plan: {', '.join(steps)}  (~{total} min of GPU time, rough)")

    timings = {}
    for s in steps:
        if s in done and not a.force:
            print(f"- {s}: already done, skipped (use --force to re-run)")
            continue
        cmds, mins, what = STEPS[s]
        print(f"- {s}: {what} (~{mins} min)  log: {R('logs/' + s + '.log')}", flush=True)
        t0 = time.time()
        ok = run_step(s, cmds, R(f"logs/{s}.log"))
        timings[s] = round((time.time() - t0) / 60, 1)
        if not ok:
            sys.exit(f"step {s} FAILED after {timings[s]} min; see {R('logs/' + s + '.log')}")
        with open(done_path, "a") as f:
            f.write(s + "\n")
        print(f"  {s} done in {timings[s]} min", flush=True)

    print("all requested steps finished:", timings)
    if a.zip:
        base = a.zip[:-4] if a.zip.endswith(".zip") else a.zip
        print("zip ->", shutil.make_archive(base, "zip", RESULTS))


if __name__ == "__main__":
    main()
