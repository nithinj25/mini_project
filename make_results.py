"""Build results/results.csv: every measured (phase, task, prompt, temperature, method) row in one file.

SPEC §11 asks that results.csv cover every (task × temperature × method) cell and that the report trace
each number to a row in it. Sources (each also kept on its own):
  P3   results/phase3.csv    fixed-γ sweep, eager draft (methods run back to back)
  P5   results/phase5.csv    O2 comparison, eager draft (interleaved, paired by repeat)
  P5b  results/phase5b.csv   O2 comparison, CUDA-graph draft (interleaved, paired by repeat)

A row's key is (phase, task, prompt_id, temperature, method). Look one up with:
  python make_results.py --show P5 math 2 0.0 adaptive
"""
import argparse
import csv

SOURCES = [("P3", "results/phase3.csv"), ("P5", "results/phase5.csv"), ("P5b", "results/phase5b.csv")]
OUT = "results/results.csv"
KEY = ["phase", "task", "prompt_id", "temperature", "method"]


def build():
    rows, fields = [], list(KEY) + ["draft"]
    for phase, path in SOURCES:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            r["phase"] = phase
            r["draft"] = "cuda-graph static cache" if phase == "P5b" else "eager"
            r["temperature"] = str(float(r["temperature"]))
            rows.append(r)
            fields += [k for k in r if k not in fields]
    keys = [tuple(r[k] for k in KEY) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (phase, task, prompt, temperature, method) rows"
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT}: {len(rows)} rows")
    for phase, _ in SOURCES:
        pr = [r for r in rows if r["phase"] == phase]
        cells = {(r["task"], r["temperature"], r["method"]) for r in pr}
        print(f"  {phase}: {len(pr)} rows, {len(cells)} (task × temperature × method) cells, "
              f"methods {sorted({r['method'] for r in pr})}")


def show(key):
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if [r[k] for k in KEY] == key[:2] + [key[2], str(float(key[3])), key[4]]:
            for k, v in r.items():
                print(f"{k:26s} {v}")
            return
    print("no such row")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs=5, metavar=("PHASE", "TASK", "PROMPT", "TEMP", "METHOD"))
    a = ap.parse_args()
    show(a.show) if a.show else build()
