"""Where results are written. Set SPECDEC_RESULTS to keep runs on different machines apart.

  SPECDEC_RESULTS=results/T4 python phase1.py     # Kaggle / Colab T4
  python phase1.py                                # default: results/
"""
import os

RESULTS = os.environ.get("SPECDEC_RESULTS", "results")


def R(name: str) -> str:
    """Path of `name` inside the results directory (created on first use)."""
    path = os.path.join(RESULTS, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path
