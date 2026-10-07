"""Shared pieces of the demo: model loading, live token view, constants."""
import html
import os
import sys
import time

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TASK_COLORS = {"code": "#2a78d6", "math": "#eb6834", "chat": "#1baf7a"}  # validated categorical slots 1-3
TASK_NAMES = {"code": "Code", "math": "Maths", "chat": "Chat"}
BEST_GAMMA = {"code": 1, "math": 2, "chat": 1}  # best fixed γ from the Phase 3 sweep
RESULTS = os.path.join(ROOT, "results")

CSS = """
<style>
.gen-box { white-space: pre-wrap; line-height: 1.7; font-size: 1.02rem; padding: 1rem 1.1rem;
           border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; min-height: 9rem; }
.tok-draft { background: rgba(27,175,122,0.28); border-bottom: 2px solid #1baf7a; border-radius: 3px; }
.tok-rej { color: #e34948; text-decoration: line-through; font-size: 0.85em; opacity: 0.85; }
.legend span { margin-right: 1.2rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading Qwen2.5-3B and Qwen2.5-0.5B onto the GPU…")
def load_models():
    from bench import DRAFT, TARGET, encode_prompt, eos_ids, load_pair
    from specdec import FixedGamma, baseline_generate, measure_cost_ratio, speculative_generate
    import torch
    tok, target, draft = load_pair(TARGET, DRAFT, "fp16")
    eos = eos_ids(target)
    w = encode_prompt(tok, "Hello")
    baseline_generate(target, w, 16, 0.0, eos)
    speculative_generate(target, draft, w, 16, FixedGamma(2), 0.0, eos)
    c = measure_cost_ratio(target, draft, target.device)["c"]
    return {"tok": tok, "target": target, "draft": draft, "eos": eos, "c": c,
            "gpu": torch.cuda.get_device_name(0)}


def models_or_stop():
    try:
        return load_models()
    except Exception as e:  # no GPU, out of memory, missing weights
        st.error(f"Could not load the models: {e}")
        st.info("The **Results** page works without the models.")
        st.stop()


class LiveView:
    """Renders tokens as they are committed. Tags: 'draft' (accepted guess), 'target', and rejected guesses."""

    def __init__(self, tok, box, stats_box, show_rejected: bool, mode: str):
        self.tok, self.box, self.stats_box = tok, box, stats_box
        self.show_rejected, self.mode = show_rejected, mode
        self.ids, self.kinds, self.pieces, self.rejected_before = [], [], [], {}
        self.t0 = time.perf_counter()
        self.display_s, self.last_draw, self.rounds, self.from_draft = 0.0, 0.0, 0, 0

    def hook(self, committed, n_draft, rejected):
        t = time.perf_counter()
        if rejected is not None and n_draft < len(committed):
            self.rejected_before[len(self.ids) + n_draft] = rejected
        for tid in committed:
            self.ids.append(tid)
            self.pieces.append(self.piece(len(self.ids) - 1))
        self.kinds.extend(["draft"] * n_draft + ["target"] * (len(committed) - n_draft))
        self.rounds += 1
        self.from_draft += n_draft
        if t - self.last_draw > 0.04:
            self.draw()
            self.last_draw = time.perf_counter()
        self.display_s += time.perf_counter() - t

    def piece(self, i: int) -> str:
        """Text added by token i. A character split across tokens decodes as U+FFFD until its last byte
        arrives, so text is held back until it is complete; the decode window keeps this O(1) per token."""
        done = getattr(self, "emitted", 0)
        lo = max(0, done - 8)
        new = self.tok.decode(self.ids[lo:i + 1], skip_special_tokens=True)
        if new.endswith("�"):
            return ""
        self.emitted = i + 1
        return new[len(self.tok.decode(self.ids[lo:done], skip_special_tokens=True)):]

    def html(self) -> str:
        parts, in_draft = [], False
        for i, (kind, piece) in enumerate(zip(self.kinds, self.pieces)):
            rej = self.show_rejected and i in self.rejected_before
            highlight = kind == "draft" and self.mode == "spec"
            if in_draft and (rej or not highlight):  # close a run of accepted guesses
                parts.append("</span>")
                in_draft = False
            if rej:
                r = self.tok.decode([self.rejected_before[i]], skip_special_tokens=True)
                parts.append(f'<span class="tok-rej" title="rejected guess">{html.escape(r).replace(chr(10), "↵")}</span>')
            if highlight and not in_draft:
                parts.append('<span class="tok-draft">')
                in_draft = True
            # one line of HTML: a blank line would end the HTML block and let markdown take over
            parts.append(html.escape(piece).replace(chr(10), "<br>"))
        if in_draft:
            parts.append("</span>")
        return '<div class="gen-box">' + "".join(parts) + "</div>"

    def draw(self):
        self.box.markdown(self.html(), unsafe_allow_html=True)
        gen_s = time.perf_counter() - self.t0 - self.display_s
        n = len(self.ids)
        line = f"**{n}** tokens · **{gen_s:.2f}s** · **{n / gen_s if gen_s > 0 else 0:.1f} tok/s**"
        if self.mode == "spec":
            line += f" · {self.from_draft} of {n} tokens came from the 0.5B draft · {self.rounds} passes of the 3B"
        self.stats_box.markdown(line)


def run_key(p):
    return (p["text"], p["max_new"], p["temp"], p["seed"])


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_) in zip(cols, items):
        col.metric(label, value, help=help_)
