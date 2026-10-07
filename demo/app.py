"""Live presentation demo: base model vs speculative decoding, plus Phase 3 results.

  .venv\\Scripts\\python -m streamlit run demo\\app.py

Page 1 streams the 3B model alone; page 2 streams speculative decoding (0.5B drafts, 3B verifies)
with accepted guesses highlighted and compares against page 1's run of the same prompt; page 3
shows the measured results from results/results.csv. Timings exclude display time (on_round hook).
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CSS, TASK_NAMES  # noqa: E402  (also puts the project root on sys.path)
from prompts import PROMPTS  # noqa: E402
from winperf import disable_power_throttling  # noqa: E402

st.set_page_config(page_title="Speculative Decoding", page_icon="⚡", layout="wide")
disable_power_throttling()
st.markdown(CSS, unsafe_allow_html=True)


def prompt_controls():
    with st.sidebar:
        st.header("Prompt")
        task = st.selectbox("Task", list(PROMPTS), format_func=TASK_NAMES.get, key="task", index=1)
        options = PROMPTS[task] + ["Custom…"]
        choice = st.selectbox("Example", options, key=f"example_{task}")
        if choice == "Custom…":
            text = st.text_area("Your prompt", "Explain how a binary search works.", key="custom")
        else:
            text = choice
        max_new = st.slider("Max new tokens", 32, 256, 128, 32, key="max_new")
        temp = st.radio("Temperature", [0.0, 0.7], horizontal=True, key="temp",
                        format_func=lambda t: "0 (greedy)" if t == 0 else str(t))
        seed = st.number_input("Seed", 0, 10_000, 17, key="seed")
    return {"task": task, "text": text, "max_new": int(max_new), "temp": float(temp), "seed": int(seed)}



st.session_state.prompt = prompt_controls()
nav = st.navigation([
    st.Page("views/base.py", title="1 · Base model", icon="🐢", default=True),
    st.Page("views/speculative.py", title="2 · Speculative decoding", icon="⚡"),
    st.Page("views/results.py", title="3 · Results", icon="📊"),
])
nav.run()
