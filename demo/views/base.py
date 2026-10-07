"""Page 1: the 3B model alone, streamed live."""
import streamlit as st

from common import LiveView, metric_row, models_or_stop, run_key

p = st.session_state.prompt
st.title("Base model: Qwen2.5-3B")
st.markdown("Ordinary decoding: **one full pass of the 3B model for every single token.** "
            "Each pass reads all 3 billion weights from GPU memory, which is why it is slow.")
st.markdown(f"> {p['text']}")
if st.button("▶ Generate", type="primary", key="go_base"):
    m = models_or_stop()
    from bench import encode_prompt
    from specdec import baseline_generate
    ids = encode_prompt(m["tok"], p["text"])
    stats_box, box = st.empty(), st.empty()
    view = LiveView(m["tok"], box, stats_box, False, "base")
    out, stats = baseline_generate(m["target"], ids, p["max_new"], p["temp"], m["eos"], p["seed"],
                                   on_round=view.hook)
    view.draw()
    st.session_state.base_run = {"key": run_key(p), "ids": out, "tps": stats.tokens_per_s,
                                 "seconds": stats.seconds, "passes": stats.target_passes,
                                 "ttft": stats.ttft_s, "html": view.html()}
    stats_box.empty()
run = st.session_state.get("base_run")
if run and run["key"] == run_key(p):
    if run["html"] and not st.session_state.get("go_base"):
        st.markdown(run["html"], unsafe_allow_html=True)
    metric_row([("Speed", f"{run['tps']:.1f} tok/s", "Generated tokens ÷ generation time (display time excluded)"),
                ("Time", f"{run['seconds']:.2f} s", None),
                ("Tokens", str(len(run["ids"])), None),
                ("3B forward passes", str(run["passes"]), "One per token")])
    st.success("Now open **Speculative decoding** and run the same prompt.")
