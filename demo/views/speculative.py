"""Page 2: speculative decoding, streamed live with accepted guesses highlighted."""
import streamlit as st

from common import BEST_GAMMA, LiveView, load_models, metric_row, models_or_stop, run_key
from prompts import PROMPTS


def page_spec():
    p = st.session_state.prompt
    st.title("Speculative decoding: 0.5B drafts, 3B verifies")
    with st.expander("How it works", expanded=False):
        st.markdown("""
1. **Draft.** The small Qwen2.5-0.5B model guesses the next **γ** tokens, one at a time. It is cheap.
2. **Verify.** The big Qwen2.5-3B model checks **all γ guesses in a single pass.** Checking γ tokens costs
   about the same as generating one, because the bottleneck is reading the weights, not the arithmetic.
3. **Accept or reject.** Each guess is kept with probability min(1, p/q), where p and q are the big and
   small models' probabilities. At the first rejection the big model supplies the correct token itself.
   If every guess is kept, it adds one bonus token for free.
4. **Repeat.** Each round yields between 1 and γ+1 tokens for a single 3B pass.

**Lossless:** the rejection rule makes the output follow exactly the 3B model's distribution
(Leviathan et al. 2023; Chen et al. 2023). In greedy mode the text is identical to the base model's.
""")
    default_g = BEST_GAMMA[p["task"]] if p["text"] in PROMPTS[p["task"]] else 2
    g = st.slider("γ: tokens the draft guesses per round", 1, 8, default_g, key=f"gamma_{p['task']}",
                  help="Default = best fixed γ for this task in our Phase 3 sweep")
    show_rej = st.toggle("Show rejected guesses", value=False)
    st.markdown('<div class="legend"><span><span class="tok-draft">text</span> guessed by the 0.5B, '
                'accepted by the 3B</span><span>text written by the 3B itself</span>'
                + ('<span><span class="tok-rej">text</span> rejected guess</span>' if show_rej else "")
                + "</div>", unsafe_allow_html=True)
    st.markdown(f"> {p['text']}")

    if st.button("▶ Generate", type="primary", key="go_spec"):
        m = models_or_stop()
        from bench import encode_prompt
        from specdec import FixedGamma, speculative_generate
        ids = encode_prompt(m["tok"], p["text"])
        stats_box, box = st.empty(), st.empty()
        view = LiveView(m["tok"], box, stats_box, show_rej, "spec")
        out, stats = speculative_generate(m["target"], m["draft"], ids, p["max_new"], FixedGamma(g), p["temp"],
                                          m["eos"], p["seed"], on_round=view.hook)
        view.draw()
        stats_box.empty()
        st.session_state.spec_run = {"key": run_key(p), "gamma": g, "ids": out, "tps": stats.tokens_per_s,
                                     "seconds": stats.seconds, "passes": stats.target_passes,
                                     "alpha": stats.alpha, "from_draft": view.from_draft, "c": m["c"],
                                     "html": view.html()}

    run = st.session_state.get("spec_run")
    if not (run and run["key"] == run_key(p) and run["gamma"] == g):
        return
    if not st.session_state.get("go_spec"):
        st.markdown(run["html"], unsafe_allow_html=True)
    metric_row([("Speed", f"{run['tps']:.1f} tok/s", "Display time excluded"),
                ("Time", f"{run['seconds']:.2f} s", None),
                ("Acceptance rate α", f"{run['alpha']:.0%}", "Guesses kept ÷ guesses checked"),
                ("3B forward passes", f"{run['passes']}", f"for {len(run['ids'])} tokens")])

    base = st.session_state.get("base_run")
    st.subheader("Compared with the base model")
    if not (base and base["key"] == run_key(p)):
        st.info("Run this same prompt on the **Base model** page to see the speed-up.")
        if st.button("Run the base model now (no display)"):
            m = models_or_stop()
            from bench import encode_prompt
            from specdec import baseline_generate
            with st.spinner("Running the 3B model alone…"):
                out, stats = baseline_generate(m["target"], encode_prompt(m["tok"], p["text"]), p["max_new"],
                                               p["temp"], m["eos"], p["seed"])
            st.session_state.base_run = {"key": run_key(p), "ids": out, "tps": stats.tokens_per_s,
                                         "seconds": stats.seconds, "passes": stats.target_passes,
                                         "ttft": stats.ttft_s, "html": ""}
            st.rerun()
        return
    speedup = run["tps"] / base["tps"]
    saved = 1 - run["passes"] / base["passes"]
    metric_row([("Speed-up", f"{speedup:.2f}×", f"{run['tps']:.1f} vs {base['tps']:.1f} tok/s"),
                ("Time", f"{run['seconds']:.2f} s", f"base model: {base['seconds']:.2f} s"),
                ("3B passes saved", f"{saved:.0%}", f"{run['passes']} vs {base['passes']}")])
    if p["temp"] == 0:
        if run["ids"] == base["ids"]:
            st.success("✓ **Identical output.** Speculative decoding produced exactly the base model's text.")
        else:
            k = next((i for i, (a, b) in enumerate(zip(run["ids"], base["ids"])) if a != b), None)
            st.warning(f"Outputs differ from token {k}. In fp16 this happens when the 3B model's top two "
                       "choices are near-tied (Phase 2 measured logit gaps of 0.016–0.031); both are valid.")
    else:
        st.info("At temperature 0.7 both methods sample, so the texts differ run to run, but speculative "
                "decoding samples from exactly the same distribution as the base model.")
    st.caption(f"Measured on {load_models()['gpu']} · fp16 · c (draft ÷ target step time) = {run['c']:.2f}. "
               "A single run on a laptop is noisy; the Results page has the averaged numbers.")


page_spec()
