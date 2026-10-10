"""Page 3: measured results (Phase 3 sweep, Phase 4-5b comparisons) from results/."""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import RESULTS, TASK_COLORS, TASK_NAMES


@st.cache_data
def load_results():
    summ = pd.read_csv(os.path.join(RESULTS, "phase3_summary.csv"))
    rows = pd.read_csv(os.path.join(RESULTS, "phase3.csv"))
    return summ, rows


def styled(fig, title, ytitle, xtitle):
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=90, t=50, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18), yaxis_title=ytitle, xaxis_title=xtitle)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def page_results():
    st.title("Results")
    st.markdown("RTX 4060 Laptop GPU · fp16 · batch size 1 · 128 new tokens · 6 prompts per task · "
                "warm-up + 3 repeats per measurement. Every number comes from `results/phase3.csv` (phase P3 of `results/results.csv`).")
    summ, rows = load_results()
    fixed = summ[summ["method"].str.startswith("fixed")].copy()
    fixed["gamma"] = fixed["gamma"].astype(int)

    best = fixed.loc[fixed.groupby(["task", "temperature"])["measured_speedup"].idxmax()]
    cols = st.columns(4)
    for col, task in zip(cols, ["math", "code", "chat"]):
        b = best[(best.task == task) & (best.temperature == 0.0)].iloc[0]
        col.metric(f"{TASK_NAMES[task]} speed-up", f"{b.measured_speedup:.2f}×")
        col.caption(f"best γ = {b.gamma} · {b.alpha:.0%} of guesses accepted")
    greedy = rows[(rows.temperature == 0) & (rows.method != "baseline")]
    match = (greedy.matches_baseline.astype(str) == "True").sum()
    cols[3].metric("Greedy output identical", f"{match}/{len(greedy)}")
    cols[3].caption("the rest: fp16 near-ties between two equally likely words")

    temp = st.radio("Temperature", [0.0, 0.7], horizontal=True, key="res_temp",
                    format_func=lambda t: "0 (greedy)" if t == 0 else str(t))
    left, right = st.columns(2)

    # speed-up vs γ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 8], y=[1, 1], mode="lines", name="base model (1×)", hoverinfo="skip",
                             line=dict(color="#8a8984", width=1.5, dash="dash")))
    for task in ["code", "math", "chat"]:
        d = fixed[(fixed.task == task) & (fixed.temperature == temp)].sort_values("gamma")
        fig.add_trace(go.Scatter(
            x=d.gamma, y=d.measured_speedup, name=TASK_NAMES[task], mode="lines+markers",
            line=dict(color=TASK_COLORS[task], width=2), marker=dict(size=8),
            error_y=dict(type="data", array=d.speedup_sd_across_prompts, thickness=1, width=3),
            customdata=d[["alpha", "tokens_per_s"]],
            hovertemplate="γ=%{x}: %{y:.2f}× · α %{customdata[0]:.0%} · %{customdata[1]:.1f} tok/s<extra>%{fullData.name}</extra>"))
        last = d.iloc[-1]
        fig.add_annotation(x=last.gamma, y=last.measured_speedup, text=TASK_NAMES[task], showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=12))
    left.plotly_chart(styled(fig, "Speed-up vs draft length γ", "speed-up (×)", "γ (tokens guessed per round)"),
                      use_container_width=True)

    # acceptance by draft position (γ = 8 rows)
    fig = go.Figure()
    for task in ["code", "math", "chat"]:
        r = rows[(rows.task == task) & (rows.temperature == temp) & (rows.method == "fixed:8")]
        tested = [0] * 8
        accepted = [0] * 8
        for t_str, a_str in zip(r.pos_tested, r.pos_accepted):
            for i, (t_, a_) in enumerate(zip(str(t_str).split(";"), str(a_str).split(";"))):
                tested[i] += int(t_)
                accepted[i] += int(a_)
        rate = [a / t if t else None for a, t in zip(accepted, tested)]
        fig.add_trace(go.Scatter(x=list(range(1, 9)), y=rate, name=TASK_NAMES[task], mode="lines+markers",
                                 line=dict(color=TASK_COLORS[task], width=2), marker=dict(size=8),
                                 customdata=tested,
                                 hovertemplate="position %{x}: %{y:.0%} kept (%{customdata} checked)"
                                               "<extra>%{fullData.name}</extra>"))
        fig.add_annotation(x=8, y=rate[-1], text=TASK_NAMES[task], showarrow=False, xanchor="left", xshift=8)
    fig.update_yaxes(tickformat=".0%", range=[0.5, 1.02])
    right.plotly_chart(styled(fig, "How often each guess is accepted (γ = 8)", "acceptance rate",
                              "position of the guess in the draft"), use_container_width=True)

    # predicted vs measured
    left2, right2 = st.columns(2)
    fig = go.Figure()
    lo = min(fixed.predicted_speedup.min(), fixed.measured_speedup.min()) - 0.05
    hi = max(fixed.predicted_speedup.max(), fixed.measured_speedup.max()) + 0.05
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="prediction = measurement",
                             line=dict(color="#8a8984", width=1.5, dash="dash"), hoverinfo="skip"))
    for task in ["code", "math", "chat"]:
        d = fixed[(fixed.task == task) & (fixed.temperature == temp)]
        fig.add_trace(go.Scatter(x=d.predicted_speedup, y=d.measured_speedup, name=TASK_NAMES[task],
                                 mode="markers", marker=dict(size=10, color=TASK_COLORS[task],
                                                             line=dict(width=2, color="rgba(255,255,255,0.9)")),
                                 customdata=d.gamma,
                                 hovertemplate="γ=%{customdata}: predicted %{x:.2f}×, measured %{y:.2f}×"
                                               "<extra>%{fullData.name}</extra>"))
    fig = styled(fig, "Formula S = E / (1 + γc) vs measurement", "measured speed-up (×)", "predicted speed-up (×)")
    fig.update_layout(hovermode="closest")
    left2.plotly_chart(fig, use_container_width=True)

    with right2:
        st.markdown("#### What the results say")
        st.markdown("""
- **Maths benefits most** (α ≈ 95%): its step-by-step text is predictable, so the small model guesses well.
- **Chat benefits least** (α ≈ 72%): open-ended wording is harder to guess, and long drafts make it *slower*.
- **Short drafts win** (γ = 1–2) because the 0.5B model is not that cheap on this GPU:
  each of its steps costs ~70% of a 3B step (c ≈ 0.7), so wasted guesses are expensive.
- **The formula predicts the best γ correctly**; measured speed-ups are slightly higher than predicted.
- **Quality is unchanged**: greedy output was identical in 99 of 108 runs; the rest were fp16 near-ties.
""")

    with st.expander("Data table (results/phase3_summary.csv)"):
        st.dataframe(summ[summ.temperature == temp], hide_index=True, use_container_width=True)


METHOD_NAMES = {"adaptive": "Adaptive γ (ours)", "adaedl:0.7": "AdaEDL"}


def load_json(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def verdict_metric(col, label, comp):
    lo, hi = comp["ci95"]
    col.metric(label, f"{comp['mean_diff']:+.3f}")
    col.caption(f"95% CI [{lo:+.3f}, {hi:+.3f}] · wins {comp['wins']}/{comp['n']} · **{comp['verdict'].replace(' vs', '')}**")


def method_chart(summary, temp, title, fixed_order):
    fig = go.Figure()
    methods = fixed_order + ["adaptive", "adaedl:0.7"]
    labels = [METHOD_NAMES.get(m, m.replace("fixed:", "fixed γ=")) for m in methods]
    for k, task in enumerate(["code", "math", "chat"]):  # methods are categories: dots, offset per task, no lines
        ys = [summary["table"].get(f"{task} T={temp} {m}", {}).get("speedup") for m in methods]
        fig.add_trace(go.Scatter(x=[i + (k - 1) * 0.18 for i in range(len(methods))], y=ys, customdata=labels,
                                 name=TASK_NAMES[task], mode="markers",
                                 marker=dict(size=11, color=TASK_COLORS[task], line=dict(width=1.5, color="white")),
                                 hovertemplate="%{customdata}: %{y:.2f}×<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(methods))), ticktext=labels)
    fig.add_trace(go.Scatter(x=[-0.5, len(methods) - 0.5], y=[1, 1], mode="lines", name="base model (1×)",
                             hoverinfo="skip", line=dict(color="#8a8984", width=1.5, dash="dash")))
    return styled(fig, title, "speed-up (×)", "")


def page_comparison():
    st.divider()
    st.header("Does adaptive γ beat the alternatives? (the O2 test)")
    st.markdown("Every method runs on the same 36 cells (3 tasks × 6 prompts × 2 temperatures), interleaved, "
                "with the decision rule committed to git **before** each run. A difference counts only if its "
                "95% confidence interval excludes 0.")
    p5, p5b = load_json("phase5_summary.json"), load_json("phase5b_summary.json")
    temp = st.session_state.get("res_temp", 0.0)
    if p5:
        st.subheader("With the ordinary draft model (c ≈ 0.70)")
        a, b = st.columns(2)
        verdict_metric(a, "vs best fixed γ", p5["comparisons"]["best_fixed_a_priori"]["all"])
        verdict_metric(b, "vs AdaEDL", p5["comparisons"]["adaedl"]["all"])
        st.plotly_chart(method_chart(p5, temp, f"Speed-up by method, T = {temp:g}", ["fixed:1", "fixed:2"]),
                        use_container_width=True)
        st.caption("The 0.5B costs ~70% of a 3B step, so γ = 1–3 all give about the same speed: there is nothing "
                   "for adaptation to gain. AdaEDL drafts too long and discards a draft pass at every early stop.")
    if p5b:
        st.subheader(f"With the CUDA-graph draft (c ≈ {p5b['c_per_cell']['mean']:.2f})")
        a, b = st.columns(2)
        verdict_metric(a, "vs best fixed γ (picked after the fact)", p5b["comparisons"]["best_fixed_per_task_after_the_fact"]["all"])
        verdict_metric(b, "vs AdaEDL", p5b["comparisons"]["adaedl"]["all"])
        st.plotly_chart(method_chart(p5b, temp, f"Speed-up by method, T = {temp:g}",
                                     ["fixed:2", "fixed:4", "fixed:6", "fixed:8"]), use_container_width=True)
        st.caption("Capturing the draft's decode step as one CUDA graph removes the kernel-launch overhead: "
                   "the 0.5B now costs ~12% of a 3B step, speed-ups reach 2–5×, and the best γ differs by task.")


page_results()
page_comparison()
