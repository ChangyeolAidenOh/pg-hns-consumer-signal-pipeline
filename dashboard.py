import ast
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# P&G Blue color scheme
PG_BLUE    = "#003087"
PG_LIGHT   = "#0066CC"
PG_SKY     = "#4A90D9"
CHURN_RED  = "#CC0000"
POSITIVE   = "#00A86B"
NEUTRAL    = "#4A90D9"
WARNING    = "#FF8C00"

st.set_page_config(
    page_title="HNS Consumer Signal Detection Pipeline",
    page_icon="🔬",
    layout="wide"
)

@st.cache_data
def load_data():
    # Layer 1
    causal    = pd.read_csv("voc_pipeline/data/processed/hns_causal_signals.csv")
    temporal  = pd.read_csv("voc_pipeline/data/processed/hns_temporal_signals.csv")
    lda       = pd.read_csv("voc_pipeline/data/processed/hns_lda_results.csv")
    bertopic  = pd.read_csv("voc_pipeline/data/processed/hns_bertopic_results.csv")
    consensus = pd.read_csv("voc_pipeline/data/processed/hns_lda_bertopic_consensus.csv")
    # Layer 2
    trend     = pd.read_csv("trend_pipeline/data/processed/trend_features.csv",
                             index_col=0, parse_dates=True)
    trend     = trend.fillna(0)
    trend["antitro_ratio"]    = trend["안티트로샴푸"] / trend["헤드앤숄더샴푸"].replace(0, 1)
    trend["hns_momentum"]     = trend["헤드앤숄더샴푸"].pct_change(3)
    trend["category_momentum"] = trend["category_click"].pct_change(3)
    forecast  = pd.read_csv("trend_pipeline/data/processed/chronos_forecast.csv",
                             parse_dates=["date"])
    # Layer 3
    seg_prob  = pd.read_csv("switching_pipeline/data/switching_prob_regression.csv")
    voc_feat  = pd.read_csv("switching_pipeline/data/voc_features.csv")
    timeline  = pd.read_csv("switching_pipeline/data/timeline_analysis.csv")
    impl      = pd.read_csv("switching_pipeline/data/switching_implications.csv")
    return (causal, temporal, lda, bertopic, consensus,
            trend, forecast, seg_prob, voc_feat, timeline, impl)

(causal, temporal, lda, bertopic, consensus,
 trend, forecast, seg_prob, voc_feat, timeline, impl) = load_data()


def parse_list(val):
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []


def format_keywords(val):
    try:
        items = ast.literal_eval(str(val))
        return ", ".join(items) if isinstance(items, list) else str(val)
    except:
        return str(val)


st.title("Head & Shoulders Consumer Signal Detection Pipeline")
st.markdown("Aiden Changyeol Oh | 3-Layer NLP × Trend × ML Pipeline")
st.divider()

st.markdown("""
    <style>
    [data-testid="stDataFrame"] { margin-left: auto; margin-right: auto; }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "VoC Analysis",
    "Trend Analysis",
    "Switching Risk",
    "BERTopic"
])


## Tab 1 — Overview
with tab1:
    st.header("Project Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Collected", "2,714 docs")
    with col2:
        st.metric("After HNS Filter", f"{len(causal):,} docs")
    with col3:
        churn_rate = (causal["signal_type"] == "이탈위험").mean() * 100
        st.metric("Churn Risk Rate", f"{churn_rate:.1f}%")
    with col4:
        trend["antitro_ratio"] = (trend["안티트로샴푸"] /
                                   trend["헤드앤숄더샴푸"].replace(0, 1))
        st.metric("Antitro/HNS Ratio", f"{trend['antitro_ratio'].iloc[-1]:.3f}")
    with col5:
        active = seg_prob[seg_prob["segment"] == "Active Switcher"]["switching_probability"].values
        st.metric("Active Switcher P(switch)", f"{active[0]:.3f}" if len(active) > 0 else "N/A")

    st.divider()

    st.subheader("3-Layer Pipeline Architecture")
    st.code(
        "Layer 1 — VoC Pipeline\n"
        "  Collection   → Naver Blog/Cafe (1,328) + YouTube (1,386) = 2,714 docs\n"
        "  Preprocessing → kiwipiepy + user word (안티트로) + 4 modes\n"
        "  LDA          → bigram best coherence 0.6329 (cafearticle)\n"
        "  Causal Signal → churn risk 24.0% | competitor mention 101 docs\n"
        "  BERTopic     → 17 topics | 12 High Confidence (LDA × BERTopic)\n"
        "\n"
        "Layer 2 — Trend Pipeline\n"
        "  Data         → Naver DataLab 76 months (2020-01 ~ 2026-04) × 16 features\n"
        "  Chronos      → Amazon Chronos zero-shot 12-month forecast\n"
        "  Analysis     → Antitro reversal 2025-06 | ratio 1.327 | HNS -23.7%\n"
        "\n"
        "Layer 3 — Switching Pipeline\n"
        "  Segments     → Active Switcher / At-risk / Passive User / Loyal\n"
        "  Classifier   → Logistic Regression (CV accuracy 0.9300, C=1.0)\n"
        "  Trend adjust → Grid search optimal weights (w_antitro=0.50, w_hns=0.40)\n"
        "  Risk Score   → Brand switching risk 0.068 | Active Switcher P=0.405",
        language="text"
    )

    st.divider()

    st.subheader("Key Cross-Layer Findings")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div style='background-color:{PG_BLUE}; padding:16px; border-radius:8px;"
            f"color:white;'>"
            f"<b>Layer 1 × Layer 2</b><br><br>"
            f"Ingredient scrutiny (설페이트, 계면활성제) is a "
            f"YouTube-exclusive phenomenon. "
            f"Blog/Cafe consumers remain in result-focused exploration. "
            f"Medical frame (피부과, 항진균) spreads across all channels."
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div style='background-color:{PG_LIGHT}; padding:16px; border-radius:8px;"
            f"color:white;'>"
            f"<b>Layer 2 Signal</b><br><br>"
            f"Antitro reversed HNS in 6 months (2024-12 → 2025-06). "
            f"Acceleration peak Jan 2026 (+40.2). "
            f"Category click volume rebounded +14.6% in 2026 — "
            f"antitro is expanding the category, not just taking share."
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div style='background-color:{CHURN_RED}; padding:16px; border-radius:8px;"
            f"color:white;'>"
            f"<b>Layer 3 Risk</b><br><br>"
            f"At-risk segment (5.2%) shows highest ingredient/medical frame — "
            f"these are pre-switch consumers not yet comparing brands. "
            f"Critical window: before antitro ratio exceeds 1.5."
            f"</div>",
            unsafe_allow_html=True
        )


## Tab 2 — VoC Analysis
with tab2:
    st.header("VoC Analysis (Layer 1)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Signal Distribution by Source")
        source_signal = causal.groupby("source").apply(
            lambda x: pd.Series({
                "Churn Risk": (x["signal_type"] == "이탈위험").mean() * 100,
                "Positive":   (x["signal_type"] == "긍정").mean() * 100,
                "Neutral":    (x["signal_type"] == "중립").mean() * 100,
            })
        ).reset_index()
        source_signal_melted = source_signal.melt(
            id_vars="source", var_name="Signal", value_name="Rate (%)"
        )
        fig = px.bar(
            source_signal_melted, x="source", y="Rate (%)",
            color="Signal", barmode="stack",
            color_discrete_map={
                "Churn Risk": CHURN_RED,
                "Positive": POSITIVE,
                "Neutral": NEUTRAL
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Competitor Mention vs Overall Churn")
        overall = (causal["signal_type"] == "이탈위험").mean() * 100
        comp_churn = (
            causal[causal["competitor_mentioned"] == True]["signal_type"] == "이탈위험"
        ).mean() * 100
        fig = go.Figure(go.Bar(
            x=["Overall HNS", "Competitor-mention Docs"],
            y=[overall, comp_churn],
            marker_color=[PG_BLUE, CHURN_RED],
            text=[f"{overall:.1f}%", f"{comp_churn:.1f}%"],
            textposition="outside"
        ))
        fig.update_layout(yaxis_title="Churn Risk Rate (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Monthly Churn / Positive Signal Trend")
    temporal["month"] = temporal["month"].astype(str)
    temporal_recent = temporal[temporal["month"] >= "2025-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temporal_recent["month"], y=temporal_recent["churn_rate"] * 100,
        name="Churn Risk", line=dict(color=CHURN_RED, width=2), mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=temporal_recent["month"], y=temporal_recent["positive_rate"] * 100,
        name="Positive", line=dict(color=POSITIVE, width=2), mode="lines+markers"
    ))
    fig.update_layout(xaxis_title="Month", yaxis_title="Rate (%)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("LDA Topic Keyword Explorer")
    coherence_data = lda.groupby(["scope", "mode"])["coherence"].first().reset_index()
    coherence_pivot = coherence_data.pivot(index="scope", columns="mode", values="coherence")
    fig = px.imshow(
        coherence_pivot, text_auto=".3f",
        color_continuous_scale=["#FFFFFF", "#B3CCE8", "#0066CC", PG_BLUE],
        title="c_v Coherence Score Heatmap"
    )
    fig.update_traces(textfont=dict(color="black", size=13))
    st.plotly_chart(fig, use_container_width=True)

    _, col1, col2, _ = st.columns([0.5, 2, 2, 0.5])
    with col1:
        scope_options = lda["scope"].unique().tolist()
        selected_scope = st.selectbox("Select Source", scope_options)
    with col2:
        mode_options = lda[lda["scope"] == selected_scope]["mode"].unique().tolist()
        selected_mode = st.selectbox("Select Mode", mode_options)

    filtered = lda[(lda["scope"] == selected_scope) & (lda["mode"] == selected_mode)]
    if len(filtered) > 0:
        coherence_val = filtered["coherence"].iloc[0]
        optimal_k = filtered["optimal_k"].iloc[0]
        st.markdown(
            f"<div style='background-color:{PG_BLUE}; padding:12px; border-radius:6px;"
            f"color:white; font-weight:bold; font-size:16px;'>"
            f"Optimal k = {optimal_k} &nbsp;|&nbsp; Coherence = {coherence_val:.4f}"
            f"</div>", unsafe_allow_html=True
        )
        st.markdown("")
        for _, row in filtered.iterrows():
            st.markdown(
                f"<div style='background-color:#2B2B2B; padding:8px 12px;"
                f"border-left:3px solid {PG_BLUE}; margin-bottom:6px;"
                f"border-radius:4px; color:#FFFFFF; font-family:monospace;'>"
                f"<b>Topic {row['topic_id']}</b>: {row['keywords']}"
                f"</div>", unsafe_allow_html=True
            )


## Tab 3 — Trend Analysis
with tab3:
    st.header("Trend Analysis (Layer 2)")

    # key trend metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Antitro/HNS Ratio", f"{trend['antitro_ratio'].iloc[-1]:.3f}",
                  delta="Reversed 2025-06")
    with col2:
        st.metric("HNS 3M Momentum", f"{trend['hns_momentum'].iloc[-1]:.3f}",
                  delta="-23.7%", delta_color="inverse")
    with col3:
        cat_2026 = trend[trend.index.year == 2026]["category_click"].mean()
        cat_2020 = trend[trend.index.year == 2020]["category_click"].mean()
        cat_chg = (cat_2026 - cat_2020) / cat_2020 * 100
        st.metric("Category 2026 vs 2020", f"{cat_chg:+.1f}%")
    with col4:
        st.metric("Antitro First Entry", "2024-12",
                  delta="Reversal in 6 months")

    st.divider()

    # antitro vs hns trend
    st.subheader("Antitro vs HNS Search Volume Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend.index, y=trend["헤드앤숄더샴푸"],
        name="HNS Core", line=dict(color=PG_BLUE, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=trend.index, y=trend["안티트로샴푸"],
        name="Antitro", line=dict(color=CHURN_RED, width=2)
    ))
    fig.add_vline(x=pd.Timestamp("2025-06-30").timestamp() * 1000,
                  line_dash="dash",
                  line_color=WARNING, annotation_text="Reversal Point")
    fig.update_layout(xaxis_title="Date", yaxis_title="Search Volume",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HNS Product Line Lifecycle")
        hns_lines = ["헤드앤숄더샴푸", "헤드앤숄더클리니컬스트렝스",
                     "헤드앤숄더프로페셔널", "헤드앤숄더차콜"]
        line_labels = ["Core", "Clinical", "Professional", "Charcoal"]
        colors = [PG_BLUE, CHURN_RED, POSITIVE, WARNING]
        fig = go.Figure()
        for col_name, label, color in zip(hns_lines, line_labels, colors):
            fig.add_trace(go.Scatter(
                x=trend.index, y=trend[col_name],
                name=label, line=dict(color=color, width=2)
            ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Search Volume",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Chronos 12-Month Forecast (HNS Core)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend["헤드앤숄더샴푸"],
            name="Historical", line=dict(color=PG_BLUE, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["forecast_median"],
            name="Forecast (median)", line=dict(color=PG_LIGHT, width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"].iloc[::-1]]),
            y=pd.concat([forecast["forecast_high"], forecast["forecast_low"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(0,102,204,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI"
        ))
        fig.update_layout(xaxis_title="Date", yaxis_title="Search Volume",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Symptom Category Language Shift")
    symptom_cols = ["비듬샴푸", "지루성두피샴푸", "안티트로샴푸"]
    symptom_labels = ["Dandruff Shampoo", "Seborrheic Shampoo", "Antitro Shampoo"]
    colors = [PG_BLUE, PG_LIGHT, CHURN_RED]
    fig = go.Figure()
    for col_name, label, color in zip(symptom_cols, symptom_labels, colors):
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend[col_name],
            name=label, line=dict(color=color, width=2)
        ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Search Volume",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


## Tab 4 — Switching Risk
with tab4:
    st.header("Switching Risk Analysis (Layer 3)")

    # segment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Consumer Segment Distribution")
        seg_counts = voc_feat["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        color_map = {
            "Active Switcher": CHURN_RED,
            "At-risk": WARNING,
            "Passive User": NEUTRAL,
            "Loyal": POSITIVE
        }
        fig = px.bar(
            seg_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Switching Probability by Segment")
        seg_order = ["Active Switcher", "At-risk", "Passive User", "Loyal"]
        seg_prob_ordered = seg_prob.copy()
        seg_prob_ordered["segment"] = pd.Categorical(
            seg_prob_ordered["segment"], categories=seg_order, ordered=True
        )
        seg_prob_ordered = seg_prob_ordered.sort_values("segment")
        fig = go.Figure(go.Bar(
            x=seg_prob_ordered["segment"],
            y=seg_prob_ordered["switching_probability"],
            marker_color=[CHURN_RED, WARNING, NEUTRAL, POSITIVE],
            text=[f"{p:.3f}" for p in seg_prob_ordered["switching_probability"]],
            textposition="outside"
        ))
        fig.update_layout(yaxis_title="Switching Probability", yaxis_range=[0, 0.6])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Segment × Source Channel Distribution")
    seg_source = voc_feat.groupby(["segment", "source"]).size().unstack(fill_value=0)
    fig = px.bar(
        seg_source.reset_index().melt(id_vars="segment"),
        x="segment", y="value", color="source", barmode="group",
        color_discrete_map={
            "blog": PG_BLUE, "cafearticle": PG_LIGHT, "youtube": PG_SKY
        },
        labels={"value": "Document Count"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Intervention Plan by Segment")
    for _, row in impl.iterrows():
        risk_color = {
            "Critical": CHURN_RED,
            "High": WARNING,
            "Moderate": PG_LIGHT,
            "Low": POSITIVE
        }.get(row["risk_level"], PG_BLUE)
        st.markdown(
            f"<div style='background-color:#2B2B2B; padding:12px 16px;"
            f"border-left:4px solid {risk_color}; margin-bottom:10px;"
            f"border-radius:4px; color:#FFFFFF;'>"
            f"<b style='color:{risk_color}'>[{row['risk_level']}] {row['segment']}</b> "
            f"&nbsp;|&nbsp; P(switch)={row['switching_probability']:.3f} "
            f"&nbsp;|&nbsp; {row['intervention_timing']}<br><br>"
            f"{row['recommended_action']}"
            f"</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Timeline: Antitro Competitive Pressure")
    tl = timeline.iloc[0]
    st.markdown(
        f"<div style='background-color:#1A1A2E; padding:16px; border-radius:8px;"
        f"color:#FFFFFF; line-height:2.0;'>"
        f"🟡 <b>2024-12</b>: Antitro first entry<br>"
        f"🔴 <b>{tl['antitro_reversal_point']}</b>: First reversal vs HNS Core<br>"
        f"⚡ <b>{tl['antitro_acceleration_point']}</b>: Acceleration peak "
        f"(+{tl['antitro_acceleration_delta']:.1f} single month)<br>"
        f"📍 <b>Current</b>: ratio={tl['current_antitro_ratio']:.3f} | "
        f"HNS momentum={tl['current_hns_momentum']:.3f}<br><br>"
        f"<b style='color:{WARNING}'>Strategic window: "
        f"{tl['strategic_window']}</b>"
        f"</div>",
        unsafe_allow_html=True
    )


## Tab 5 — BERTopic
with tab5:
    st.header("BERTopic Analysis (Layer 1)")

    st.subheader("Document Count per Topic")
    _, col_center, _ = st.columns([0.1, 9.8, 0.1])
    with col_center:
        bertopic_sorted = bertopic.sort_values("Count", ascending=False)
        fig = px.bar(
            bertopic_sorted, x="Count", y="Name", orientation="h",
            color="Count",
            color_continuous_scale=["#B3CCE8", "#0066CC", PG_BLUE],
            title="BERTopic Topic Distribution"
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600,
                          yaxis_tickfont=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("LDA × BERTopic Consensus")
    _, col_table, _ = st.columns([0.5, 9, 0.5])
    with col_table:
        consensus_display = consensus[
            ["lda_topic", "bertopic_id", "overlap_keywords", "confidence"]
        ].copy()
        consensus_display["overlap_keywords"] = consensus_display[
            "overlap_keywords"
        ].apply(format_keywords)
        consensus_display["confidence"] = consensus_display["confidence"].str.capitalize()
        st.dataframe(consensus_display, use_container_width=True, hide_index=True)
        st.caption(
            "**High**: both models agree → high-confidence signal  \n"
            "**Low**: LDA-only signal → requires further validation"
        )

    st.divider()

    st.subheader("Methodology Comparison")
    _, col_method, _ = st.columns([0.5, 9, 0.5])
    with col_method:
        comparison_data = {
            "Item": ["Algorithm", "Num Topics", "Best Coherence",
                     "Strength", "Limitation"],
            "LDA": [
                "Latent Dirichlet Allocation",
                "k=2~7 (auto-optimized)",
                "0.6329 (cafearticle, bigram)",
                "Interpretable, domain-specific",
                "No word order or context"
            ],
            "BERTopic": [
                "BERT Embeddings + HDBSCAN",
                "17 (auto-detected, min_topic_size=8)",
                "Semantic similarity-based",
                "Context-aware, handles synonyms",
                "Broad clusters due to small corpus"
            ]
        }
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True
        )