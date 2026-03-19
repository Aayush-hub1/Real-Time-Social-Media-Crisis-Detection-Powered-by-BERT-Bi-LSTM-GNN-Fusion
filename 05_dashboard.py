"""
DeepSentinel — Real-Time Crisis Detection Dashboard
Run with: streamlit run 05_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepSentinel — Social Media Crisis Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border: 1px solid #3a3a5e;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .crisis-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-misinfo    { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-mental     { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .badge-bully      { background: #8b5cf622; color: #8b5cf6; border: 1px solid #8b5cf655; }
    .badge-political  { background: #06b6d422; color: #06b6d4; border: 1px solid #06b6d455; }
    .badge-normal     { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .risk-high   { color: #ef4444; font-weight: 700; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-low    { color: #10b981; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─── Simulated Inference (replace with real model in production) ──────────────

LABELS     = ["misinformation", "mental_health", "cyberbullying", "political_unrest", "normal"]
LABEL_COLS = {"misinformation": "#f59e0b", "mental_health": "#ef4444",
              "cyberbullying": "#8b5cf6", "political_unrest": "#06b6d4", "normal": "#10b981"}

SAMPLE_POSTS = [
    "This vaccine contains microchips! Share before they delete this!! #truth #fakemedia",
    "I can't do this anymore. Nobody cares. Feeling completely alone tonight 😭",
    "You're pathetic and deserve everything bad that happens to you 🤬",
    "BREAKING: Police crackdown on protesters downtown. Tensions rising #protest #riot",
    "Just had the best coffee of my life ☕ Weekend vibes!",
    "Scientists say COVID was engineered in a lab. The media won't tell you the truth!",
    "I just want to disappear. Everything is falling apart and I don't see a way out 😢",
    "Mass arrests happening right now. Government is silencing opposition #resist",
    "Can't believe they're still bullying that kid in class. People are disgusting 😡",
    "Beautiful sunset today 🌅 Grateful for life!",
]

def fake_predict(text: str) -> dict:
    """Simulate model prediction for demo purposes."""
    rng  = random.Random(hash(text) % 10000)
    base = rng.choices(LABELS, weights=[0.2, 0.2, 0.15, 0.15, 0.3])[0]
    probs = np.array([rng.uniform(0.02, 0.12) for _ in LABELS])

    # Boost the predicted class
    idx = LABELS.index(base)
    probs[idx] = rng.uniform(0.55, 0.90)
    probs = probs / probs.sum()

    risk = "HIGH" if probs[idx] > 0.75 else "MEDIUM" if probs[idx] > 0.55 else "LOW"
    return {
        "label":       base,
        "confidence":  float(probs[idx]),
        "probs":       {l: float(p) for l, p in zip(LABELS, probs)},
        "risk_level":  risk,
    }

def generate_live_feed(n=25) -> pd.DataFrame:
    """Generate simulated live feed for demo."""
    rows = []
    now  = datetime.now()
    for i in range(n):
        text   = random.choice(SAMPLE_POSTS)
        pred   = fake_predict(text)
        offset = random.randint(0, 3600)
        rows.append({
            "time":       (now - timedelta(seconds=offset)).strftime("%H:%M:%S"),
            "source":     random.choice(["Twitter/X", "Reddit", "NewsAPI"]),
            "text":       text[:90] + ("…" if len(text) > 90 else ""),
            "label":      pred["label"],
            "confidence": pred["confidence"],
            "risk":       pred["risk_level"],
        })
    return pd.DataFrame(rows).sort_values("time", ascending=False)


def generate_trend_data() -> pd.DataFrame:
    """Generate 24h trend data for plotting."""
    hours = pd.date_range(end=datetime.now(), periods=24, freq="H")
    rows  = []
    for h in hours:
        for label in LABELS:
            rows.append({
                "hour":  h,
                "label": label,
                "count": max(0, int(np.random.normal(
                    loc=30 if label != "normal" else 80,
                    scale=12
                ))),
            })
    return pd.DataFrame(rows)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1e1e2e/ffffff?text=DeepSentinel", use_column_width=True)
    st.markdown("### ⚙️ Settings")

    refresh_rate = st.slider("Auto-refresh (seconds)", 5, 60, 15)
    threshold    = st.slider("Alert threshold (confidence)", 0.5, 0.95, 0.70, 0.05)
    sources      = st.multiselect("Data sources", ["Twitter/X", "Reddit", "NewsAPI"],
                                  default=["Twitter/X", "Reddit", "NewsAPI"])
    crisis_types = st.multiselect("Monitor crisis types", LABELS, default=LABELS[:-1])

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.metric("Architecture", "BERT + Bi-LSTM + GNN")
    st.metric("Macro F1",     "0.847")
    st.metric("Latency",      "~120ms / post")
    st.markdown("---")
    st.caption("MCA AI & ML Major Project · DeepSentinel v1.0")

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🛡️ DeepSentinel")
st.markdown("**Real-Time Social Media Crisis Detection** · Powered by BERT + Bi-LSTM + GNN Fusion")
st.markdown("---")

# ─── Live Predict Section ────────────────────────────────────────────────────

st.subheader("🔍 Live Crisis Analyzer")
col_input, col_result = st.columns([2, 1])

with col_input:
    user_text = st.text_area(
        "Paste a social media post to analyze:",
        placeholder="Enter post text here...",
        height=100,
    )
    analyze_btn = st.button("Analyze Post", type="primary")

if analyze_btn and user_text.strip():
    with st.spinner("Running DeepSentinel model..."):
        time.sleep(0.8)
        result = fake_predict(user_text)

    with col_result:
        risk_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}[result["risk_level"]]
        st.markdown(f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:16px;border:1px solid {risk_color}55">
            <p style="color:#aaa;font-size:12px;margin:0">Prediction</p>
            <p style="font-size:22px;font-weight:700;color:{LABEL_COLS[result['label']]};margin:4px 0">
                {result['label'].replace('_',' ').title()}
            </p>
            <p style="color:#aaa;font-size:12px;margin:0">Risk Level</p>
            <p style="font-size:18px;font-weight:700;color:{risk_color};margin:4px 0">
                {result['risk_level']}
            </p>
            <p style="color:#aaa;font-size:12px;margin:0">Confidence</p>
            <p style="font-size:18px;font-weight:600;color:#fff;margin:4px 0">
                {result['confidence']*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Class probability breakdown:**")
    prob_df = pd.DataFrame(
        {"Class": list(result["probs"].keys()), "Probability": list(result["probs"].values())}
    )
    fig_prob = px.bar(
        prob_df, x="Probability", y="Class", orientation="h",
        color="Class", color_discrete_map=LABEL_COLS,
        text_auto=".2%", template="plotly_dark", height=220,
    )
    fig_prob.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

# ─── KPI Metrics Row ─────────────────────────────────────────────────────────

feed = generate_live_feed(50)
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("🔴 Total Alerts",       len(feed[feed["risk"] == "HIGH"]),   "+3 last hour")
m2.metric("📰 Misinformation",     len(feed[feed["label"]=="misinformation"]),  "")
m3.metric("💙 Mental Health",       len(feed[feed["label"]=="mental_health"]),   "")
m4.metric("🛑 Cyberbullying",      len(feed[feed["label"]=="cyberbullying"]),   "")
m5.metric("⚡ Political Unrest",   len(feed[feed["label"]=="political_unrest"]), "")

st.markdown("---")

# ─── Charts Row ──────────────────────────────────────────────────────────────

col_chart1, col_chart2 = st.columns([3, 2])

with col_chart1:
    st.subheader("📈 24-Hour Crisis Trend")
    trend = generate_trend_data()
    trend_filtered = trend[trend["label"].isin(crisis_types)]
    fig_trend = px.line(
        trend_filtered, x="hour", y="count", color="label",
        color_discrete_map=LABEL_COLS,
        template="plotly_dark", height=300,
    )
    fig_trend.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title="Crisis Type",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_chart2:
    st.subheader("📊 Distribution")
    label_counts = feed["label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    fig_pie = px.pie(
        label_counts, names="label", values="count",
        color="label", color_discrete_map=LABEL_COLS,
        template="plotly_dark", height=300, hole=0.4,
    )
    fig_pie.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── Live Feed Table ─────────────────────────────────────────────────────────

st.subheader("📡 Live Alert Feed")

feed_display = generate_live_feed(20)
feed_display = feed_display[feed_display["label"].isin(crisis_types)]
feed_display = feed_display[feed_display["confidence"] >= threshold]

# Color-code risk
def style_risk(val):
    colors = {"HIGH": "color: #ef4444; font-weight: 700",
              "MEDIUM": "color: #f59e0b; font-weight: 600",
              "LOW": "color: #10b981"}
    return colors.get(val, "")

def style_label(val):
    return f"color: {LABEL_COLS.get(val, '#fff')}"

styled = (
    feed_display.style
    .applymap(style_risk, subset=["risk"])
    .applymap(style_label, subset=["label"])
    .format({"confidence": "{:.1%}"})
)
st.dataframe(styled, use_container_width=True, height=400)

# ─── Auto Refresh Notice ─────────────────────────────────────────────────────

st.caption(f"🔄 Dashboard auto-refreshes every {refresh_rate}s · Data is simulated for demo · Connect live API keys for production")
