import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background-color: #0a0d14;
        color: #e2e8f0;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 3rem 2.5rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: -40px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0 0 0.5rem;
    }
    .hero-title span { color: #818cf8; }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.3px;
    }
    .badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #818cf8;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Section Headers ── */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1.2rem;
    }

    /* ── Cards ── */
    .card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-accent {
        border-left: 3px solid #6366f1;
    }

    /* ── Model Result Cards ── */
    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 0.8rem;
    }
    .result-fraud {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(153,27,27,0.08) 100%);
        border: 1px solid rgba(239,68,68,0.3);
    }
    .result-legit {
        background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,95,70,0.08) 100%);
        border: 1px solid rgba(16,185,129,0.3);
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .result-label-fraud { color: #f87171; }
    .result-label-legit { color: #34d399; }
    .result-verdict {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .result-verdict-fraud { color: #fca5a5; }
    .result-verdict-legit { color: #6ee7b7; }
    .result-prob {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* ── Probability Bar ── */
    .prob-bar-wrap {
        background: #1e293b;
        border-radius: 999px;
        height: 6px;
        margin: 0.6rem 0 0.4rem;
        overflow: hidden;
    }
    .prob-bar-fill-fraud {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    .prob-bar-fill-legit {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #10b981, #059669);
    }

    /* ── Confidence Winner ── */
    .winner-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .winner-icon { font-size: 2rem; }
    .winner-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.2rem;
    }
    .winner-name { font-size: 1rem; font-weight: 600; color: #e2e8f0; }

    /* ── Dataset pill ── */
    .dataset-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #34d399;
        font-family: 'Space Mono', monospace;
        margin-bottom: 1.5rem;
    }
    .dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; display: inline-block; }

    /* ── Streamlit overrides ── */
    .stNumberInput > div > div > input {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'Space Mono', monospace !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stDataFrameContainer"] {
        border: 1px solid #1e293b;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Hero Banner
# =========================
st.markdown("""
<div class="hero">
    <p class="hero-title">💳 <span>Fraud</span> Detection</p>
    <p class="hero-sub">Real-time credit card transaction analysis powered by ensemble machine learning</p>
    <span class="badge">RF + XGBoost · Threshold 0.20</span>
</div>
""", unsafe_allow_html=True)

# =========================
# Paths & Models
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
rf_model  = joblib.load(BASE_DIR / "models" / "random_forest.pkl")
xgb_model = joblib.load(BASE_DIR / "models" / "xgboost.pkl")
data_path = BASE_DIR / "data" / "raw" / "creditcard.csv"

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

st.markdown(f"""
<div class="dataset-pill">
    <span class="dot"></span>
    Dataset loaded &nbsp;·&nbsp; {df.shape[0]:,} rows &nbsp;×&nbsp; {df.shape[1]} columns
</div>
""", unsafe_allow_html=True)

# =========================
# Feature Engineering
# =========================
def feature_engineering(df_in):
    d = df_in.copy()
    d['scaled_amount']   = (d['Amount'] - d['Amount'].mean()) / (d['Amount'].std() + 1e-8)
    d['scaled_time']     = (d['Time'] - d['Time'].mean()) / (d['Time'].std() + 1e-8)
    d['log_amount']      = np.log1p(d['Amount'])
    d['Hour']            = (d['Time'] // 3600) % 24
    d['is_night']        = (d['Hour'] <= 6).astype(int)
    d['high_amount']     = (d['Amount'] > d['Amount'].quantile(0.95)).astype(int)
    d['amount_per_hour'] = d['Amount'] / (d['Hour'] + 1)
    return d

df_feat = feature_engineering(df)

# =========================
# Transaction Selector
# =========================
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Select Transaction</p>', unsafe_allow_html=True)

    row_index = st.number_input(
        "Transaction index",
        min_value=0,
        max_value=len(df_feat) - 1,
        value=0,
        help="Pick any row from the dataset to analyse"
    )

    actual_label = df_feat.iloc[row_index]["Class"]
    label_html = (
        '<span style="color:#f87171;font-weight:600;">⚠ Fraudulent</span>'
        if actual_label == 1
        else '<span style="color:#34d399;font-weight:600;">✔ Legitimate</span>'
    )
    st.markdown(
        f'<div class="card card-accent" style="margin-top:1rem;">'
        f'<p class="section-label">Ground Truth</p>'
        f'{label_html}</div>',
        unsafe_allow_html=True
    )

    amount = df_feat.iloc[row_index]["Amount"]
    hour   = int(df_feat.iloc[row_index]["Hour"])
    st.markdown(
        f'<div class="card">'
        f'<p class="section-label">Quick Stats</p>'
        f'<p style="margin:0;font-size:0.9rem;color:#94a3b8;">Amount</p>'
        f'<p style="margin:0 0 0.8rem;font-family:\'Space Mono\',monospace;font-size:1.1rem;color:#e2e8f0;">'
        f'${amount:,.2f}</p>'
        f'<p style="margin:0;font-size:0.9rem;color:#94a3b8;">Hour of Day</p>'
        f'<p style="margin:0;font-family:\'Space Mono\',monospace;font-size:1.1rem;color:#e2e8f0;">'
        f'{hour:02d}:00</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    predict_btn = st.button("🚀  Run Prediction")

with col_right:
    st.markdown('<p class="section-label">Preview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Transaction Data</p>', unsafe_allow_html=True)
    st.dataframe(df_feat.iloc[[row_index]], use_container_width=True, height=220)

# =========================
# Results
# =========================
if predict_btn:
    X = df_feat.drop(columns=["Class", "Amount", "Time"])
    selected_row = X.iloc[row_index].values.reshape(1, -1)

    rf_prob  = rf_model.predict_proba(selected_row)[0][1]
    xgb_prob = xgb_model.predict_proba(selected_row)[0][1]
    rf_pred  = rf_prob  > 0.2
    xgb_pred = xgb_prob > 0.2

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Model Results</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    def render_result(col, name, icon, prob, is_fraud):
        verdict   = "🚨 FRAUD DETECTED"     if is_fraud else "✅ LEGIT TRANSACTION"
        card_cls  = "result-fraud"           if is_fraud else "result-legit"
        lbl_cls   = "result-label-fraud"     if is_fraud else "result-label-legit"
        vrd_cls   = "result-verdict-fraud"   if is_fraud else "result-verdict-legit"
        bar_cls   = "prob-bar-fill-fraud"    if is_fraud else "prob-bar-fill-legit"
        fill_pct  = prob * 100

        col.markdown(f"""
        <div class="result-card {card_cls}">
            <p class="result-label {lbl_cls}">{icon} {name}</p>
            <p class="result-verdict {vrd_cls}">{verdict}</p>
            <div class="prob-bar-wrap">
                <div class="{bar_cls}" style="width:{fill_pct:.1f}%"></div>
            </div>
            <p class="result-prob">Fraud probability: {prob:.6f} &nbsp;({fill_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)

    render_result(c1, "Random Forest", "🌲", rf_prob,  rf_pred)
    render_result(c2, "XGBoost",       "⚡", xgb_prob, xgb_pred)

    # ── Confidence winner ──
    rf_conf  = abs(rf_prob  - 0.5)
    xgb_conf = abs(xgb_prob - 0.5)

    if rf_conf > xgb_conf:
        winner, winner_icon, conf = "Random Forest", "🌲", rf_conf
    else:
        winner, winner_icon, conf = "XGBoost", "⚡", xgb_conf

    st.markdown(f"""
    <div class="winner-card">
        <span class="winner-icon">{winner_icon}</span>
        <div>
            <p class="winner-title">Most Confident Model</p>
            <p class="winner-name">{winner} &nbsp;<span style="color:#64748b;font-size:0.85rem;font-weight:400;">
            — confidence delta {conf:.4f}</span></p>
        </div>
    </div>
    """, unsafe_allow_html=True)