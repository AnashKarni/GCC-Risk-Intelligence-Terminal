"""
GCC Risk Intelligence & Prediction Hub — v7 PREMIUM
=====================================================
Bloomberg Terminal aesthetic · Production ML · Full UI Upgrade

New in v7:
  🎨 Bloomberg/Fintech terminal UI — scanlines, glow effects, terminal fonts
  📊 Dashboard KPI cards with animated counters at top
  📈 Colored graphs with gradient fills under curves
  🎯 Risk Gauge chart in Live Predictor
  📄 PDF report export
  🔄 Animated transitions and micro-interactions
  ✅ All v6 ML fixes retained
"""

from __future__ import annotations

# --- SECURE LOCKER IMPORTS ---
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() # Ye command start hote hi tera .env locker khol degi

# --- STANDARD ML & UI IMPORTS ---
import io
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# BLOOMBERG TERMINAL COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
C_BG        = "#0a0a0f"
C_SURFACE   = "#0f0f1a"
C_CARD      = "#13131f"
C_BORDER    = "#1e2d40"
C_ACCENT    = "#00d4ff"      # Bloomberg cyan
C_ACCENT2   = "#ff6b35"      # Orange accent
C_GREEN     = "#00ff88"      # Profit green
C_RED       = "#ff3366"      # Loss red
C_YELLOW    = "#ffcc00"      # Warning amber
C_TEXT      = "#a8c4d4"
C_TEXT_DIM  = "#4a6070"
C_GRID      = "#1a2535"

MODEL_COLORS = [C_ACCENT, C_GREEN, C_ACCENT2]

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GCC Risk Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS — Bloomberg Terminal Aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {C_BG} !important;
    color: {C_TEXT};
}}
code, .mono {{ font-family: 'IBM Plex Mono', monospace !important; }}
.block-container {{ padding: 3rem 1.5rem 1rem 1.5rem !important; max-width: 100% !important; }}
.stApp {{ background-color: {C_BG}; }}

/* ── Scanline effect on main bg ── */
.stApp::before {{
    content: '';
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,212,255,0.012) 2px,
        rgba(0,212,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 0;
}}

/* ── Terminal Header ── */
.terminal-header {{
    background: linear-gradient(135deg, {C_CARD} 0%, #0d1520 50%, {C_CARD} 100%);
    border: 1px solid {C_BORDER};
    border-top: 2px solid {C_ACCENT};
    border-radius: 4px;
    padding: 1rem 1.5rem 0.8rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}}
.terminal-header::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, {C_ACCENT}, {C_GREEN}, transparent);
    animation: scanH 3s linear infinite;
}}
@keyframes scanH {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
.terminal-header .title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 700;
    color: {C_ACCENT};
    letter-spacing: 0.08em;
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
}}
.terminal-header .subtitle {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: {C_TEXT_DIM};
    letter-spacing: 0.15em; margin-top: 0.2rem;
}}
.terminal-header .status-bar {{
    display: flex; gap: 1.5rem; margin-top: 0.6rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
}}
.status-dot {{
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: {C_GREEN};
    box-shadow: 0 0 6px {C_GREEN};
    animation: pulse 2s ease-in-out infinite;
    margin-right: 5px; vertical-align: middle;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 6px {C_GREEN}; }}
    50% {{ opacity: 0.4; box-shadow: 0 0 2px {C_GREEN}; }}
}}

/* ── KPI Cards ── */
.kpi-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
    margin-bottom: 0.8rem;
}}
.kpi-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-left: 3px solid {C_ACCENT};
    border-radius: 4px;
    padding: 0.8rem 1rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.kpi-card:hover {{
    border-left-color: {C_GREEN};
    box-shadow: 0 0 15px rgba(0,212,255,0.1);
}}
.kpi-card::after {{
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, {C_ACCENT}44, transparent);
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: {C_TEXT_DIM};
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.3rem;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    color: {C_ACCENT};
    text-shadow: 0 0 15px rgba(0,212,255,0.4);
    line-height: 1;
}}
.kpi-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: {C_TEXT_DIM};
    margin-top: 0.25rem;
}}
.kpi-green {{ border-left-color: {C_GREEN} !important; }}
.kpi-green .kpi-value {{ color: {C_GREEN}; text-shadow: 0 0 15px rgba(0,255,136,0.4); }}
.kpi-red {{ border-left-color: {C_RED} !important; }}
.kpi-red .kpi-value {{ color: {C_RED}; text-shadow: 0 0 15px rgba(255,51,102,0.4); }}
.kpi-amber {{ border-left-color: {C_YELLOW} !important; }}
.kpi-amber .kpi-value {{ color: {C_YELLOW}; text-shadow: 0 0 15px rgba(255,204,0,0.4); }}

/* ── Section Headers ── */
.section-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    color: {C_ACCENT};
    letter-spacing: 0.2em; text-transform: uppercase;
    border-bottom: 1px solid {C_BORDER};
    padding-bottom: 0.4rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}}
.section-header::before {{
    content: '▶'; font-size: 0.5rem; color: {C_ACCENT};
}}

/* ── Model Leaderboard ── */
.leaderboard-row {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    transition: background 0.2s;
}}
.leaderboard-row:hover {{ background: #16192a; }}
.leaderboard-row.winner {{
    border-color: {C_ACCENT};
    background: linear-gradient(135deg, #0d1520, {C_CARD});
    box-shadow: 0 0 20px rgba(0,212,255,0.08);
}}
.lb-rank {{ color: {C_TEXT_DIM}; width: 20px; }}
.lb-name {{ color: {C_TEXT}; flex: 1; font-weight: 600; }}
.lb-metric {{ color: {C_ACCENT}; width: 80px; text-align: right; }}
.lb-badge {{ padding: 2px 8px; border-radius: 2px; font-size: 0.65rem; font-weight: 700; }}
.badge-excellent {{ background: rgba(0,255,136,0.15); color: {C_GREEN}; border: 1px solid {C_GREEN}44; }}
.badge-good      {{ background: rgba(255,204,0,0.15);  color: {C_YELLOW}; border: 1px solid {C_YELLOW}44; }}
.badge-poor      {{ background: rgba(255,51,102,0.15); color: {C_RED}; border: 1px solid {C_RED}44; }}

/* ── Prediction Result Box ── */
.pred-terminal {{
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 4px; padding: 1.2rem 1.5rem;
    margin: 0.8rem 0; text-align: center;
    font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.05em;
    position: relative; overflow: hidden;
}}
.pred-safe {{
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,212,255,0.05));
    border: 1px solid {C_GREEN};
    color: {C_GREEN};
    box-shadow: 0 0 30px rgba(0,255,136,0.1), inset 0 0 30px rgba(0,255,136,0.03);
    text-shadow: 0 0 20px rgba(0,255,136,0.6);
}}
.pred-danger {{
    background: linear-gradient(135deg, rgba(255,51,102,0.08), rgba(255,107,53,0.05));
    border: 1px solid {C_RED};
    color: {C_RED};
    box-shadow: 0 0 30px rgba(255,51,102,0.15), inset 0 0 30px rgba(255,51,102,0.03);
    text-shadow: 0 0 20px rgba(255,51,102,0.6);
}}
.pred-terminal .conf-bar {{
    height: 3px; border-radius: 2px; margin-top: 0.6rem;
    background: linear-gradient(90deg, currentColor, transparent);
}}

/* ── Data Table ── */
.dataframe {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C_SURFACE} !important;
    border-right: 1px solid {C_BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1rem !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C_CARD};
    border-bottom: 1px solid {C_BORDER};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.08em;
    color: {C_TEXT_DIM}; padding: 0.5rem 1rem;
    border-bottom: 2px solid transparent;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {C_ACCENT} !important;
    border-bottom-color: {C_ACCENT} !important;
    background: rgba(0,212,255,0.05) !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 0.1em !important;
    background: linear-gradient(135deg, #0d1a26, #0a1520) !important;
    color: {C_ACCENT} !important;
    border: 1px solid {C_ACCENT}66 !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    background: rgba(0,212,255,0.1) !important;
    border-color: {C_ACCENT} !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.2) !important;
    color: #fff !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
}}
[data-testid="metric-container"] label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem !important; letter-spacing: 0.1em;
    color: {C_TEXT_DIM} !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    color: {C_ACCENT} !important; font-size: 1.4rem !important;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important; color: {C_TEXT_DIM} !important;
    background: {C_CARD} !important; border: 1px solid {C_BORDER} !important;
}}
.streamlit-expanderContent {{ background: {C_CARD} !important; border: 1px solid {C_BORDER} !important; border-top: none !important; }}

/* ── Selectbox / Slider ── */
.stSelectbox label, .stSlider label, .stRadio label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important; color: {C_TEXT_DIM} !important;
    letter-spacing: 0.06em;
}}
div[data-baseweb="select"] > div {{
    background: {C_CARD} !important;
    border-color: {C_BORDER} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}}

/* ── Progress bar ── */
.stProgress > div > div {{ background: {C_ACCENT} !important; }}

/* ── Alert boxes ── */
.stAlert {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{ background: {C_BORDER}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C_ACCENT}66; }}

/* ── Risk gauge container ── */
.gauge-wrap {{ display: flex; justify-content: center; margin: 0.5rem 0; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TERMINAL HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="terminal-header">
    <div class="title">📡 GCC RISK INTELLIGENCE TERMINAL</div>
    <div class="subtitle">QUANTITATIVE RISK ANALYTICS · ML PREDICTION ENGINE · REAL-TIME EXPLAINABILITY</div>
    <div class="status-bar">
        <span><span class="status-dot"></span>SYSTEM ONLINE</span>
        <span style="color:#4a6070">│</span>
        <span style="color:#ffcc00">ENGINE: XGBoost + RF + LR</span>
        <span style="color:#4a6070">│</span>
        <span style="color:#00d4ff">SHAP EXPLAINER READY</span>
        <span style="color:#4a6070">│</span>
        <span style="color:#00ff88">ZERO-LEAKAGE PIPELINE</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def terminal_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT)
    ax.yaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7)
    return fig, ax


def get_badge_html(score: float) -> str:
    if score >= 0.85:
        return f'<span class="lb-badge badge-excellent">▲ {score:.3f}</span>'
    elif score >= 0.70:
        return f'<span class="lb-badge badge-good">◆ {score:.3f}</span>'
    else:
        return f'<span class="lb-badge badge-poor">▼ {score:.3f}</span>'


def kpi_card(label: str, value: str, sub: str = "", color_class: str = "") -> str:
    return f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def section_header(icon: str, title: str):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if df.empty or df.shape[1] < 2:
            raise ValueError("File has fewer than 2 columns.")
        return df
    except Exception as exc:
        st.error(f"Cannot read file: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CONTEXT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
def detect_context_and_target(df: pd.DataFrame) -> tuple:
    cols = df.columns.tolist()
    if "class" in cols and any("v" in c for c in cols):
        return "Credit Card Fraud", "class"
    elif "exited" in cols:
        return "Customer Churn", "exited"
    elif "churn" in cols:
        return "Customer Churn", "churn"
    elif "loan_status" in cols:
        return "Loan Risk", "loan_status"
    elif "default" in cols:
        return "Loan Risk", "default"
    elif "fraud" in cols:
        return "Financial Fraud", "fraud"
    elif "risk" in cols:
        return "General Risk", "risk"
    for col in df.columns:
        if df[col].nunique() == 2:
            return "General Finance", col
    return "General Finance", None


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, target_col: str) -> tuple:
    df_proc = df.copy()
    drop_cols: list = []
    for col in df_proc.columns:
        if col == target_col:
            continue
        lc = col.lower()
        if any(kw in lc for kw in ("id", "name", "surname", "rownum")):
            drop_cols.append(col)
        elif df_proc[col].dtype == "object" and df_proc[col].nunique() > 50:
            drop_cols.append(col)
    df_proc.drop(columns=list(set(drop_cols)), inplace=True, errors="ignore")

    cat_feature_cols = [c for c in df_proc.select_dtypes(include="object").columns if c != target_col]
    for col in cat_feature_cols:
        df_proc[col] = df_proc[col].fillna("MISSING").astype(str)

    cat_mappings: dict = {col: sorted(df_proc[col].unique().tolist()) for col in cat_feature_cols}

    le = LabelEncoder()
    if df_proc[target_col].dtype == "object":
        df_proc[target_col] = le.fit_transform(df_proc[target_col].astype(str))
    else:
        df_proc[target_col] = df_proc[target_col].astype(int)

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    return X, y, cat_mappings


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train_all_models(X_train_raw, X_test_raw, y_train, y_test, progress_bar) -> tuple:
    cat_cols = X_train_raw.select_dtypes(include="object").columns.tolist()
    num_cols = X_train_raw.select_dtypes(include="number").columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")

    if cat_cols:
        ohe_train = ohe.fit_transform(X_train_raw[cat_cols])
        ohe_test  = ohe.transform(X_test_raw[cat_cols])
        ohe_cols  = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe_train = np.empty((len(X_train_raw), 0))
        ohe_test  = np.empty((len(X_test_raw), 0))
        ohe_cols  = []
    progress_bar.progress(15, "OHE fitted ✓")

    imputer = SimpleImputer(strategy="median")
    if num_cols:
        num_train = imputer.fit_transform(X_train_raw[num_cols])
        num_test  = imputer.transform(X_test_raw[num_cols])
    else:
        num_train = np.empty((len(X_train_raw), 0))
        num_test  = np.empty((len(X_test_raw), 0))
    progress_bar.progress(25, "Imputer fitted ✓")

    X_train = pd.DataFrame(np.hstack([num_train, ohe_train]), columns=num_cols + ohe_cols)
    X_test  = pd.DataFrame(np.hstack([num_test,  ohe_test]),  columns=num_cols + ohe_cols)
    final_feature_cols = num_cols + ohe_cols

    class_counts = y_train.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Only one class in training set: {class_counts.index[0]}")
    neg_count = int(class_counts.get(0, 1))
    pos_count = int(class_counts.get(1, 1))
    imbalance_ratio = float(min(max(neg_count / pos_count, 1.0), 100.0))

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models: dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGBoost":             xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0, scale_pos_weight=imbalance_ratio),
    }

    results: dict = {}
    for step_i, (name, model) in enumerate(models.items()):
        progress_bar.progress(30 + step_i * 23, f"Training {name}…")
        if name == "Logistic Regression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.0
        try:
            cv_X      = X_train_sc if name == "Logistic Regression" else X_train
            cv_scores = cross_val_score(model, cv_X, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())
        except Exception:
            cv_mean, cv_std = 0.0, 0.0

        results[name] = {
            "model": model, "y_pred": y_pred, "y_prob": y_prob,
            "accuracy": acc, "f1": f1, "auc": auc,
            "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
        }

    progress_bar.progress(100, "All models trained ✓")
    return results, ohe, imputer, scaler, final_feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# MAKE PROCESSED (SHAP)
# ──────────────────────────────────────────────────────────────────────────────
def make_processed(X_raw_df, ohe, imputer, cat_cols, num_cols) -> pd.DataFrame:
    if cat_cols:
        ohe_part  = ohe.transform(X_raw_df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        ohe_part, ohe_names = np.empty((len(X_raw_df), 0)), []
    if num_cols:
        num_part = imputer.transform(X_raw_df[num_cols])
    else:
        num_part = np.empty((len(X_raw_df), 0))
    return pd.DataFrame(np.hstack([num_part, ohe_part]), columns=num_cols + ohe_names)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS — Bloomberg style with gradient fills
# ──────────────────────────────────────────────────────────────────────────────
def plot_roc_premium(results: dict, y_test):
    fig, ax = terminal_fig((7, 4))
    for (name, res), col in zip(results.items(), MODEL_COLORS):
        try:
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  AUC={res['auc']:.3f}", alpha=0.9)
            ax.fill_between(fpr, tpr, alpha=0.06, color=col)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "--", color=C_GRID, lw=1, alpha=0.6)
    ax.set_xlabel("FALSE POSITIVE RATE", color=C_TEXT_DIM, fontsize=7, labelpad=6)
    ax.set_ylabel("TRUE POSITIVE RATE", color=C_TEXT_DIM, fontsize=7, labelpad=6)
    ax.set_title("ROC CURVE COMPARISON", color=C_ACCENT, fontsize=9, pad=8,
                 fontfamily="monospace", fontweight="bold")
    legend = ax.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT,
                       fontsize=7.5, framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout(pad=1.2)
    return fig


def plot_confusion_premium(y_test, y_pred, title: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = terminal_fig((4.5, 3.5))
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "bb", [C_CARD, C_ACCENT + "88", C_ACCENT], N=256
    )
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                linewidths=1, linecolor=C_BG,
                annot_kws={"size": 13, "color": "white", "fontfamily": "monospace", "fontweight": "bold"})
    ax.set_title(title, color=C_ACCENT, fontsize=8, pad=8, fontfamily="monospace")
    ax.set_xlabel("PREDICTED", color=C_TEXT_DIM, fontsize=7)
    ax.set_ylabel("ACTUAL", color=C_TEXT_DIM, fontsize=7)
    plt.tight_layout(pad=1.0)
    return fig


def plot_feature_importance_premium(model, feature_names: list, title: str):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])
    else:
        return None
    fi = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(15)
    n = len(fi)
    fig, ax = terminal_fig((7, max(3.5, n * 0.38)))

    colors = plt.cm.YlOrRd(np.linspace(0.35, 1.0, n))
    bars = ax.barh(fi.index, fi.values, color=colors, edgecolor="none", height=0.65)

    # Gradient glow effect on bars
    for bar, val, col in zip(bars, fi.values, colors):
        ax.barh(bar.get_y() + bar.get_height() / 2, val * 0.95,
                height=0.2, left=0, color=col, alpha=0.3, edgecolor="none")

    ax.set_title(f"FEATURE IMPORTANCE — {title.upper()}", color=C_ACCENT,
                 fontsize=8, pad=8, fontfamily="monospace", fontweight="bold")
    ax.set_xlabel("IMPORTANCE SCORE", color=C_TEXT_DIM, fontsize=7)
    ax.tick_params(labelsize=7.5)
    plt.tight_layout(pad=1.2)
    return fig


def plot_distribution_premium(series: pd.Series, col_name: str, is_numeric: bool):
    fig, ax = terminal_fig((6, 3))
    if is_numeric:
        n, bins, patches = ax.hist(series.dropna(), bins=40, edgecolor="none", alpha=0.0)
        for i, (patch, bin_left) in enumerate(zip(patches, bins)):
            t = i / len(patches)
            patch.set_facecolor(plt.cm.cool(t * 0.6 + 0.2))
            patch.set_alpha(0.85)
        # Fill under curve
        ax.fill_between(
            [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)],
            n, alpha=0.15, color=C_ACCENT
        )
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series.dropna())
            xs = np.linspace(series.min(), series.max(), 200)
            ys = kde(xs) * len(series) * (bins[1] - bins[0])
            ax.plot(xs, ys, color=C_ACCENT, lw=2, alpha=0.9)
            ax.fill_between(xs, ys, alpha=0.08, color=C_ACCENT)
        except Exception:
            pass
    else:
        vc = series.value_counts().head(12)
        colors = [C_ACCENT, C_GREEN, C_ACCENT2, C_YELLOW, C_RED] * 3
        bars = ax.bar(range(len(vc)), vc.values, color=colors[:len(vc)], edgecolor="none", width=0.7)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=30, ha="right", fontsize=7)
        for bar, val in zip(bars, vc.values):
            ax.fill_between(
                [bar.get_x(), bar.get_x() + bar.get_width()],
                [0, 0], [val, val], alpha=0.15,
                color=bar.get_facecolor()
            )

    ax.set_title(f"DISTRIBUTION: {col_name.upper()}", color=C_ACCENT,
                 fontsize=8, pad=8, fontfamily="monospace", fontweight="bold")
    ax.set_xlabel(col_name.upper(), color=C_TEXT_DIM, fontsize=7)
    ax.set_ylabel("COUNT / DENSITY", color=C_TEXT_DIM, fontsize=7)
    plt.tight_layout(pad=1.2)
    return fig


def plot_risk_gauge(probability: float):
    """Semicircular risk gauge — Bloomberg style."""
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    # Background arc zones
    theta = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.55

    # Safe zone (0–0.4)
    t1 = np.linspace(np.pi, np.pi * 0.6, 100)
    ax.fill_between(np.cos(t1), np.sin(t1) * r_inner, np.sin(t1) * r_outer,
                    color=C_GREEN, alpha=0.25)

    # Warning zone (0.4–0.7)
    t2 = np.linspace(np.pi * 0.6, np.pi * 0.3, 100)
    ax.fill_between(np.cos(t2), np.sin(t2) * r_inner, np.sin(t2) * r_outer,
                    color=C_YELLOW, alpha=0.25)

    # Danger zone (0.7–1.0)
    t3 = np.linspace(np.pi * 0.3, 0, 100)
    ax.fill_between(np.cos(t3), np.sin(t3) * r_inner, np.sin(t3) * r_outer,
                    color=C_RED, alpha=0.25)

    # Arc outlines
    for t_seg, col in [(t1, C_GREEN), (t2, C_YELLOW), (t3, C_RED)]:
        ax.plot(np.cos(t_seg) * r_outer, np.sin(t_seg) * r_outer, color=col, lw=2, alpha=0.8)
        ax.plot(np.cos(t_seg) * r_inner, np.sin(t_seg) * r_inner, color=col, lw=1, alpha=0.4)

    # Needle
    angle = np.pi * (1 - probability)
    needle_len = 0.72
    nx = np.cos(angle) * needle_len
    ny = np.sin(angle) * needle_len
    needle_color = C_GREEN if probability < 0.4 else (C_YELLOW if probability < 0.7 else C_RED)

    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=needle_color,
                                lw=2.5, mutation_scale=14,
                                connectionstyle="arc3,rad=0"))
    ax.plot(0, 0, "o", color=needle_color, markersize=8,
            markerfacecolor=C_CARD, markeredgewidth=2)

    # Labels
    ax.text(-1.08, -0.15, "LOW", color=C_GREEN, fontsize=7,
            fontfamily="monospace", ha="center", fontweight="bold")
    ax.text(0,     0.55,  "MED", color=C_YELLOW, fontsize=7,
            fontfamily="monospace", ha="center", fontweight="bold")
    ax.text(1.08,  -0.15, "HIGH", color=C_RED, fontsize=7,
            fontfamily="monospace", ha="center", fontweight="bold")

    # Central probability display
    ax.text(0, -0.28, f"{probability*100:.1f}%",
            color=needle_color, fontsize=18, fontfamily="monospace",
            ha="center", va="center", fontweight="bold")
    ax.text(0, -0.46, "RISK PROBABILITY",
            color=C_TEXT_DIM, fontsize=6.5, fontfamily="monospace",
            ha="center", va="center")

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.15)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig


def plot_prob_bars(pred_prob: np.ndarray):
    fig, ax = terminal_fig((6, 1.8))
    labels = ["LOW RISK", "HIGH RISK"]
    colors = [C_GREEN, C_RED]
    y_pos  = [0.65, 0.1]

    for i, (val, col, lbl, yp) in enumerate(zip(pred_prob, colors, labels, y_pos)):
        # Background track
        ax.barh(yp, 1.0, height=0.25, color=C_GRID, left=0, edgecolor="none")
        # Value bar
        ax.barh(yp, val, height=0.25, color=col, left=0, edgecolor="none", alpha=0.9)
        # Glow
        ax.barh(yp, val, height=0.08, color=col, left=0, edgecolor="none", alpha=0.35)
        ax.text(1.02, yp, f"{val*100:.1f}%", color=col, fontsize=8.5,
                fontfamily="monospace", fontweight="bold", va="center")
        ax.text(-0.01, yp, lbl, color=C_TEXT_DIM, fontsize=6.5,
                fontfamily="monospace", va="center", ha="right")

    ax.set_xlim(-0.22, 1.18)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(df, results, best_name, target_col, context) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()

        BG   = rl_colors.HexColor("#0a0a0f")
        CYAN = rl_colors.HexColor("#00d4ff")
        GRN  = rl_colors.HexColor("#00ff88")
        DIM  = rl_colors.HexColor("#a8c4d4")
        WHITE = rl_colors.white

        title_style = ParagraphStyle("title", fontSize=16, textColor=CYAN,
                                     fontName="Courier-Bold", spaceAfter=4)
        sub_style   = ParagraphStyle("sub",   fontSize=8,  textColor=DIM,
                                     fontName="Courier", spaceAfter=12)
        head_style  = ParagraphStyle("head",  fontSize=10, textColor=CYAN,
                                     fontName="Courier-Bold", spaceAfter=6, spaceBefore=12)
        body_style  = ParagraphStyle("body",  fontSize=8,  textColor=DIM,
                                     fontName="Courier", spaceAfter=4)

        story = []
        story.append(Paragraph("GCC RISK INTELLIGENCE TERMINAL", title_style))
        story.append(Paragraph(f"AUTOMATED RISK REPORT  ·  CONTEXT: {context.upper()}  ·  TARGET: {target_col.upper()}", sub_style))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("DATASET SUMMARY", head_style))
        data_rows = [
            ["METRIC", "VALUE"],
            ["Total Records", f"{len(df):,}"],
            ["Features", f"{df.shape[1] - 1}"],
            ["Target Column", target_col],
            ["Missing Values", f"{df.isnull().sum().sum():,}"],
            ["Context", context],
        ]
        tbl = Table(data_rows, colWidths=[8*cm, 8*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), CYAN),
            ("TEXTCOLOR",  (0,0), (-1,0), BG),
            ("FONTNAME",   (0,0), (-1,-1), "Courier"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#13131f"), rl_colors.HexColor("#0f0f1a")]),
            ("TEXTCOLOR",  (0,1), (-1,-1), DIM),
            ("GRID",       (0,0), (-1,-1), 0.5, rl_colors.HexColor("#1e2d40")),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(tbl)

        story.append(Paragraph("MODEL PERFORMANCE LEADERBOARD", head_style))
        model_rows = [["MODEL", "ACCURACY", "F1 MACRO", "AUC (TEST)", "CV AUC", "RANK"]]
        sorted_models = sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True)
        for rank, (name, r) in enumerate(sorted_models, 1):
            model_rows.append([
                name, f"{r['accuracy']:.4f}", f"{r['f1']:.4f}", f"{r['auc']:.4f}",
                f"{r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}",
                f"#{rank} {'★ BEST' if name == best_name else ''}",
            ])
        tbl2 = Table(model_rows, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 2*cm])
        tbl2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), CYAN),
            ("TEXTCOLOR",  (0,0), (-1,0), BG),
            ("FONTNAME",   (0,0), (-1,-1), "Courier"),
            ("FONTSIZE",   (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#13131f"), rl_colors.HexColor("#0f0f1a")]),
            ("TEXTCOLOR",  (0,1), (-1,-1), DIM),
            ("GRID",       (0,0), (-1,-1), 0.5, rl_colors.HexColor("#1e2d40")),
            ("PADDING",    (0,0), (-1,-1), 5),
        ]))
        story.append(tbl2)

        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            "Report generated by GCC Risk Intelligence Terminal · Production ML Pipeline · Zero-Leakage Architecture",
            ParagraphStyle("foot", fontSize=7, textColor=rl_colors.HexColor("#4a6070"), fontName="Courier")
        ))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except ImportError:
        return b""


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{C_TEXT_DIM};
            letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.8rem;
            border-bottom:1px solid {C_BORDER}; padding-bottom:0.5rem;">
    ⚙ CONTROL PANEL
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("UPLOAD DATASET (CSV)", type="csv",
                                          help="Max 200MB · CSV format")

RESET_KEYS = [
    "results", "best_model", "best_name", "best_scaler", "best_imputer",
    "best_ohe", "model_feature_cols", "cat_mappings",
    "X", "y", "X_train", "X_test", "y_train", "y_test", "features",
    "train_cat_cols", "train_num_cols",
]
if uploaded_file is not None:
    current_name = uploaded_file.name
    if st.session_state.get("_last_file") != current_name:
        for k in RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state["_last_file"] = current_name

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    df = load_data(file_bytes, uploaded_file.name)
    if df is None:
        st.stop()

    context, auto_target = detect_context_and_target(df)

    strictly_binary = [c for c in df.columns if df[c].nunique() == 2]
    if not strictly_binary:
        st.error("❌ No binary column found.")
        st.stop()

    default_idx = strictly_binary.index(auto_target) if auto_target in strictly_binary else 0

    st.sidebar.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; margin:0.5rem 0;">
        <div style="color:{C_GREEN};">✔ {len(df):,} ROWS LOADED</div>
        <div style="color:{C_TEXT_DIM}; margin-top:0.2rem;">CONTEXT: <span style="color:{C_ACCENT};">{context.upper()}</span></div>
    </div>
    """, unsafe_allow_html=True)

    target_col = st.sidebar.selectbox("TARGET COLUMN:", options=strictly_binary, index=default_idx)

    # ── Top KPI Dashboard ──────────────────────────────────────────────────
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    vc = df[target_col].value_counts()
    class_ratio = f"{vc.iloc[0]/len(df)*100:.0f}% / {vc.iloc[1]/len(df)*100:.0f}%" if len(vc) >= 2 else "N/A"
    num_features = df.shape[1] - 1

    kpi_color_rows = f"{kpi_card('TOTAL RECORDS', f'{len(df):,}', f'{df.shape[1]} columns', '')}"
    kpi_color_rows += kpi_card("FEATURES", str(num_features), "after preprocessing", "kpi-green")
    kpi_color_rows += kpi_card("MISSING VALUES", f"{df.isnull().sum().sum():,}", f"{missing_pct:.1f}% of cells", "kpi-amber" if missing_pct > 0 else "kpi-green")
    kpi_color_rows += kpi_card("CLASS BALANCE", class_ratio, f"target: {target_col}", "kpi-amber" if (vc.max()/vc.min() > 3 if len(vc)>=2 else False) else "")

    st.markdown(f'<div class="kpi-grid">{kpi_color_rows}</div>', unsafe_allow_html=True)

    if "results" in st.session_state:
        results   = st.session_state["results"]
        best_name = st.session_state.get("best_name", max(results, key=lambda k: results[k]["auc"]))
        best_auc  = results[best_name]["auc"]
        best_f1   = results[best_name]["f1"]
        best_acc  = results[best_name]["accuracy"]

        model_kpis  = kpi_card("BEST MODEL", best_name.split()[0].upper(), best_name, "kpi-green")
        model_kpis += kpi_card("AUC-ROC", f"{best_auc:.3f}", "primary metric", "kpi-green" if best_auc >= 0.85 else "kpi-amber")
        model_kpis += kpi_card("F1 MACRO", f"{best_f1:.3f}", "macro averaged", "kpi-green" if best_f1 >= 0.80 else "kpi-amber")
        model_kpis += kpi_card("ACCURACY", f"{best_acc:.3f}", "on test set", "")
        st.markdown(f'<div class="kpi-grid">{model_kpis}</div>', unsafe_allow_html=True)

    # ── PDF Download button in sidebar ────────────────────────────────────
    if "results" in st.session_state:
        pdf_bytes = generate_pdf_report(
            df, st.session_state["results"],
            st.session_state.get("best_name", ""),
            target_col, context
        )
        if pdf_bytes:
            st.sidebar.download_button(
                "📄 EXPORT PDF REPORT", pdf_bytes,
                "GCC_Risk_Report.pdf", "application/pdf",
                use_container_width=True,
            )

    # ── Excel Download ────────────────────────────────────────────────────
    st.sidebar.divider()
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.head(10_000).to_excel(writer, sheet_name="Raw Data",  index=False)
        df.describe().to_excel(writer,   sheet_name="Statistics")
        if "results" in st.session_state:
            pd.DataFrame([
                {"Model": n, "Accuracy": r["accuracy"], "F1": r["f1"], "AUC": r["auc"]}
                for n, r in st.session_state["results"].items()
            ]).to_excel(writer, sheet_name="Model Results", index=False)
    buffer.seek(0)
    st.sidebar.download_button("📊 EXPORT EXCEL REPORT", buffer, "GCC_Risk_Report.xlsx", use_container_width=True)

    # ── TABS ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊  DATA INSPECTOR", "⚠  RISK SCANNER", "🤖  MODEL ARENA",
        "🔮  LIVE PREDICTOR", "🧠  SHAP EXPLAINER", "📘  GUIDE", "💬  CHAT WITH DATA"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA INSPECTOR
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        section_header("◈", "DEEP DATA INSPECTOR")

        total_cells = df.shape[0] * df.shape[1]
        view_mode = st.radio("VIEW MODE:", ["FULL DATA", "MISSING ROWS ONLY"], horizontal=True)

        if total_cells > 200_000:
            st.warning("Large dataset — cell styling disabled for performance.")
            st.dataframe(df, use_container_width=True, height=380)
        elif view_mode == "FULL DATA":
            st.dataframe(df.style.highlight_null(color="#ff336622"), use_container_width=True, height=380)
        else:
            missing = df[df.isnull().any(axis=1)]
            if not missing.empty:
                st.dataframe(missing.style.highlight_null(color="#ff3366"), use_container_width=True)
            else:
                st.success("✔ NO MISSING VALUES DETECTED")

        c1, c2 = st.columns([1, 1])
        with c1:
            section_header("◈", "DESCRIPTIVE STATISTICS")
            st.dataframe(df.describe().T.style.format("{:.4f}", na_rep="—"), use_container_width=True)
        with c2:
            section_header("◈", "COLUMN DISTRIBUTION")
            col_viz = st.selectbox("SELECT COLUMN:", df.columns[:30], key="dist_col")
            is_num  = df[col_viz].dtype in ["int64", "float64"]
            data_s  = df[col_viz].sample(5000) if len(df) > 5000 else df[col_viz]
            try:
                st.pyplot(plot_distribution_premium(data_s, col_viz, is_num))
            except Exception:
                st.error("Cannot plot this column.")

        section_header("◈", "TARGET CLASS DISTRIBUTION")
        vc = df[target_col].value_counts()
        fig_tgt, ax_tgt = terminal_fig((7, 2.5))
        bar_cols = [C_GREEN, C_RED][:len(vc)]
        bars = ax_tgt.bar(vc.index.astype(str), vc.values, color=bar_cols, edgecolor="none", width=0.4)
        for bar, val, col in zip(bars, vc.values, bar_cols):
            ax_tgt.fill_between(
                [bar.get_x(), bar.get_x() + bar.get_width()],
                [0, 0], [val*0.85, val*0.85], alpha=0.2, color=col
            )
            ax_tgt.text(bar.get_x() + bar.get_width()/2, val + vc.max()*0.02,
                        f"{val:,}\n({val/len(df)*100:.1f}%)",
                        ha="center", color=col, fontsize=8, fontfamily="monospace")
        ax_tgt.set_xlabel("CLASS", color=C_TEXT_DIM, fontsize=7)
        ax_tgt.set_ylabel("COUNT", color=C_TEXT_DIM, fontsize=7)
        ax_tgt.set_title(f"TARGET: {target_col.upper()}", color=C_ACCENT, fontsize=8,
                         fontfamily="monospace", fontweight="bold")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig_tgt)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — RISK SCANNER
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        section_header("◈", "AUTOMATED RISK SCANNER — IQR OUTLIER DETECTION")
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.68rem; color:{C_TEXT_DIM}; margin-bottom:0.8rem;">METHOD: Interquartile Range (Q1 − 1.5×IQR, Q3 + 1.5×IQR) · Robust to skewed financial distributions</div>', unsafe_allow_html=True)

        num_scan = [c for c in df.select_dtypes(include="number").columns if c != target_col][:15]
        total_outliers = 0
        risk_summary   = []

        for col in num_scan:
            q1  = df[col].quantile(0.25)
            q3  = df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            out   = df[(df[col] < lo) | (df[col] > hi)]
            count = len(out)
            pct   = count / len(df) * 100
            total_outliers += count
            risk_summary.append({"Feature": col.upper(), "Outliers": count, "Pct": pct, "Lo": lo, "Hi": hi})

            icon  = "⚠" if count else "✔"
            label = f"{icon}  {col.upper()} — {count:,} outliers ({pct:.1f}%)" if count else f"{icon}  {col.upper()} — STABLE"
            with st.expander(label, expanded=(0 < count < 500)):
                cc1, cc2 = st.columns([1, 2])
                with cc1:
                    if count:
                        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; color:{C_RED}; font-weight:700;">RISK DETECTED</div>', unsafe_allow_html=True)
                        st.metric("OUTLIERS", f"{count:,}", delta=f"+{pct:.1f}% of rows", delta_color="inverse")
                        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:{C_TEXT_DIM};">BOUNDS: [{lo:.2f}, {hi:.2f}]</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; color:{C_GREEN}; font-weight:700;">STABLE</div>', unsafe_allow_html=True)
                        st.metric("OUTLIERS", 0)
                with cc2:
                    fig_b, ax_b = terminal_fig((6, 1.8))
                    plot_d = df[col].sample(5000) if len(df) > 5000 else df[col]
                    bp = ax_b.boxplot(plot_d.dropna(), vert=False, patch_artist=True,
                                      flierprops=dict(marker=".", color=C_RED, alpha=0.5, markersize=3),
                                      medianprops=dict(color=C_ACCENT, lw=2),
                                      boxprops=dict(facecolor=C_RED+"33" if count else C_GREEN+"22",
                                        edgecolor=C_RED if count else C_GREEN),
                                      whiskerprops=dict(color=C_TEXT_DIM),
                                      capprops=dict(color=C_TEXT_DIM))
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig_b)

        # Risk summary table
        section_header("◈", f"SCAN SUMMARY — {total_outliers:,} TOTAL OUTLIER RECORDS")
        rdf = pd.DataFrame(risk_summary)
        rdf["STATUS"] = rdf["Outliers"].apply(lambda x: "⚠ RISK" if x > 0 else "✔ STABLE")
        st.dataframe(
            rdf.rename(columns={"Feature": "FEATURE", "Outliers": "OUTLIERS",
                                  "Pct": "PCT %", "Lo": "IQR LOW", "Hi": "IQR HIGH"})
            .style.format({"PCT %": "{:.2f}", "IQR LOW": "{:.2f}", "IQR HIGH": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — ML MODEL ARENA
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        section_header("◈", "ML MODEL ARENA — ZERO-LEAKAGE TRAINING PIPELINE")

        if st.button("🚀  INITIATE TRAINING SEQUENCE", type="primary", use_container_width=True):
            prog = st.progress(0, "Initialising…")
            with st.spinner("Preprocessing data…"):
                try:
                    X_raw, y, cat_mappings = preprocess_data(df, target_col)
                    prog.progress(10, "Preprocessing ✓")
                except Exception as e:
                    st.error(f"Preprocessing error: {e}"); st.stop()

            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.2, random_state=42, stratify=y
                )
            except Exception as e:
                st.error(f"Split failed: {e}"); st.stop()

            cc = y_train.value_counts()
            is_imbalanced = len(cc) >= 2 and (cc.max() / cc.min()) > 5

            try:
                results, ohe, imputer, scaler, feat_cols = train_all_models(
                    X_train_raw, X_test_raw, y_train, y_test, prog
                )
            except ValueError as e:
                st.error(str(e)); st.stop()
            except Exception as e:
                st.error(f"Training error: {e}"); st.stop()

            st.session_state.update({
                "results": results, "X": X_raw, "y": y,
                "X_train": X_train_raw, "X_test": X_test_raw,
                "y_train": y_train, "y_test": y_test,
                "features": feat_cols, "cat_mappings": cat_mappings,
                "model_feature_cols": feat_cols,
                "best_ohe": ohe, "best_imputer": imputer, "best_scaler": scaler,
                "train_cat_cols": X_train_raw.select_dtypes(include="object").columns.tolist(),
                "train_num_cols": X_train_raw.select_dtypes(include="number").columns.tolist(),
            })

            if is_imbalanced:
                ratio = cc.max() / cc.min()
                st.warning(f"⚠ Imbalanced dataset (ratio ≈ {ratio:.0f}:1). Auto-handled via class_weight & scale_pos_weight. Evaluate using F1 & AUC, not Accuracy.")
            prog.empty()
            st.success("✔ ALL 3 MODELS TRAINED SUCCESSFULLY")
            st.rerun()

        if "results" in st.session_state:
            results   = st.session_state["results"]
            y_test    = st.session_state["y_test"]
            features  = st.session_state["features"]
            best_name = max(results, key=lambda k: results[k]["auc"])
            st.session_state["best_model"] = results[best_name]["model"]
            st.session_state["best_name"]  = best_name

            # ── Leaderboard ──
            section_header("◈", f"MODEL LEADERBOARD — WINNER: {best_name.upper()}")
            for rank, (name, r) in enumerate(sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True), 1):
                winner_class = "winner" if name == best_name else ""
                trophy = "🏆" if name == best_name else f"#{rank}"
                st.markdown(f"""
                <div class="leaderboard-row {winner_class}">
                    <span class="lb-rank">{trophy}</span>
                    <span class="lb-name">{name.upper()}</span>
                    <span style="color:{C_TEXT_DIM}; font-size:0.68rem; flex:1;">
                        ACC {r['accuracy']:.3f} &nbsp;·&nbsp; F1 {r['f1']:.3f} &nbsp;·&nbsp;
                        CV {r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}
                    </span>
                    <span class="lb-metric">AUC {r['auc']:.3f}</span>
                    {get_badge_html(r['auc'])}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Charts ──
            c1, c2 = st.columns([1.1, 0.9])
            with c1:
                section_header("◈", "ROC CURVE COMPARISON")
                st.pyplot(plot_roc_premium(results, y_test))
            with c2:
                section_header("◈", f"CONFUSION MATRIX — {best_name.upper()}")
                st.pyplot(plot_confusion_premium(y_test, results[best_name]["y_pred"],
                                                  f"BEST MODEL: {best_name.upper()}"))

            section_header("◈", "FEATURE IMPORTANCE")
            fi_fig = plot_feature_importance_premium(results[best_name]["model"], features, best_name)
            if fi_fig:
                st.pyplot(fi_fig)

            with st.expander("📋  FULL CLASSIFICATION REPORT"):
                report = classification_report(y_test, results[best_name]["y_pred"], output_dict=True)
                st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}", na_rep="—"),
                             use_container_width=True)
        else:
            st.info("👆  Click INITIATE TRAINING SEQUENCE to begin.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — LIVE PREDICTOR
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        section_header("◈", "LIVE RISK PREDICTOR — REAL-TIME INFERENCE ENGINE")

        if "best_model" not in st.session_state:
            st.warning("⚠ Train models first in the MODEL ARENA tab.")
        else:
            model        = st.session_state["best_model"]
            scaler       = st.session_state["best_scaler"]
            imp_live     = st.session_state["best_imputer"]
            ohe_live     = st.session_state["best_ohe"]
            X_raw        = st.session_state["X"]
            cat_mappings = st.session_state.get("cat_mappings", {})
            feat_cols    = st.session_state["model_feature_cols"]
            best_name    = st.session_state["best_name"]

            ohe_generated = (
                set(ohe_live.get_feature_names_out(list(cat_mappings.keys())).tolist())
                if cat_mappings else set()
            )
            numeric_feats = [c for c in feat_cols if c not in ohe_generated]

            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.7rem; color:{C_TEXT_DIM}; margin-bottom:0.8rem;">ACTIVE MODEL: <span style="color:{C_ACCENT}; font-weight:700;">{best_name.upper()}</span> &nbsp;·&nbsp; Enter values below and click PREDICT</div>', unsafe_allow_html=True)

            raw_input: dict = {}
            ui_cols = st.columns(3)
            idx = 0

            for orig_col, categories in cat_mappings.items():
                with ui_cols[idx % 3]:
                    raw_input[orig_col] = st.selectbox(
                        orig_col.replace("_", " ").upper(),
                        options=categories, key=f"raw_{orig_col}",
                    )
                idx += 1

            for feat in numeric_feats:
                if feat not in X_raw.columns:
                    continue
                with ui_cols[idx % 3]:
                    min_v = float(X_raw[feat].min())
                    max_v = float(X_raw[feat].max())
                    med_v = float(X_raw[feat].median())
                    label = feat.replace("_", " ").upper()
                    if X_raw[feat].nunique() <= 10 and X_raw[feat].dtype in ["int64", "int32"]:
                        raw_input[feat] = st.selectbox(label, sorted(X_raw[feat].unique()), key=f"num_{feat}")
                    else:
                        raw_input[feat] = st.slider(label, min_value=min_v, max_value=max_v, value=med_v, key=f"num_{feat}")
                idx += 1

            if st.button("⚡  EXECUTE PREDICTION", type="primary", use_container_width=True):

                num_row = {f: raw_input[f] for f in numeric_feats if f in raw_input}
                num_df  = pd.DataFrame([num_row])

                if cat_mappings:
                    cat_df  = pd.DataFrame({col: [raw_input[col]] for col in cat_mappings})
                    ohe_arr = ohe_live.transform(cat_df)
                    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe_live.get_feature_names_out(list(cat_mappings.keys())).tolist())
                else:
                    ohe_df = pd.DataFrame()

                combined = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
                for col in feat_cols:
                    if col not in combined.columns:
                        combined[col] = 0
                combined = combined[feat_cols]

                num_live_cols = [c for c in st.session_state.get("train_num_cols", []) if c in combined.columns]
                if num_live_cols:
                    combined[num_live_cols] = imp_live.transform(combined[num_live_cols])

                if scaler and best_name == "Logistic Regression":
                    combined_sc = combined.copy()
                    if num_live_cols:
                        combined_sc[num_live_cols] = scaler.transform(combined[num_live_cols])
                    pred      = model.predict(combined_sc)[0]
                    pred_prob = model.predict_proba(combined_sc)[0]
                else:
                    pred      = model.predict(combined)[0]
                    pred_prob = model.predict_proba(combined)[0]

                confidence = max(pred_prob) * 100
                risk_prob  = float(pred_prob[1]) if len(pred_prob) > 1 else float(pred_prob[0])

                # Result display
                if pred == 1:
                    st.markdown(f"""
                    <div class="pred-terminal pred-danger">
                        🚨 &nbsp; HIGH RISK DETECTED &nbsp; · &nbsp; {confidence:.1f}% CONFIDENCE
                        <div class="conf-bar" style="width:{confidence}%;"></div>
                    </div>""", unsafe_allow_html=True)
                    st.error("ACTION REQUIRED: High-risk profile identified. Immediate review recommended.")
                else:
                    st.markdown(f"""
                    <div class="pred-terminal pred-safe">
                        ✔ &nbsp; LOW RISK — WITHIN SAFE PARAMETERS &nbsp; · &nbsp; {confidence:.1f}% CONFIDENCE
                        <div class="conf-bar" style="width:{confidence}%;"></div>
                    </div>""", unsafe_allow_html=True)
                    st.success("ALL CLEAR: Record within acceptable risk parameters.")

                # Gauge + Bars
                g1, g2 = st.columns([1, 1.2])
                with g1:
                    section_header("◈", "RISK GAUGE")
                    st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
                    st.pyplot(plot_risk_gauge(risk_prob))
                    st.markdown('</div>', unsafe_allow_html=True)
                with g2:
                    section_header("◈", "CLASS PROBABILITY")
                    st.pyplot(plot_prob_bars(pred_prob))

                with st.expander("🔍  ENCODED INPUT SENT TO MODEL"):
                    st.dataframe(combined, use_container_width=True)
                    st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:{C_TEXT_DIM};">Human-readable input encoded to numeric dummies — mirrors training pipeline exactly. No data leakage.</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — SHAP EXPLAINER
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        section_header("◈", "SHAP EXPLAINABILITY ENGINE")
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.68rem; color:{C_TEXT_DIM}; margin-bottom:0.8rem;">SHapley Additive exPlanations — feature-level attribution for every prediction</div>', unsafe_allow_html=True)

        if "best_model" not in st.session_state:
            st.warning("⚠ Train models first.")
        else:
            model     = st.session_state["best_model"]
            X_test_r  = st.session_state["X_test"]
            X_train_r = st.session_state["X_train"]
            best_name = st.session_state["best_name"]
            ohe_s     = st.session_state["best_ohe"]
            imp_s     = st.session_state["best_imputer"]
            sc_s      = st.session_state["best_scaler"]
            cat_s     = st.session_state.get("train_cat_cols", [])
            num_s     = st.session_state.get("train_num_cols", [])

            n_samples = st.slider("SAMPLE SIZE FOR SHAP:", 50, min(500, len(X_test_r)), 100)

            if st.button("⚡  COMPUTE SHAP VALUES", type="primary", use_container_width=True):
                with st.spinner("Computing SHAP values…"):
                    try:
                        X_proc_test  = make_processed(X_test_r.iloc[:n_samples], ohe_s, imp_s, cat_s, num_s)
                        X_proc_train = make_processed(X_train_r, ohe_s, imp_s, cat_s, num_s)

                        if best_name in ("Random Forest", "XGBoost"):
                            explainer = shap.TreeExplainer(model)
                            shap_vals = explainer.shap_values(X_proc_test)
                            sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                            base_val = (
                                explainer.expected_value[1]
                                if isinstance(explainer.expected_value, (list, np.ndarray))
                                else float(explainer.expected_value)
                            )
                        else:
                            X_sc_test  = sc_s.transform(X_proc_test)
                            X_sc_train = sc_s.transform(X_proc_train)
                            explainer  = shap.LinearExplainer(model, X_sc_train)
                            shap_vals  = explainer.shap_values(X_sc_test)
                            sv         = shap_vals
                            base_val   = float(explainer.expected_value)

                        section_header("◈", "FEATURE IMPACT — ALL SAMPLES")
                        plt.figure(figsize=(8, 5))
                        shap.summary_plot(sv, X_proc_test, plot_type="bar", show=False)
                        fig_sum = plt.gcf()
                        fig_sum.patch.set_facecolor(C_BG)
                        plt.tight_layout()
                        st.pyplot(fig_sum)
                        plt.close(fig_sum)

                        if best_name in ("Random Forest", "XGBoost"):
                            section_header("◈", "BEESWARM — DIRECTION OF IMPACT")
                            plt.figure(figsize=(8, 5))
                            shap.summary_plot(sv, X_proc_test, show=False)
                            fig_bw = plt.gcf()
                            fig_bw.patch.set_facecolor(C_BG)
                            plt.tight_layout()
                            st.pyplot(fig_bw)
                            plt.close(fig_bw)

                        section_header("◈", "WATERFALL — SINGLE PREDICTION BREAKDOWN (ROW 0)")
                        try:
                            data_row = (
                                X_proc_test.iloc[0].values
                                if best_name in ("Random Forest", "XGBoost")
                                else sc_s.transform(X_proc_test)[0]
                            )
                            exp = shap.Explanation(
                                values=sv[0], base_values=base_val,
                                data=data_row, feature_names=X_proc_test.columns.tolist(),
                            )
                            plt.figure(figsize=(8, 4))
                            shap.plots.waterfall(exp, show=False)
                            fig_wf = plt.gcf()
                            fig_wf.patch.set_facecolor(C_BG)
                            plt.tight_layout()
                            st.pyplot(fig_wf)
                            plt.close(fig_wf)
                        except Exception as e:
                            st.warning(f"Waterfall skipped: {e}")

                        st.success("✔ SHAP ANALYSIS COMPLETE")
                        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.7rem; color:{C_TEXT_DIM}; margin-top:0.5rem;"> 🔴 Red → pushes toward HIGH RISK &nbsp;·&nbsp; 🔵 Blue → pushes toward LOW RISK &nbsp;·&nbsp; Longer bar = stronger influence</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"SHAP error: {e}")
                        st.info("Try fewer samples or switch to Random Forest / XGBoost.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — GUIDE
    # ════════════════════════════════════════════════════════════════════════
    with tab6:
        section_header("◈", "SYSTEM GUIDE & DOCUMENTATION")

        guides = {
            "Credit Card Fraud": "FRAUD DETECTION ENGINE — Identifies abnormal transaction patterns using ML anomaly scoring.",
            "Customer Churn":    "CHURN PREDICTION MODULE — Quantifies customer defection probability for proactive retention.",
            "Loan Risk":         "CREDIT RISK ANALYSER — Estimates default probability using borrower profile features.",
            "General Finance":   "GENERAL RISK TERMINAL — Flexible binary risk scoring for any financial dataset.",
        }
        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:0.8rem;">
            <div class="kpi-label">ACTIVE MODULE</div>
            <div class="kpi-value" style="font-size:1rem;">{context.upper()}</div>
            <div class="kpi-sub" style="margin-top:0.4rem;">{guides.get(context, guides["General Finance"])}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🤖  THE 3 ML ENGINES"):
            st.markdown("""
            | MODEL | ANALOGY | BEST FOR |
            |---|---|---|
            | **Logistic Regression** | A linear scoring formula | Fast baseline, fully explainable |
            | **Random Forest** | 100 independent experts voting | Robust, handles noise & outliers |
            | **XGBoost** | 100 experts learning from each other's mistakes | Highest accuracy, industry standard |
            """)

        with st.expander("📊  METRICS DECODED"):
            st.markdown("""
            | METRIC | MEANING | WHEN TO TRUST |
            |---|---|---|
            | **Accuracy** | % correct predictions — misleading if imbalanced | Only when classes are balanced |
            | **F1 Macro** | Harmonic mean of Precision & Recall | Always — especially for fraud/churn |
            | **AUC-ROC** | Separability score: 1.0 perfect, 0.5 random | Always — primary selection metric |
            | **CV AUC** | 5-fold cross-validated AUC | Most reliable — use this to compare |
            """)

        with st.expander("🎓  PRODUCTION ML PRACTICES — INTERVIEW GOLD ⭐"):
            st.markdown("""
            1. **Zero Imputation Leakage** — SimpleImputer fitted on X_train only
            2. **Zero OHE Leakage** — OneHotEncoder fitted on X_train only (handle_unknown='ignore')
            3. **Categorical NaN → "MISSING"** — prevents silent all-zero dummy rows
            4. **LabelEncoder for target only** — avoids false ordinal ordering on features
            5. **Imbalanced data** — class_weight='balanced' (LR/RF) + scale_pos_weight (XGB)
            6. **Division-by-zero guard** — imbalance ratio clamped to [1, 100]
            7. **IQR outlier detection** — robust vs mean±2σ for skewed finance data
            8. **Session state reset** — stale models cleared on new file upload
            9. **Binary target filter** — prevents multi-class crash in ROC-AUC/SHAP
            10. **sklearn version compat** — sparse_output vs sparse try/except
            11. **F1 macro not weighted** — weighted hides poor minority-class recall
            12. **5-fold CV AUC displayed** — far more reliable than single split
            13. **Cache keyed on file bytes** — prevents stale cache on same filename
            14. **make_processed at module level** — not redefined on every re-render
            15. **Imputer/Scaler on num_cols only** — OHE columns never passed to imputer
            16. **SHAP without ax=** — compatible with all SHAP library versions
            17. **Risk gauge** — visual probability semicircle for non-technical stakeholders
            18. **PDF report export** — professional deliverable via reportlab
            """)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — CHAT WITH YOUR DATA (GEMINI POWERED & SECURE)
    # ════════════════════════════════════════════════════════════════════════
    with tab7:
        section_header("◈", "CHAT WITH YOUR DATA — GCC INTELLIGENCE LLM")
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.68rem; color:{C_TEXT_DIM}; margin-bottom:1rem;">Ask anything about your dataset in English or Hinglish. Securely powered by Gemini AI.</div>', unsafe_allow_html=True)

        # ── Build data context for AI ──────────────────────────────────────
        def build_data_context(df: pd.DataFrame, target_col: str, context: str, results: dict = None) -> str:
            ctx_parts = []
            ctx_parts.append(f"DATASET CONTEXT: {context}")
            ctx_parts.append(f"TARGET COLUMN: {target_col} (binary: {df[target_col].unique().tolist()})")
            ctx_parts.append(f"SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")
            ctx_parts.append(f"COLUMNS: {', '.join(df.columns.tolist())}")

            # Target distribution
            vc = df[target_col].value_counts()
            ctx_parts.append(f"TARGET DISTRIBUTION: {dict(vc.items())}")
            
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if num_cols:
                stats = df[num_cols[:8]].describe().round(2)
                ctx_parts.append(f"NUMERIC STATS (top 8):\n{stats.to_string()}")

            if results:
                best = max(results, key=lambda k: results[k]["auc"])
                ctx_parts.append(f"BEST ML MODEL: {best} (AUC={results[best]['auc']:.3f})")

            return "\n\n".join(ctx_parts)

        # ── AI Chat function (100% STABLE DYNAMIC GEMINI INTEGRATION) ──────────────
        def ask_ai(question: str, data_context: str, chat_history: list) -> str:
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                return "🚨 **SECURITY ERROR:** GEMINI_API_KEY `.env` file mein nahi mili!"

            try:
                genai.configure(api_key=api_key)
                
                # 🔥 DYNAMIC MODEL FINDER (API khud batayega kaunsa model zinda hai)
                valid_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                if not valid_models:
                    return "🚨 **Error:** Teri API key par koi bhi Text Model active nahi hai."
                
                # Best model select karna
                chosen_model = valid_models[0]
                for m in valid_models:
                    if 'gemini-1.5-flash' in m:
                        chosen_model = m
                        break
                    elif 'gemini-1.0-pro' in m:
                        chosen_model = m
                
                model = genai.GenerativeModel(chosen_model)
                
                formatted_history = []
                for msg in chat_history[-6:]: 
                    role = "user" if msg["role"] == "user" else "model"
                    formatted_history.append({"role": role, "parts": [msg["content"]]})
                
                chat = model.start_chat(history=formatted_history)
                
                # PROMPT FIX: Single line with \n to avoid copy-paste indentation breaks
                full_prompt = f"You are an expert Data Analyst working at a top GCC.\nHere is the context of the dataset currently loaded in the dashboard:\n{data_context}\n\nUser Question: {question}\n\nAnswer the user's question clearly, concisely, and accurately based ONLY on the dataset context above. Use actual numbers. Respond in English or Hinglish depending on how the user asks."

                response = chat.send_message(full_prompt)
                return response.text
                
            except Exception as e:
                return f"🚨 **[API Error]**: {e}\n\nModel tried: {chosen_model if 'chosen_model' in locals() else 'Unknown'}"

       # ── Chat UI ────────────────────────────────────────────────────────
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        results_for_ctx = st.session_state.get("results", None)
        data_ctx = build_data_context(df, target_col, context, results_for_ctx)

        # Dynamic Suggested questions
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.65rem; color:{C_TEXT_DIM}; margin-bottom:0.4rem; letter-spacing:0.1em;">QUICK QUESTIONS:</div>', unsafe_allow_html=True)
        
        dynamic_suggestions = {
            "Customer Churn": ["Why are customers leaving?", "Kaunsi age group mein churn zyada hai?", "Dataset mein anomalies kitni hain?", "Explain the ML model results."],
            "Credit Card Fraud": ["Fraud transactions ka pattern kya hai?", "Which features are most suspicious?", "Total outliers kitne hain?", "Explain the model results."],
            "Loan Risk": ["Kaun log loan default kar rahe hain?", "Which factors predict default the most?", "Risk scanner mein kya anomalies aayi?", "Explain the ML model results."],
            "General Finance": ["Dataset mein key insights kya hain?", "Which features matter the most?", "Are there any major outliers?", "Explain the model results."],
        }
        
        sugg_list = dynamic_suggestions.get(context, dynamic_suggestions["General Finance"])

        sq_cols = st.columns(len(sugg_list))
        for i, (col, q) in enumerate(zip(sq_cols, sugg_list)):
            with col:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    st.session_state["chat_pending"] = q

        st.markdown("<br>", unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="display:flex; justify-content:flex-end; margin-bottom:0.6rem;">
                        <div style="background:{C_CARD}; border:1px solid {C_ACCENT}44; border-radius:8px 8px 2px 8px; padding:0.6rem 0.9rem; max-width:75%; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:{C_ACCENT};">
                            <span style="color:{C_TEXT_DIM}; font-size:0.6rem;">YOU</span><br>{msg["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    formatted = msg["content"].replace("\n", "<br>")
                    st.markdown(f"""
                    <div style="display:flex; justify-content:flex-start; margin-bottom:0.6rem;">
                        <div style="background:{C_SURFACE}; border:1px solid {C_BORDER}; border-left:3px solid {C_GREEN}; border-radius:8px 8px 8px 2px; padding:0.6rem 0.9rem; max-width:85%; font-family:'IBM Plex Sans',sans-serif; font-size:0.8rem; color:{C_TEXT}; line-height:1.6;">
                            <span style="font-family:'IBM Plex Mono',monospace; color:{C_GREEN}; font-size:0.6rem; font-weight:700;">◈ RISK-AI TERMINAL</span><br>{formatted}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 🔥 THE FIX: Input clear karne wala Master Callback Function
        def handle_chat_submit():
            if st.session_state.user_chat_input:
                st.session_state.chat_pending = st.session_state.user_chat_input
                st.session_state.user_chat_input = ""  # Dabba turant khali karo!

        st.markdown("<br>", unsafe_allow_html=True)
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            st.text_input("ASK YOUR QUESTION:", key="user_chat_input", on_change=handle_chat_submit, placeholder="e.g. Age ka standard deviation kya hai?", label_visibility="collapsed")
        with btn_col:
            st.button("SEND ⚡", type="primary", on_click=handle_chat_submit, use_container_width=True)

        # Logic to generate answer
        if "chat_pending" in st.session_state and st.session_state["chat_pending"]:
            pending_q = st.session_state.pop("chat_pending")
            st.session_state["chat_history"].append({"role": "user", "content": pending_q})
            with st.spinner("🧠 AI is analyzing your data..."):
                reply = ask_ai(pending_q, data_ctx, st.session_state["chat_history"][:-1])
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("🗑 CLEAR CONVERSATION", use_container_width=False):
                st.session_state["chat_history"] = []
                st.rerun()

        if not st.session_state["chat_history"]:
            st.markdown(f"""
            <div style="text-align:center; padding:2rem; border:1px dashed {C_BORDER}; border-radius:8px; font-family:'IBM Plex Mono',monospace; color:{C_TEXT_DIM};">
                <div style="font-size:2rem; margin-bottom:0.5rem;">💬</div>
                <div style="font-size:0.75rem; color:{C_ACCENT}; margin-bottom:0.3rem;">LLM ENGINE READY</div>
                <div style="font-size:0.65rem; line-height:1.8;">Secure API integration active. Data context is loaded.</div>
            </div>
            """, unsafe_allow_html=True)