# ======================================================
# GLOBAL CONFIGURATION & STYLING
# ======================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['OMP_NUM_THREADS'] = '1'  # Limit CPU threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
import warnings
warnings.filterwarnings('ignore')
import gc  # Garbage collection

MODE = "streamlit"  # options: "streamlit", "api", "batch"

# Core imports (always needed)
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import uuid
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# Conditional Streamlit imports
if MODE == "streamlit":
    import streamlit as st
    st.set_page_config(
        page_title="ML-TSSP Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # Lazy imports - only load when needed
    @st.cache_resource
    def load_heavy_libraries():
        """Lazy load TensorFlow and other heavy dependencies."""
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')  # Disable GPU
            return tf
        except Exception as e:
            st.warning(f"TensorFlow not loaded: {e}")
            return None
else:
    # Dummy streamlit module for non-streamlit modes
    class DummyStreamlit:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    st = DummyStreamlit()
    
    def load_heavy_libraries():
        """Lazy load TensorFlow for non-streamlit modes."""
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            return tf
        except Exception as e:
            print(f"TensorFlow not loaded: {e}")
            return None

# Optional dependencies - gracefully handle if not available
try:
    import pyomo
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    if MODE == "streamlit":
        st.warning("⚠️ Pyomo not available - optimization features may be limited")
    else:
        print("⚠️ Pyomo not available - optimization features may be limited")

try:
    import cvxpy
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Consistent color scheme and behavior classes
COLORS = {
    "cooperative": "#10b981",
    "uncertain": "#f59e0b",
    "coerced": "#7c3aed",
    "deceptive": "#ef4444",
    "baseline": "#3b82f6",
    "neutral": "#6b7280"
}

BEHAVIOR_CLASSES = ["Cooperative", "Uncertain", "Coerced", "Deceptive"]

TASK_ROSTER = [f"Task {i + 1:02d}" for i in range(20)]

def render_kpi_indicator(title: str, value: float | None, *, reference: float | None = None,
                         suffix: str = "", note: str = "", height: int = 150, key: str | None = None):
    """Plotly-based KPI with hover, zoom, and optional delta comparison."""
    display_value = 0.0 if value is None else float(value)
    indicator_cfg = dict(
        mode="number+delta" if reference is not None else "number",
        value=display_value,
        number={"suffix": suffix, "font": {"size": 30, "color": "#1e3a8a"}},
        title={"text": title, "font": {"size": 12, "color": "#6b7280"}}
    )
    if reference is not None:
        indicator_cfg["delta"] = {
            "reference": reference,
            "valueformat": ".3f",
            "increasing": {"color": "#10b981"},
            "decreasing": {"color": "#ef4444"}
        }
    fig = go.Figure(go.Indicator(**indicator_cfg))
    if value is None:
        fig.add_annotation(text="Awaiting run", x=0.5, y=0.1, showarrow=False,
                           font=dict(size=11, color="#9ca3af"))
    elif note:
        fig.add_annotation(text=note, x=0.5, y=0.1, showarrow=False,
                           font=dict(size=11, color="#4b5563"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=25, b=0),
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, key=key or f"kpi_{title}_{note}")

# ...existing _init_streamlit() function...
def _init_streamlit():
    """Initialize Streamlit config with enhanced typography and styling."""
    # Page config already set at top of file - skip here

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Lato:wght@400;700&family=Roboto+Mono:wght@400;500&display=swap');
    
    * {
        font-family: 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    html, body {
        background: linear-gradient(135deg, #e0e7ff 0%, #dbeafe 50%, #e0f2fe 100%);
        color: #1e293b;
        font-size: 15px;
    }
    
    .main {
        background-image: url('https://img.freepik.com/free-photo/close-up-business-items_23-2147679156.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        position: relative;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.92);
        pointer-events: none;
        z-index: 0;
    }
    
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -0.5px;
        line-height: 1.2;
        color: #1e3a8a;
    }
    
    h2 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.3;
        color: #1e40af;
    }
    
    h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 22px;
        font-weight: 600;
        letter-spacing: -0.2px;
        color: #1e40af;
    }
    
    h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #1e3a8a;
    }
    
    body, p {
        font-family: 'Lato', sans-serif;
        font-size: 15px;
        line-height: 1.6;
        color: #334155;
    }
    
    .metric-value {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #1e3a8a;
    }
    
    .metric-label {
        font-family: 'Lato', sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #475569;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .dashboard-header h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        margin: 0;
        font-size: 40px;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .dashboard-header p {
        font-family: 'Lato', sans-serif;
        margin: 0.8rem 0 0 0;
        font-size: 16px;
        opacity: 0.95;
    }
    
    .control-panel {
        background: linear-gradient(145deg, #064e3b 0%, #047857 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #065f46;
        position: sticky;
        top: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    
    .control-panel-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #6b21a8;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #10b981;
    }
    
    /* Sidebar styling for dark green control panel */
    [data-testid="stSidebar"] {
        background-color: #064e3b;
    }
    
    [data-testid="stSidebar"] label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #6b21a8 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #10b981;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #6b21a8 !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stNumberInput label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: #1a1a1a !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(203, 213, 225, 0.8);
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.98);
    }
    
    .metric-card.success {
        border-left-color: #10b981;
    }
    
    .metric-card.warning {
        border-left-color: #f59e0b;
    }
    
    .metric-card.danger {
        border-left-color: #ef4444;
    }
    
    .kpi-value {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-family: 'Lato', sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-frame {
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(203, 213, 225, 0.8);
        border-top: 4px solid #3b82f6;
    }
    
    .section-header {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        color: #1e3a8a;
        font-size: 22px;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 0.8rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
        text-align: center;
        letter-spacing: -0.3px;
    }
    
    .chart-card {
        background: rgba(255, 255, 255, 0.96);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(203, 213, 225, 0.8);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .chart-card-title {
        font-size: 16px;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .insight-box {
        background: rgba(239, 246, 255, 0.95);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        border: 1px solid rgba(191, 219, 254, 0.8);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        font-size: 15px;
    }
    
    .success-box {
        background: rgba(236, 253, 245, 0.95);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        border: 1px solid rgba(167, 243, 208, 0.8);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        font-size: 15px;
    }
    
    .warning-box {
        background: rgba(255, 251, 235, 0.95);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        border: 1px solid rgba(253, 230, 138, 0.8);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        font-size: 15px;
    }
    
    .error-box {
        background: rgba(254, 242, 242, 0.95);
        backdrop-filter: blur(8px);
        padding: 1.2rem;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        border: 1px solid rgba(254, 202, 202, 0.8);
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        font-size: 15px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px !important;
        text-transform: none !important;
        box-shadow: 0 2px 8px rgba(96, 165, 250, 0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.3) !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }
    
    .stButton button:active {
        transform: scale(0.98) !important;
        box-shadow: 0 2px 6px rgba(96, 165, 250, 0.25) !important;
    }
    
    [data-baseweb="tab"] {
        background: #e5e7eb !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 0.7rem 1.3rem !important;
        margin-right: 0.25rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.25s ease !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 10px rgba(96, 165, 250, 0.25) !important;
        border-color: #60a5fa !important;
    }
    
    [data-baseweb="tab"][aria-selected="false"] {
        background: #e5e7eb !important;
        color: #4b5563 !important;
    }
    
    [data-baseweb="tab"][aria-selected="false"]:hover {
        background: #dbeafe !important;
        color: #3b82f6 !important;
        box-shadow: 0 2px 8px rgba(96, 165, 250, 0.15) !important;
    }
    
    [data-testid="stExpander"] {
        background: rgba(248, 250, 252, 0.95) !important;
        backdrop-filter: blur(8px);
        border-radius: 10px !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="dataframe"] {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        font-size: 14px;
    }
    
    hr {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 2rem 0;
    }
    
    pre {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 10px;
        padding: 1.2rem;
        color: #f3f4f6;
        border: 1px solid #374151;
        overflow-x: auto;
        font-size: 13px;
    }
    
    code {
        color: #f3f4f6;
        font-family: 'Courier New', monospace;
        font-size: 13px;
    }
    
    a {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #1e40af;
        text-decoration: underline;
    }
    
    caption {
        font-size: 13px;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

# ...existing API helper functions...
# Import API functions with graceful fallback
USE_LOCAL_API = False
BACKEND_URL = "http://backend:8000"

try:
    from api import run_optimization as local_run_optimization
    from api import explain_source as local_explain_source
    USE_LOCAL_API = True
except ImportError as e:
    # API module not available - will use fallback functions
    pass
except Exception as e:
    # Any other import error - log but continue
    pass

def run_optimization(payload: dict):
    if USE_LOCAL_API:
        return local_run_optimization(payload)
    else:
        try:
            r = requests.post(f"{BACKEND_URL}/optimize", json=payload, timeout=5)
            r.raise_for_status()
            result = r.json()
            
            # Validate that we got meaningful data
            if not result or not result.get("policies") or not result.get("policies", {}).get("ml_tssp"):
                # API returned but with no data, use fallback
                return _fallback_optimization(payload)
            
            return result
        except:
            # Fallback to local simulation if API fails
            return _fallback_optimization(payload)

def _fallback_optimization(payload: dict):
    """Local fallback optimization when API is unavailable."""
    sources = payload.get("sources", [])
    seed = payload.get("seed", 42)
    rng = np.random.default_rng(seed)
    
    policies = {"ml_tssp": [], "deterministic": [], "uniform": []}
    
    for source in sources:
        features = source.get("features", {})
        tsr = features.get("task_success_rate", 0.5)
        cor = features.get("corroboration_score", 0.5)
        time = features.get("report_timeliness", 0.5)
        handler = features.get("handler_confidence", 0.5)
        dec_score = features.get("deception_score", 0.3)
        ci = features.get("ci_flag", 0)
        
        # Simulate ML reliability scoring using training formula
        reliability = np.clip(
            0.30 * tsr + 0.25 * cor + 0.20 * time + 0.15 * handler
            - 0.15 * dec_score - 0.10 * ci + 0.05 * rng.normal(0, 0.03),
            0.0, 1.0
        )
        
        # Simulate deception scoring (inverse relationship + pattern noise)
        deception = np.clip(
            0.30 * dec_score + 0.25 * ci + 0.20 * (1 - cor) + 0.15 * (1 - handler)
            + 0.10 * rng.beta(2, 5),
            0.0, 1.0
        )
        
        # Determine action based on scores
        recourse = source.get("recourse_rules", {})
        rel_disengage = recourse.get("rel_disengage", 0.35)
        rel_flag = recourse.get("rel_ci_flag", 0.50)
        dec_disengage = recourse.get("dec_disengage", 0.75)
        dec_flag = recourse.get("dec_ci_flag", 0.60)
        
        if deception >= dec_disengage or reliability < rel_disengage:
            action = "disengage"
            task = None
        elif deception >= dec_flag:
            action = "flag_for_ci"
            task = rng.choice(TASK_ROSTER)
        elif reliability < rel_flag:
            action = "flag_and_task"
            task = rng.choice(TASK_ROSTER)
        else:
            action = "task"
            task = rng.choice(TASK_ROSTER)
        
        # Calculate expected risk
        expected_risk = 0.3 * (1 - reliability) + 0.4 * deception + 0.3 * rng.beta(2, 3)
        expected_risk = np.clip(expected_risk, 0.0, 1.0)
        
        # Calculate optimization score
        score = reliability * (1 - deception) * (1 - expected_risk)
        
        policy_item = {
            "source_id": source.get("source_id"),
            "reliability": float(reliability),
            "deception": float(deception),
            "action": action,
            "task": task,
            "expected_risk": float(expected_risk),
            "score": float(score)
        }
        
        policies["ml_tssp"].append(policy_item)
        policies["deterministic"].append(policy_item.copy())
        policies["uniform"].append(policy_item.copy())
    
    # Calculate EMV
    ml_emv = sum(p["score"] for p in policies["ml_tssp"])
    
    return {
        "policies": policies,
        "emv": {
            "ml_tssp": ml_emv,
            "deterministic": ml_emv * 0.85,
            "uniform": ml_emv * 0.70
        },
        "_using_fallback": True  # Flag to indicate fallback was used
    }

def request_shap_explanation(source_payload: dict):
    if USE_LOCAL_API:
        source_id = source_payload.get("source_id", "UNKNOWN")
        features = source_payload.get("features", {})
        
        shap_values = {}
        for behavior in BEHAVIOR_CLASSES:
            behavior_shap = {}
            
            tsr = float(features.get("task_success_rate", 0.5))
            cor = float(features.get("corroboration_score", 0.5))
            time = float(features.get("report_timeliness", 0.5))
            
            if behavior == "Cooperative":
                behavior_shap["task_success_rate"] = tsr * 0.3
                behavior_shap["corroboration_score"] = cor * 0.25
                behavior_shap["report_timeliness"] = time * 0.15
                behavior_shap["reliability_trend"] = (1 - tsr) * -0.05
            elif behavior == "Uncertain":
                behavior_shap["task_success_rate"] = (1 - tsr) * 0.2
                behavior_shap["corroboration_score"] = (1 - cor) * 0.25
                behavior_shap["report_timeliness"] = (1 - time) * 0.15
                behavior_shap["reliability_trend"] = abs(0.5 - tsr) * 0.2
            elif behavior == "Coerced":
                behavior_shap["corroboration_score"] = (1 - cor) * 0.3
                behavior_shap["task_success_rate"] = (1 - tsr) * 0.25
                behavior_shap["report_timeliness"] = (1 - time) * 0.2
                behavior_shap["consistency_volatility"] = abs(0.5 - cor) * 0.15
            elif behavior == "Deceptive":
                behavior_shap["corroboration_score"] = (1 - cor) * 0.35
                behavior_shap["task_success_rate"] = abs(0.7 - tsr) * 0.25
                behavior_shap["reliability_trend"] = (1 - tsr) * 0.2
                behavior_shap["consistency_volatility"] = (1 - cor) * 0.2
            
            shap_values[behavior] = behavior_shap
        
        return {"shap_values": shap_values}
    else:
        r = requests.post(f"{BACKEND_URL}/explain", json=source_payload)
        r.raise_for_status()
        return r.json()

def fetch_gru_drift(source_id: str):
    if USE_LOCAL_API:
        dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
        return [
            {"timestamp": d.isoformat(), "reliability": 0.6 + i*0.02, "deception": 0.3 - i*0.01}
            for i, d in enumerate(dates)
        ]
    else:
        r = requests.get(f"{BACKEND_URL}/drift/{source_id}")
        r.raise_for_status()
        return r.json()

def _check_system_health():
    """Check system health and return status."""
    import sys
    from pathlib import Path
    
    health = {
        'packages_loaded': 0,
        'packages_total': 10,
        'packages_ok': True,
        'api_available': USE_LOCAL_API,
        'models_found': 0,
        'models_expected': 4,
        'models_ok': False,
        'warnings': [],
        'details': {}
    }
    
    # Check core packages
    packages = {
        'streamlit': None,
        'numpy': None,
        'pandas': None,
        'plotly': None,
        'matplotlib': None,
        'xgboost': None,
        'sklearn': None,
        'tensorflow': None,
        'shap': None,
        'pyomo': None
    }
    
    for pkg in packages:
        try:
            if pkg == 'sklearn':
                import sklearn
                packages[pkg] = sklearn.__version__
            else:
                mod = __import__(pkg)
                packages[pkg] = getattr(mod, '__version__', 'installed')
            health['packages_loaded'] += 1
        except ImportError:
            packages[pkg] = 'NOT INSTALLED'
            health['packages_ok'] = False
            health['warnings'].append(f"⚠️ {pkg} not available")
    
    health['details']['packages'] = packages
    health['details']['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Check model files
    model_files = [
        'gru_model_reliability.h5',
        'gru_model_deception.h5',
        'label_encoder.joblib',
        'xgb_classifier_features.json'
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            health['models_found'] += 1
    
    health['models_ok'] = health['models_found'] >= 2  # At least core models
    
    if health['models_found'] < health['models_expected']:
        health['warnings'].append(f"⚠️ Only {health['models_found']}/{health['models_expected']} model files found")
    
    health['details']['use_local_api'] = USE_LOCAL_API
    health['details']['pyomo_available'] = PYOMO_AVAILABLE
    health['details']['cvxpy_available'] = CVXPY_AVAILABLE
    health['details']['shap_available'] = SHAP_AVAILABLE
    
    return health

def _decompose_risk(policy_data):
    """Risk decomposition by behavior class."""
    totals = {b: 0.0 for b in BEHAVIOR_CLASSES}
    for assignment in policy_data or []:
        probs = assignment.get("behavior_probs")
        costs = assignment.get("behavior_costs")
        if isinstance(probs, dict) and isinstance(costs, dict):
            for b in BEHAVIOR_CLASSES:
                p = float(probs.get(b, 0.0))
                c = float(costs.get(b, 0.0))
                totals[b] += p * c
        else:
            r = float(assignment.get("expected_risk", 0))
            totals["Cooperative"] += r * 0.20
            totals["Uncertain"] += r * 0.30
            totals["Coerced"] += r * 0.25
            totals["Deceptive"] += r * 0.25
    return totals

def compute_emv(policy_data):
    """Compute EMV from policy assignments."""
    emv = 0.0
    for assignment in policy_data or []:
        probs = assignment.get("behavior_probs")
        costs = assignment.get("behavior_costs")
        if isinstance(probs, dict) and isinstance(costs, dict):
            for b in BEHAVIOR_CLASSES:
                emv += float(probs.get(b, 0.0)) * float(costs.get(b, 0.0))
        else:
            emv += float(assignment.get("expected_risk", 0.0))
    return emv

def calculate_optimization_score(assignment: dict, source_features: dict = None) -> float:
    """
    Calculate dynamic optimization score (0-100) based on multiple factors.
    
    Args:
        assignment: Assignment dict with expected_risk, task, etc.
        source_features: Optional dict with task_success_rate, corroboration_score, report_timeliness
    
    Returns:
        float: Optimization score (0-100), higher is better
    """
    # Base score from expected risk (inverted: lower risk = higher score)
    expected_risk = float(assignment.get("expected_risk", 0.5))
    risk_score = (1.0 - expected_risk) * 40.0  # 40% weight
    
    # If source features are available, incorporate them
    if source_features:
        tsr = float(source_features.get("task_success_rate", 0.5))
        cor = float(source_features.get("corroboration_score", 0.5))
        tim = float(source_features.get("report_timeliness", 0.5))
        
        tsr_score = tsr * 25.0  # 25% weight
        cor_score = cor * 20.0  # 20% weight
        tim_score = tim * 15.0  # 15% weight
        
        total_score = risk_score + tsr_score + cor_score + tim_score
    else:
        # Without features, scale risk component to 0-100
        total_score = risk_score * 2.5
    
    return round(total_score, 2)


def enforce_assignment_constraints(policy_data, sources_map=None):
    """
    One task per source; randomize task assignment based on probabilities.
    Now also calculates and adds optimization scores.
    
    Args:
        policy_data: List of assignment dictionaries
        sources_map: Optional dict mapping source_id to source data with features
    """
    if not policy_data:
        return []
    seen_sources = set()
    tasks = TASK_ROSTER
    fixed = []
    rng = np.random.default_rng(42)
    
    for a in policy_data:
        sid = a.get("source_id")
        if sid in seen_sources:
            continue
        seen_sources.add(sid)
        new_a = dict(a)
        
        risk = float(a.get("expected_risk", 0.5))
        weights = np.array([1.0 / (1.0 + i * risk) for i in range(len(tasks))])
        weights = weights / weights.sum()
        
        new_a["task"] = rng.choice(tasks, p=weights)
        
        # Calculate optimization score
        source_features = None
        if sources_map and sid in sources_map:
            source_features = sources_map[sid].get("features")
        new_a["score"] = calculate_optimization_score(new_a, source_features)
        
        fixed.append(new_a)
    
    return fixed

# ======================================================
# DECISION INTELLIGENCE HELPER RENDERERS
# (moved above render_streamlit_app to avoid NameErrors)
# ======================================================
def _generate_dynamic_recommendation(ml_emv, risk_reduction, low_risk_count, total_sources, ml_coverage):
    """
    Generate dynamic recommendation based on actual metrics.
    """
    low_risk_pct = (low_risk_count / total_sources * 100) if total_sources > 0 else 0
    
    # Determine recommendation based on metrics
    if risk_reduction > 30 and low_risk_pct > 60:
        recommendation = f"Deploy ML–TSSP policy immediately. Achieves {risk_reduction:.1f}% risk reduction with {low_risk_pct:.0f}% low-risk assignments, demonstrating exceptional operational advantage."
        box_type = "success-box"
    elif risk_reduction > 20 and low_risk_pct > 50:
        recommendation = f"Deploy ML–TSSP policy. Delivers {risk_reduction:.1f}% risk reduction with {low_risk_pct:.0f}% low-risk sources, offering strong operational improvements over baselines."
        box_type = "success-box"
    elif risk_reduction > 10:
        recommendation = f"Consider deploying ML–TSSP policy. Shows {risk_reduction:.1f}% risk reduction, though only {low_risk_pct:.0f}% sources are low-risk. Monitor high-risk assignments closely."
        box_type = "warning-box"
    elif risk_reduction > 0:
        recommendation = f"Exercise caution. ML–TSSP shows modest {risk_reduction:.1f}% improvement with {low_risk_pct:.0f}% low-risk sources. Review source quality and consider additional intelligence before deployment."
        box_type = "warning-box"
    else:
        recommendation = f"Hold deployment. ML–TSSP shows minimal advantage ({risk_reduction:.1f}% improvement). Investigate baseline assumptions and source data quality before proceeding."
        box_type = "error-box"
    
    # Add coverage assessment
    if ml_coverage < 5:
        recommendation += f" Note: Limited task coverage ({ml_coverage} tasks) may indicate resource constraints."
    
    return recommendation, box_type

def _render_strategic_decision_section(sources, ml_policy, ml_emv, risk_reduction):
    st.markdown("""
    <div class="insight-box">
        <strong>📊 Optimization Complete!</strong> Key outcomes from the latest ML–TSSP run.
    </div>
    """, unsafe_allow_html=True)
    
    low_risk_count = len([a for a in ml_policy if a.get("expected_risk", 0) < 0.3])
    ml_coverage = len(set(a.get("task") for a in ml_policy))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_indicator("Total Sources", len(sources), note="All assigned", key="kpi_total_sources_tab0")
    with col2:
        render_kpi_indicator("Risk (EMV)", ml_emv, key="kpi_risk_tab0")
    with col3:
        render_kpi_indicator("Low Risk", low_risk_count, key="kpi_low_risk_tab0")
    with col4:
        render_kpi_indicator("Improvement", risk_reduction, suffix="%", note="Vs baseline", key="kpi_improvement_tab0")
    st.divider()
    
    # Generate dynamic recommendation
    recommendation, box_type = _generate_dynamic_recommendation(ml_emv, risk_reduction, low_risk_count, len(sources), ml_coverage)
    
    st.markdown(f"""
    <div class="{box_type}">
        <p style="margin:0;"><strong>Recommendation:</strong> {recommendation}</p>
    </div>
    """, unsafe_allow_html=True)

def _render_policy_framework_section(ml_policy, det_policy, uni_policy, ml_emv, det_emv, uni_emv):
    if not ml_policy:
        st.info("No ML–TSSP assignments yet. Run the optimizer to populate policy comparisons.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Task Distribution (ML–TSSP)</div>', unsafe_allow_html=True)
        task_counts = pd.Series([a.get("task", "Unassigned") for a in ml_policy]).value_counts()
        if task_counts.empty:
            st.warning("Nothing to display for task distribution.")
        else:
            fig = go.Figure(data=[go.Pie(labels=task_counts.index, values=task_counts.values, hole=.45)])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="policy_task_split")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card"><div class="chart-card-title">⚠️ Risk Distribution (ML–TSSP)</div>', unsafe_allow_html=True)
        bins = {"Low (<0.3)": 0, "Medium (0.3-0.6)": 0, "High (>0.6)": 0}
        for r in [a.get("expected_risk", 0) for a in ml_policy]:
            if r < 0.3:
                bins["Low (<0.3)"] += 1
            elif r < 0.6:
                bins["Medium (0.3-0.6)"] += 1
            else:
                bins["High (>0.6)"] += 1
        if not any(bins.values()):
            st.warning("Nothing to display for risk distribution.")
        else:
            fig = go.Figure(data=[go.Pie(labels=list(bins.keys()), values=list(bins.values()), hole=.45)])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, key="policy_risk_split")
        st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="chart-card"><div class="chart-card-title">🫧 Risk vs Coverage (All Policies)</div>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {"Policy": "ML–TSSP", "Risk": float(ml_emv), "Coverage": len(set(a.get("task") for a in ml_policy)), "Sources": len(ml_policy)},
        {"Policy": "Deterministic", "Risk": float(det_emv), "Coverage": len(set(a.get("task") for a in det_policy)), "Sources": len(det_policy)},
        {"Policy": "Uniform", "Risk": float(uni_emv), "Coverage": len(set(a.get("task") for a in uni_policy)), "Sources": len(uni_policy)},
    ])
    bubble = px.scatter(df, x="Risk", y="Coverage", size="Sources", color="Policy")
    bubble.update_layout(height=360, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(bubble, use_container_width=True, key="policy_bubble")
    st.markdown('</div>', unsafe_allow_html=True)

def _render_comparative_policy_section(results, ml_policy, det_policy, uni_policy, ml_emv, det_emv, uni_emv, risk_reduction):
    """
    Unified comparative policy evaluation section showing executive summary 
    followed by detailed policy breakdowns.
    """
    
    # ========== EXECUTIVE COMPARISON SUMMARY (ALWAYS VISIBLE) ==========
    # Generate dynamic summary based on actual performance
    ml_vs_det = ((det_emv - ml_emv) / det_emv * 100) if det_emv > 0 else 0
    ml_vs_uni = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0
    
    if ml_vs_uni > 25 and ml_vs_det > 15:
        summary_text = f"ML–TSSP achieves substantial risk reduction of {ml_vs_uni:.1f}% vs uniform and {ml_vs_det:.1f}% vs deterministic policies through learned behavioral probabilities and forward-looking reliability forecasts."
    elif ml_vs_uni > 15:
        summary_text = f"ML–TSSP demonstrates {ml_vs_uni:.1f}% lower expected risk compared to uniform baseline by incorporating learned behavioral patterns, validating the ML modeling approach."
    elif ml_vs_uni > 5:
        summary_text = f"ML–TSSP shows moderate improvement ({ml_vs_uni:.1f}% vs uniform) by modeling behavioral uncertainty, though deterministic gap is smaller ({ml_vs_det:.1f}%)."
    else:
        summary_text = f"ML–TSSP shows limited advantage over baselines ({ml_vs_uni:.1f}% vs uniform, {ml_vs_det:.1f}% vs deterministic). Consider reviewing source quality and model assumptions."
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                padding: 1.5rem; border-radius: 12px; border: 1px solid #bfdbfe; 
                margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);'>
        <h4 style='margin: 0 0 0.8rem 0; color: #1e40af; font-size: 18px; font-weight: 700; text-align: center;'>
            📊 Executive Policy Comparison
        </h4>
        <p style='margin: 0; font-size: 13px; color: #475569; text-align: center; line-height: 1.6;'>
            {summary_text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== COMPARISON METRICS (3 COLUMNS) ==========
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); 
                    padding: 1.2rem; border-radius: 10px; border: 1px solid #e5e7eb; 
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0 0 0.8rem 0; font-size: 13px; color: #6b7280; font-weight: 600; 
                      text-transform: uppercase; letter-spacing: 0.5px;'>Expected Risk (EMV)</p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-bottom: 0.6rem;'>
            <span style='color: #3b82f6; font-size: 24px; font-weight: 700;'>🏆 {ml_emv:.3f}</span>
            <span style='font-size: 12px; color: #6b7280; margin-left: 0.3rem;'>ML–TSSP</span>
        </div>
        <div style='margin-bottom: 0.4rem;'>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📐 {det_emv:.3f}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>Deterministic</span>
        </div>
        <div>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📊 {uni_emv:.3f}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>Uniform</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with comp_col2:
        ml_coverage = len(set(a.get("task") for a in ml_policy))
        det_coverage = len(set(a.get("task") for a in det_policy))
        uni_coverage = len(set(a.get("task") for a in uni_policy))
        
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); 
                    padding: 1.2rem; border-radius: 10px; border: 1px solid #e5e7eb; 
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0 0 0.8rem 0; font-size: 13px; color: #6b7280; font-weight: 600; 
                      text-transform: uppercase; letter-spacing: 0.5px;'>Task Coverage</p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-bottom: 0.6rem;'>
            <span style='color: #3b82f6; font-size: 24px; font-weight: 700;'>🏆 {ml_coverage}</span>
            <span style='font-size: 12px; color: #6b7280; margin-left: 0.3rem;'>tasks</span>
        </div>
        <div style='margin-bottom: 0.4rem;'>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📐 {det_coverage}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>tasks</span>
        </div>
        <div>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📊 {uni_coverage}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>tasks</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with comp_col3:
        ml_low_risk = len([a for a in ml_policy if a.get("expected_risk", 0) < 0.3])
        det_low_risk = len([a for a in det_policy if a.get("expected_risk", 0) < 0.3])
        uni_low_risk = len([a for a in uni_policy if a.get("expected_risk", 0) < 0.3])
        
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(8px); 
                    padding: 1.2rem; border-radius: 10px; border: 1px solid #e5e7eb; 
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);'>
            <p style='margin: 0 0 0.8rem 0; font-size: 13px; color: #6b7280; font-weight: 600; 
                      text-transform: uppercase; letter-spacing: 0.5px;'>Low-Risk Sources</p>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-bottom: 0.6rem;'>
            <span style='color: #3b82f6; font-size: 24px; font-weight: 700;'>🏆 {ml_low_risk}</span>
            <span style='font-size: 12px; color: #6b7280; margin-left: 0.3rem;'>sources</span>
        </div>
        <div style='margin-bottom: 0.4rem;'>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📐 {det_low_risk}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>sources</span>
        </div>
        <div>
            <span style='color: #9ca3af; font-size: 16px; font-weight: 500;'>📊 {uni_low_risk}</span>
            <span style='font-size: 11px; color: #9ca3af; margin-left: 0.3rem;'>sources</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========== POLICY DETAIL PANELS (SUB-EXPANDERS) ==========
    st.markdown("""
    <p style='text-align: center; color: #6b7280; font-size: 13px; margin: 1rem 0;'>
        Expand each policy below to view detailed assignments and risk distributions
    </p>
    """, unsafe_allow_html=True)
    
    # Sub-expander 1: ML-TSSP (Recommended)
    with st.expander("🏆 ML–TSSP (Recommended)", expanded=False):
        st.markdown("""
        <div style='background: rgba(239, 246, 255, 0.95); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #3b82f6; margin-bottom: 1rem;'>
            <p style='margin: 0; font-size: 13px; color: #1e40af; line-height: 1.6;'>
                <strong>Optimized task assignments</strong> leveraging behavior probabilities from GRU forecasts 
                to minimize expected operational risk under uncertainty. This policy explicitly models behavioral 
                heterogeneity and recourse costs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if ml_policy:
            # Create display dataframe
            display_df = pd.DataFrame(ml_policy)
            if 'behavior_probs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_probs'], errors='ignore')
            if 'behavior_costs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_costs'], errors='ignore')
            st.dataframe(display_df, use_container_width=True)
            
            # Risk distribution visualization
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="chart-card"><div class="chart-card-title">⚠️ Risk Distribution</div>', unsafe_allow_html=True)
                risk_levels = pd.Series(["Low (<0.3)" if a.get("expected_risk", 0) < 0.3 else "High (>0.6)" if a.get("expected_risk", 0) > 0.6 else "Medium (0.3-0.6)" for a in ml_policy]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45,
                                             marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True, key="ml_tssp_risk_dist")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Task Distribution</div>', unsafe_allow_html=True)
                task_counts = pd.Series([a.get("task", "Unassigned") for a in ml_policy]).value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=task_counts.values, y=task_counts.index, orientation='h',
                                             marker=dict(color='#3b82f6'))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="ml_tssp_task_dist")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No ML–TSSP policy assignments available.")
    
    # Sub-expander 2: Deterministic Baseline
    with st.expander("📐 Deterministic Baseline", expanded=False):
        st.markdown("""
        <div style='background: rgba(249, 250, 251, 0.95); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #9ca3af; margin-bottom: 1rem;'>
            <p style='margin: 0; font-size: 13px; color: #4b5563; line-height: 1.6;'>
                <strong>Ignores behavioral uncertainty</strong> and assumes outcomes are known with certainty. 
                Serves as a control case to demonstrate the value of probabilistic modeling. This policy 
                represents traditional optimization without machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if det_policy:
            display_df = pd.DataFrame(det_policy)
            if 'behavior_probs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_probs'], errors='ignore')
            if 'behavior_costs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_costs'], errors='ignore')
            st.dataframe(display_df, use_container_width=True)
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="chart-card"><div class="chart-card-title">⚠️ Risk Distribution</div>', unsafe_allow_html=True)
                risk_levels = pd.Series(["Low (<0.3)" if a.get("expected_risk", 0) < 0.3 else "High (>0.6)" if a.get("expected_risk", 0) > 0.6 else "Medium (0.3-0.6)" for a in det_policy]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45,
                                             marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True, key="det_risk_dist")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Task Distribution</div>', unsafe_allow_html=True)
                task_counts = pd.Series([a.get("task", "Unassigned") for a in det_policy]).value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=task_counts.values, y=task_counts.index, orientation='h',
                                             marker=dict(color='#9ca3af'))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="det_task_dist")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No deterministic policy assignments available.")
    
    # Sub-expander 3: Uniform Allocation Baseline
    with st.expander("📊 Uniform Allocation Baseline", expanded=False):
        st.markdown("""
        <div style='background: rgba(240, 249, 255, 0.95); padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #6b7280; margin-bottom: 1rem;'>
            <p style='margin: 0; font-size: 13px; color: #374151; line-height: 1.6;'>
                <strong>Assumes equal likelihood of all behaviors</strong> without learning from historical data. 
                Captures uncertainty without ML-driven probabilities, testing the value of learned behavioral 
                patterns. Represents naive uncertainty quantification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if uni_policy:
            display_df = pd.DataFrame(uni_policy)
            if 'behavior_probs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_probs'], errors='ignore')
            if 'behavior_costs' in display_df.columns:
                display_df = display_df.drop(columns=['behavior_costs'], errors='ignore')
            st.dataframe(display_df, use_container_width=True)
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="chart-card"><div class="chart-card-title">⚠️ Risk Distribution</div>', unsafe_allow_html=True)
                risk_levels = pd.Series(["Low (<0.3)" if a.get("expected_risk", 0) < 0.3 else "High (>0.6)" if a.get("expected_risk", 0) > 0.6 else "Medium (0.3-0.6)" for a in uni_policy]).value_counts()
                fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45,
                                             marker=dict(colors=['#10b981', '#f59e0b', '#ef4444']))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True, key="uni_risk_dist")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                st.markdown('<div class="chart-card"><div class="chart-card-title">📊 Task Distribution</div>', unsafe_allow_html=True)
                task_counts = pd.Series([a.get("task", "Unassigned") for a in uni_policy]).value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=task_counts.values, y=task_counts.index, orientation='h',
                                             marker=dict(color='#6b7280'))])
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="uni_task_dist")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No uniform allocation policy assignments available.")

def _render_optimal_policy_section(results):
    st.markdown('<div class="insight-box">Recommended ML–TSSP assignment details.</div>', unsafe_allow_html=True)
    policy = results.get("policies", {}).get("ml_tssp", [])
    if policy:
        st.dataframe(pd.DataFrame(policy))
        risk_levels = pd.Series(["Low" if a.get("expected_risk", 0) < 0.3 else "High" if a.get("expected_risk", 0) > 0.6 else "Medium" for a in policy]).value_counts()
        fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45)])
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key="optimal_policy_risk")

def _render_baseline_policy_section(title, policy_key, results):
    st.markdown(f'<div class="insight-box">{title} breakdown.</div>', unsafe_allow_html=True)
    policy = results.get("policies", {}).get(policy_key, [])
    if policy:
        st.dataframe(pd.DataFrame(policy))
        risk_levels = pd.Series(["Low" if a.get("expected_risk", 0) < 0.3 else "High" if a.get("expected_risk", 0) > 0.6 else "Medium" for a in policy]).value_counts()
        fig = go.Figure(data=[go.Pie(labels=risk_levels.index, values=risk_levels.values, hole=.45)])
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"{policy_key}_risk_split")

def _render_shap_section(num_sources):
    """
    Enhanced SHAP explanations with visualizations, narratives, and source-specific insights.
    Answers: Why does the system trust or distrust this source?
    """
    
    # ========== SOURCE SELECTOR & MODE TOGGLE ==========
    col_selector, col_mode = st.columns([2, 1])
    
    with col_selector:
        if st.session_state.get("results"):
            ml_policy = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
            source_ids = [p["source_id"] for p in ml_policy if "source_id" in p]
            if not source_ids:
                st.warning("No sources available. Please run optimization first.")
                return
        else:
            source_ids = [f"SRC_{i + 1:03d}" for i in range(num_sources)]
        
        selected_source = st.selectbox(
            "Select Source for Explanation",
            source_ids,
            key="shap_source_selector",
            help="Choose a source to view its behavioral attribution"
        )
    
    with col_mode:
        explanation_mode = st.radio(
            "View Mode",
            ["🔍 Explain", "⚖️ Compare"],
            key="shap_mode",
            horizontal=True,
            label_visibility="visible"
        )
    
    st.divider()
    
    # ========== COMPARISON MODE ==========
    if explanation_mode == "⚖️ Compare":
        st.markdown("""
        <div style='background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); 
                    padding: 1rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0; color: #1e40af; font-size: 15px; font-weight: 700;'>
                ⚖️ Side-by-Side Source Comparison
            </h4>
            <p style='margin: 0; font-size: 12px; color: #475569;'>
                Compare behavioral attributions to understand why the model assigns different tasks or risk levels.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            source_a = st.selectbox("Source A", source_ids, key="compare_source_a")
        with comp_col2:
            source_b = st.selectbox("Source B", [s for s in source_ids if s != source_a], key="compare_source_b")
        
        # Render two side-by-side explanations
        st.markdown("### Comparative Analysis")
        comp_col_left, comp_col_right = st.columns(2)
        
        with comp_col_left:
            _render_single_source_shap(source_a, compact=True)
        
        with comp_col_right:
            _render_single_source_shap(source_b, compact=True)
        
        return
    
    # ========== SINGLE SOURCE EXPLANATION MODE ==========
    _render_single_source_shap(selected_source, compact=False)


def _render_single_source_shap(source_id, compact=False):
    """Render SHAP explanation for a single source."""
    
    # Generate source features and behavior prediction
    src_idx = int(source_id.split("_")[1]) - 1
    rng = np.random.default_rng(src_idx + 1)
    
    # Source features
    tsr = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
    cor = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
    tim = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
    rel = float(np.clip(rng.beta(6, 3), 0.0, 1.0))
    
    # Predict behavior (simplified classification)
    risk_score = 1.0 - (tsr * 0.4 + cor * 0.3 + tim * 0.2 + rel * 0.1)
    
    if risk_score < 0.3:
        predicted_behavior = "Cooperative"
        confidence = 0.85 + rng.random() * 0.1
        behavior_color = "#10b981"
    elif risk_score < 0.5:
        predicted_behavior = "Uncertain"
        confidence = 0.70 + rng.random() * 0.15
        behavior_color = "#f59e0b"
    elif risk_score < 0.7:
        predicted_behavior = "Coerced"
        confidence = 0.65 + rng.random() * 0.15
        behavior_color = "#7c3aed"
    else:
        predicted_behavior = "Deceptive"
        confidence = 0.75 + rng.random() * 0.15
        behavior_color = "#ef4444"
    
    # ========== EXPLANATION HEADER ==========
    if not compact:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                    padding: 1.2rem; border-radius: 12px; border: 2px solid {behavior_color}; 
                    margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
            <h4 style='margin: 0 0 0.8rem 0; color: #1e3a8a; font-size: 17px; font-weight: 700;'>
                🧠 Source Explanation — {source_id}
            </h4>
            <div style='display: flex; gap: 2rem; align-items: center;'>
                <div>
                    <p style='margin: 0; font-size: 11px; color: #6b7280; font-weight: 600;'>Predicted Behavior</p>
                    <p style='margin: 0.2rem 0 0 0; font-size: 18px; font-weight: 700; color: {behavior_color};'>
                        {predicted_behavior}
                    </p>
                </div>
                <div>
                    <p style='margin: 0; font-size: 11px; color: #6b7280; font-weight: 600;'>Model Confidence</p>
                    <p style='margin: 0.2rem 0 0 0; font-size: 18px; font-weight: 700; color: #1e40af;'>
                        {confidence:.2f}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #f8fafc; padding: 0.8rem; border-radius: 8px; border-left: 4px solid {behavior_color}; margin-bottom: 1rem;'>
            <h5 style='margin: 0 0 0.3rem 0; color: #1e3a8a; font-size: 14px; font-weight: 700;'>{source_id}</h5>
            <p style='margin: 0; font-size: 11px; color: #6b7280;'>
                <strong style='color: {behavior_color};'>{predicted_behavior}</strong> | Conf: {confidence:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== CALCULATE SHAP VALUES ==========
    # Feature contributions to predicted behavior
    shap_data = []
    
    if predicted_behavior == "Cooperative":
        shap_data = [
            {"feature": "task_success_rate", "value": tsr, "shap": tsr * 0.41, "direction": "increases"},
            {"feature": "corroboration_score", "value": cor, "shap": cor * 0.28, "direction": "increases"},
            {"feature": "report_timeliness", "value": tim, "shap": tim * 0.18, "direction": "increases"},
            {"feature": "reliability_trend", "value": rel, "shap": rel * 0.13, "direction": "increases"},
            {"feature": "response_delay", "value": 1-tim, "shap": -(1-tim) * 0.09, "direction": "reduces"},
        ]
    elif predicted_behavior == "Uncertain":
        shap_data = [
            {"feature": "corroboration_score", "value": cor, "shap": (1-cor) * 0.32, "direction": "reduces"},
            {"feature": "task_success_rate", "value": tsr, "shap": (1-tsr) * 0.25, "direction": "reduces"},
            {"feature": "report_timeliness", "value": tim, "shap": (1-tim) * 0.21, "direction": "reduces"},
            {"feature": "consistency_volatility", "value": abs(0.5-cor), "shap": abs(0.5-cor) * 0.15, "direction": "increases"},
            {"feature": "reliability_trend", "value": rel, "shap": rel * 0.07, "direction": "increases"},
        ]
    elif predicted_behavior == "Coerced":
        shap_data = [
            {"feature": "corroboration_score", "value": cor, "shap": (1-cor) * 0.35, "direction": "reduces"},
            {"feature": "report_timeliness", "value": tim, "shap": (1-tim) * 0.24, "direction": "reduces"},
            {"feature": "consistency_volatility", "value": abs(0.5-cor), "shap": abs(0.5-cor) * 0.20, "direction": "increases"},
            {"feature": "task_success_rate", "value": tsr, "shap": tsr * 0.15, "direction": "increases"},
            {"feature": "reliability_trend", "value": rel, "shap": (1-rel) * 0.06, "direction": "reduces"},
        ]
    else:  # Deceptive
        shap_data = [
            {"feature": "corroboration_score", "value": cor, "shap": (1-cor) * 0.42, "direction": "reduces"},
            {"feature": "reliability_trend", "value": rel, "shap": (1-rel) * 0.28, "direction": "reduces"},
            {"feature": "task_success_rate", "value": tsr, "shap": abs(0.7-tsr) * 0.18, "direction": "increases" if tsr < 0.7 else "reduces"},
            {"feature": "consistency_volatility", "value": abs(0.5-cor), "shap": (1-cor) * 0.12, "direction": "increases"},
        ]
    
    # Sort by absolute SHAP magnitude
    shap_data.sort(key=lambda x: abs(x["shap"]), reverse=True)
    
    # ========== TOP DRIVERS TABLE ==========
    st.markdown("""
    <p style='margin: 1rem 0 0.5rem 0; font-size: 14px; font-weight: 700; color: #1e3a8a;'>
        📊 Top Drivers of This Decision
    </p>
    """, unsafe_allow_html=True)
    
    # Create ranked feature table
    for idx, item in enumerate(shap_data[:5]):  # Top 5 features
        feature_name = item["feature"].replace("_", " ").title()
        shap_val = item["shap"]
        direction = item["direction"]
        
        # Determine impact level
        abs_shap = abs(shap_val)
        if abs_shap > 0.25:
            impact_level = "High"
            impact_color = "#ef4444" if shap_val < 0 else "#10b981"
        elif abs_shap > 0.15:
            impact_level = "Medium"
            impact_color = "#f59e0b"
        else:
            impact_level = "Low"
            impact_color = "#6b7280"
        
        # Direction icon and text
        if direction == "increases":
            direction_icon = "⬆"
            direction_text = "increases trust"
            dir_color = "#10b981"
        else:
            direction_icon = "⬇"
            direction_text = "reduces trust"
            dir_color = "#ef4444"
        
        # Generate explanation
        if feature_name == "Task Success Rate":
            if direction == "increases":
                explanation = "High completion history strongly supports cooperative classification"
            else:
                explanation = "Lower task completion raises concerns about capability or intent"
        elif feature_name == "Corroboration Score":
            if direction == "increases":
                explanation = "Strong corroboration validates information reliability"
            else:
                explanation = "Lack of external validation weakens confidence in reporting"
        elif feature_name == "Report Timeliness":
            if direction == "increases":
                explanation = "Consistent timely reporting indicates operational commitment"
            else:
                explanation = "Recent delays slightly weaken operational confidence"
        elif feature_name == "Reliability Trend":
            if direction == "increases":
                explanation = "Positive historical trajectory supports continued trust"
            else:
                explanation = "Declining reliability pattern suggests increased risk"
        elif feature_name == "Response Delay":
            explanation = "Increasing response times may indicate operational constraints"
        elif feature_name == "Consistency Volatility":
            explanation = "Erratic behavioral patterns increase uncertainty"
        else:
            explanation = f"{feature_name} contributes to behavioral classification"
        
        # Render feature row
        bar_width = min(abs(shap_val) * 200, 100)  # Cap at 100%
        
        st.markdown(f"""
        <div style='background: white; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem; 
                    border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem;'>
                    <span style='background: {impact_color}; color: white; padding: 0.2rem 0.5rem; 
                                 border-radius: 4px; font-size: 10px; font-weight: 700;'>#{idx + 1}</span>
                    <span style='font-size: 13px; font-weight: 600; color: #1e3a8a;'>{feature_name}</span>
                </div>
                <div style='text-align: right;'>
                    <span style='font-size: 11px; color: #6b7280; font-weight: 600;'>{impact_level} Impact</span>
                </div>
            </div>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem;'>
                <span style='font-size: 11px; color: {dir_color}; font-weight: 600;'>
                    {direction_icon} {direction_text}
                </span>
                <span style='font-size: 11px; color: #9ca3af;'>|</span>
                <span style='font-size: 11px; color: #1e40af; font-weight: 700;'>
                    SHAP: {shap_val:+.3f}
                </span>
            </div>
            <div style='background: #f3f4f6; border-radius: 4px; height: 6px; margin-bottom: 0.4rem; overflow: hidden;'>
                <div style='background: {impact_color}; height: 100%; width: {bar_width}%; transition: width 0.3s ease;'></div>
            </div>
            <p style='margin: 0; font-size: 10px; color: #6b7280; line-height: 1.4; font-style: italic;'>
                {explanation}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if not compact:
        st.divider()
        
        # ========== WATERFALL VISUALIZATION ==========
        st.markdown("**📈 Feature Impact Waterfall**")
        
        # Create waterfall chart
        fig = go.Figure()
        
        features_display = [item["feature"].replace("_", " ").title() for item in shap_data[:5]]
        shap_values = [item["shap"] for item in shap_data[:5]]
        colors = ["#10b981" if v > 0 else "#ef4444" for v in shap_values]
        
        fig.add_trace(go.Waterfall(
            x=features_display,
            y=shap_values,
            connector={"line": {"color": "#9ca3af"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#3b82f6"}},
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            showlegend=False,
            xaxis=dict(title="Feature", tickangle=-45),
            yaxis=dict(title="SHAP Contribution")
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"shap_waterfall_{source_id}")
        
        st.divider()
        
        # ========== MODEL INTERPRETATION NARRATIVE ==========
        st.markdown("""
        <p style='margin: 1rem 0 0.5rem 0; font-size: 14px; font-weight: 700; color: #1e3a8a;'>
            💡 Model Interpretation Summary
        </p>
        """, unsafe_allow_html=True)
        
        # Generate narrative based on behavior
        if predicted_behavior == "Cooperative":
            narrative = f"""{source_id} is classified as <strong style='color: #10b981;'>Cooperative</strong> primarily due to strong task performance ({tsr:.2f}) and stable corroboration patterns ({cor:.2f}). 
            The model exhibits high confidence ({confidence:.2f}) in this assessment. Minor risk is introduced by timing variations, but not enough to override positive performance signals."""
        elif predicted_behavior == "Uncertain":
            narrative = f"""{source_id} exhibits <strong style='color: #f59e0b;'>Uncertain</strong> behavioral patterns, characterized by inconsistent corroboration ({cor:.2f}) and variable task performance ({tsr:.2f}). 
            Model confidence ({confidence:.2f}) reflects ambiguity in behavioral signals. Enhanced monitoring recommended to resolve classification uncertainty."""
        elif predicted_behavior == "Coerced":
            narrative = f"""{source_id} shows indicators consistent with <strong style='color: #7c3aed;'>Coerced</strong> behavior, including reduced corroboration ({cor:.2f}) and timing irregularities. 
            Model confidence ({confidence:.2f}) suggests external influence patterns. Recommend heightened verification protocols and alternative source development."""
        else:  # Deceptive
            narrative = f"""{source_id} exhibits strong <strong style='color: #ef4444;'>Deceptive</strong> indicators, primarily driven by poor corroboration ({cor:.2f}) and declining reliability trends. 
            Model confidence ({confidence:.2f}) in deception classification warrants immediate action. Recommend source suspension pending thorough investigation."""
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
            <p style='margin: 0; font-size: 12px; color: #475569; line-height: 1.6;'>
                {narrative}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ========== OPERATIONAL IMPLICATION ==========
        st.markdown("""
        <p style='margin: 1rem 0 0.5rem 0; font-size: 14px; font-weight: 700; color: #1e3a8a;'>
            🎯 Operational Implication
        </p>
        """, unsafe_allow_html=True)
        
        # Determine recommendation
        if predicted_behavior == "Cooperative" and confidence > 0.8:
            recommendation = "✅ <strong>Suitable for high-value tasking</strong> under current risk posture. Source demonstrates consistent reliability patterns."
            rec_color = "#10b981"
        elif predicted_behavior == "Cooperative":
            recommendation = "✅ <strong>Cleared for standard operations</strong>. Monitor for sustained performance before high-value assignments."
            rec_color = "#3b82f6"
        elif predicted_behavior == "Uncertain":
            recommendation = "⚠️ <strong>Limit to low-criticality tasks</strong>. Implement enhanced corroboration requirements until behavioral patterns stabilize."
            rec_color = "#f59e0b"
        elif predicted_behavior == "Coerced":
            recommendation = "⚠️ <strong>Restrict to controlled engagements</strong>. Elevated verification protocols mandatory. Consider source rotation."
            rec_color = "#7c3aed"
        else:  # Deceptive
            recommendation = "🛑 <strong>Immediate suspension recommended</strong>. Deception indicators exceed acceptable thresholds. Initiate counterintelligence review."
            rec_color = "#ef4444"
        
        st.markdown(f"""
        <div style='background: white; padding: 1rem; border-radius: 8px; 
                    border: 2px solid {rec_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            <p style='margin: 0; font-size: 13px; color: #1e3a8a; line-height: 1.6;'>
                {recommendation}
            </p>
        </div>
        """, unsafe_allow_html=True)

def _render_evpi_section(ml_policy, uni_policy):
    ml_risk_map = {a.get("source_id"): float(a.get("expected_risk", 0)) for a in ml_policy}
    uni_risk_map = {a.get("source_id"): float(a.get("expected_risk", 0)) for a in uni_policy}
    evpi_rows = []
    for sid, ml_risk in ml_risk_map.items():
        uniform_risk = uni_risk_map.get(sid, ml_risk)
        evpi_val = max(0.0, uniform_risk - ml_risk)
        # Determine decision state based on EVPI and risk level
        if evpi_val > 0.4:
            decision_state = "Critical Uncertainty - Invest in Vetting"
        elif evpi_val > 0.2:
            decision_state = "Material Uncertainty - Enhanced Monitoring"
        elif evpi_val > 0.1:
            decision_state = "Marginal Uncertainty - Standard Oversight"
        else:
            decision_state = "Well-Characterized - Execute with Confidence"
        
        evpi_rows.append({
            "Source": sid, 
            "EVPI": evpi_val, 
            "Value of Perfect Info": uniform_risk - ml_risk,
            "Decision State": decision_state
        })
    evpi_df = pd.DataFrame(evpi_rows).sort_values("EVPI", ascending=False)
    k1, k2, k3 = st.columns(3)
    max_evpi = evpi_df["EVPI"].max() if not evpi_df.empty else 0.0
    avg_evpi = evpi_df["EVPI"].mean() if not evpi_df.empty else 0.0
    high_uncertainty_threshold = 0.2  # Threshold for material uncertainty requiring action
    high_uncertainty_count = len(evpi_df[evpi_df["EVPI"] > high_uncertainty_threshold])
    pct = (high_uncertainty_count / len(evpi_df) * 100) if len(evpi_df) else 0.0
    
    with k1:
        render_kpi_indicator("🔴 Max EVPI", max_evpi, key="kpi_evpi_max_exp")
    with k2:
        render_kpi_indicator("📊 Portfolio Avg EVPI", avg_evpi, key="kpi_evpi_avg_exp")
    with k3:
        render_kpi_indicator("🎯 Material Uncertainty", pct, suffix="%", note=f"EVPI > {high_uncertainty_threshold}", key="kpi_evpi_high_value_exp")
    
    # Threshold reference guide
    st.markdown("""
    <div style='background: #f8fafc; padding: 0.8rem; border-radius: 6px; border: 1px solid #cbd5e1; margin: 1rem 0 1.5rem 0;'>
        <p style='margin: 0 0 0.5rem 0; font-size: 11px; font-weight: 700; color: #1e3a8a;'>EVPI Threshold Interpretation:</p>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; font-size: 10px;'>
            <div style='background: #fee2e2; padding: 0.4rem; border-radius: 4px; border-left: 3px solid #ef4444;'>
                <strong style='color: #991b1b;'>Critical (>0.40)</strong><br/>
                <span style='color: #7f1d1d;'>Uncertainty dominates decision value—invest in source validation</span>
            </div>
            <div style='background: #fef3c7; padding: 0.4rem; border-radius: 4px; border-left: 3px solid #f59e0b;'>
                <strong style='color: #92400e;'>Material (0.20-0.40)</strong><br/>
                <span style='color: #78350f;'>Uncertainty materially impacts tasking—enhanced collection warranted</span>
            </div>
            <div style='background: #dbeafe; padding: 0.4rem; border-radius: 4px; border-left: 3px solid #3b82f6;'>
                <strong style='color: #1e40af;'>Marginal (0.10-0.20)</strong><br/>
                <span style='color: #1e3a8a;'>Residual uncertainty manageable—maintain standard monitoring</span>
            </div>
            <div style='background: #d1fae5; padding: 0.4rem; border-radius: 4px; border-left: 3px solid #10b981;'>
                <strong style='color: #065f46;'>Minimal (<0.10)</strong><br/>
                <span style='color: #064e3b;'>Well-characterized source—execute with operational confidence</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic EVPI recommendation with sharper value vs uncertainty language
    if avg_evpi > 0.5:
        evpi_rec = f"<strong>Critical Uncertainty Exposure:</strong> Portfolio-average EVPI of {avg_evpi:.3f} indicates that behavioral uncertainty severely constrains decision value. {high_uncertainty_count} sources ({pct:.0f}%) exceed material uncertainty threshold (>{high_uncertainty_threshold}). <strong>Actionable Decision:</strong> Prioritize intensive vetting, multi-source corroboration, and controlled elicitation for top sources (max EVPI: {max_evpi:.3f}) before committing to high-stakes tasking. Expected value gain from perfect information justifies substantial collection investment."
        evpi_box = "warning-box"
    elif avg_evpi > 0.3:
        evpi_rec = f"<strong>Material Uncertainty Opportunity:</strong> Average EVPI of {avg_evpi:.3f} signals actionable value in resolving source ambiguity. {high_uncertainty_count} sources ({pct:.0f}%) warrant enhanced monitoring. <strong>Actionable Decision:</strong> Deploy targeted collection efforts (verification checks, pattern analysis, behavioral profiling) on high-EVPI sources to improve tasking precision and reduce avoidable risk exposure."
        evpi_box = "insight-box"
    elif avg_evpi > 0.1:
        evpi_rec = f"<strong>Residual Uncertainty—Well-Calibrated Model:</strong> Average EVPI of {avg_evpi:.3f} indicates the ML model has effectively extracted signal from available data. {high_uncertainty_count} sources ({pct:.0f}%) retain material uncertainty. <strong>Actionable Decision:</strong> Maintain standard monitoring protocols; focus discretionary collection resources on the {int(evpi_df['EVPI'].quantile(0.75) * 100)}th percentile (EVPI > {evpi_df['EVPI'].quantile(0.75):.3f}) to refine edge-case sources."
        evpi_box = "success-box"
    else:
        evpi_rec = f"<strong>Minimal Uncertainty—Execution Readiness:</strong> Portfolio-average EVPI of {avg_evpi:.3f} confirms sources are well-characterized with minimal residual ambiguity. <strong>Actionable Decision:</strong> Shift focus from collection to operational execution. ML-TSSP policy is information-saturated—additional vetting yields diminishing marginal value. Allocate resources to mission execution rather than incremental source refinement."
        evpi_box = "success-box"
    
    st.markdown(f"""
    <div class="{evpi_box}" style="margin: 1rem 0;">
        <p style="margin:0; line-height: 1.6;">{evpi_rec}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(evpi_df.reset_index(drop=True), use_container_width=True)

def _render_stress_section(ml_policy, ml_emv, det_emv, uni_emv, risk_reduction):
    """
    Decision stress testing demonstrates the robustness of the ML–TSSP policy under adverse 
    reliability and deception scenarios, highlighting conditions under which tasking 
    recommendations become unstable or risk exposure escalates.
    """
    
    # ========== SECTION HEADER WITH PURPOSE ==========
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                padding: 1.2rem; border-radius: 10px; border-left: 4px solid #f59e0b; 
                margin-bottom: 1.5rem;'>
        <h4 style='margin: 0; color: #92400e; font-size: 16px; font-weight: 700;'>
            🔬 Behavioral Uncertainty & Stress Analysis (What-If)
        </h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 12px; color: #78350f; line-height: 1.6;'>
            Explore how tasking recommendations and risk exposure change under alternative assumptions 
            about source reliability, deception risk, and operational priorities.
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 11px; color: #92400e; font-style: italic;'>
            Stress testing evaluates whether the recommended ML–TSSP policy remains effective when 
            assumptions about source behavior or operational priorities change.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate dynamic stress testing guidance
    if risk_reduction > 25:
        stress_guidance = f"Policy demonstrates strong advantage ({risk_reduction:.1f}% improvement). Stress testing recommended to identify operational boundaries and failure modes under adverse conditions."
    elif risk_reduction > 15:
        stress_guidance = f"Policy shows good performance ({risk_reduction:.1f}% improvement). Use stress testing to assess robustness margins and validate assumptions about source behavior."
    elif risk_reduction > 5:
        stress_guidance = f"Policy advantage is moderate ({risk_reduction:.1f}%). Stress testing critical to determine if limited margin holds under uncertainty escalation or assumption violations."
    else:
        stress_guidance = f"Policy shows minimal advantage ({risk_reduction:.1f}%). Stress testing essential to understand when and why ML–TSSP may underperform—consider baseline policies under high uncertainty."
    
    st.markdown(f"""
    <div class="insight-box" style="margin-bottom: 1rem;">
        <p style="margin:0;"><strong>Testing Priority:</strong> {stress_guidance}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== TWO-COLUMN LAYOUT: CONTROLS + RESULTS ==========
    control_col, results_col = st.columns([1, 2])
    
    # ========== LEFT: CONTROL PANEL ==========
    with control_col:
        with st.expander("⚙️ STRESS PARAMETERS", expanded=True):
            # Scenario Presets
            st.markdown("**📋 Scenario Presets**")
            scenario_preset = st.radio(
            "Quick scenarios",
            ["Normal Intelligence Environment", "High Threat Environment", "Denied/Contested Environment"],
            horizontal=True,
            key="stress_scenario",
            help="Pre-configured stress scenarios"
        )
        
        st.divider()
        
        # Core Levers
        st.markdown("**🎚️ Core Stress Levers**")
        
        # Reliability degradation/uplift
        if scenario_preset == "High Threat Environment":
            rel_default = -0.20
        elif scenario_preset == "Denied/Contested Environment":
            rel_default = 0.10
        else:
            rel_default = 0.0
        
        reliability_shift = st.slider(
            "Reliability Shift",
            min_value=-0.30,
            max_value=0.15,
            value=rel_default,
            step=0.05,
            format="%.2f",
            key="rel_shift",
            help="Global reliability degradation (-) or uplift (+)"
        )
        
        # Deception risk inflation
        if scenario_preset == "High Threat Environment":
            dec_default = 0.40
        elif scenario_preset == "Denied/Contested Environment":
            dec_default = 0.0
        else:
            dec_default = 0.0
        
        deception_inflation = st.slider(
            "Deception Risk Inflation",
            min_value=0.0,
            max_value=0.50,
            value=dec_default,
            step=0.05,
            format="%.2f",
            key="dec_inflation",
            help="Simulates adversarial pressure or source compromise"
        )
        
        # Risk tolerance
        if scenario_preset == "High Threat Environment":
            risk_default = 0.3
        elif scenario_preset == "Denied/Contested Environment":
            risk_default = 0.7
        else:
            risk_default = 0.5
        
        risk_tolerance = st.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=risk_default,
            step=0.1,
            format="%.1f",
            key="risk_tol",
            help="Conservative (0) ↔ Aggressive (1)"
        )
        
        # Coverage priority
        if scenario_preset == "Denied/Contested Environment":
            cov_default = 0.7
        else:
            cov_default = 0.5
        
            coverage_priority = st.slider(
                "Coverage Priority",
                min_value=0.0,
                max_value=1.0,
                value=cov_default,
                step=0.1,
                format="%.1f",
                key="cov_priority",
                help="Low (0) ↔ High coverage (1)"
            )
        
        st.divider()
        
        # Execute button (outside expander)
        execute_stress = st.button(
            "▶ Execute Stress Test",
            type="primary",
            use_container_width=True,
            key="stress_execute"
        )
    
    # ========== RIGHT: RESULTS & VISUALIZATIONS ==========
    with results_col:
        if not execute_stress and "stress_results" not in st.session_state:
            st.markdown("""
            <div style='background: #f9fafb; padding: 3rem; border-radius: 10px; 
                        border: 2px dashed #d1d5db; text-align: center;'>
                <p style='margin: 0; font-size: 14px; color: #6b7280; font-weight: 600;'>
                    ⏳ Configure parameters and execute stress test
                </p>
                <p style='margin: 0.5rem 0 0 0; font-size: 12px; color: #9ca3af;'>
                    Results will appear here
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # ========== COMPUTE STRESS SCENARIOS ==========
        if execute_stress or "stress_results" in st.session_state:
            # Store parameters
            st.session_state["stress_results"] = {
                "reliability_shift": reliability_shift,
                "deception_inflation": deception_inflation,
                "risk_tolerance": risk_tolerance,
                "coverage_priority": coverage_priority
            }
            
            # Compute stress scenarios
            stress_scenarios = []
            
            # Generate response curve data (vary one parameter at a time)
            rel_sweep = np.linspace(-0.30, 0.15, 12)
            dec_sweep = np.linspace(0.0, 0.50, 12)
            
            baseline_emv = ml_emv
            
            for rel_shift in rel_sweep:
                # Adjust EMV based on reliability shift
                # Lower reliability = higher risk
                emv_adj = baseline_emv * (1 + abs(rel_shift) * 0.8) if rel_shift < 0 else baseline_emv * (1 - rel_shift * 0.3)
                
                # Calculate derived metrics
                num_sources = len(ml_policy)
                low_risk_sources = max(1, int(num_sources * (0.3 - rel_shift)))
                coverage = min(100, int((70 + rel_shift * 50) * (1 + coverage_priority * 0.3)))
                policy_changes = max(0, int(abs(rel_shift) * 15))
                worst_risk = min(0.95, baseline_emv + abs(rel_shift) * 0.5)
                
                stress_scenarios.append({
                    "scenario": f"Rel {rel_shift:.2f}",
                    "reliability_shift": rel_shift,
                    "deception_inflation": 0,
                    "emv": emv_adj,
                    "low_risk_sources": low_risk_sources,
                    "coverage": coverage,
                    "policy_changes": policy_changes,
                    "worst_risk": worst_risk
                })
            
            for dec_inf in dec_sweep:
                # Adjust EMV based on deception inflation
                emv_adj = baseline_emv * (1 + dec_inf * 1.2)
                
                num_sources = len(ml_policy)
                low_risk_sources = max(1, int(num_sources * (0.3 - dec_inf * 0.5)))
                coverage = min(100, int((70 - dec_inf * 40) * (1 + coverage_priority * 0.3)))
                policy_changes = max(0, int(dec_inf * 20))
                worst_risk = min(0.95, baseline_emv + dec_inf * 0.6)
                
                stress_scenarios.append({
                    "scenario": f"Dec +{dec_inf:.2f}",
                    "reliability_shift": 0,
                    "deception_inflation": dec_inf,
                    "emv": emv_adj,
                    "low_risk_sources": low_risk_sources,
                    "coverage": coverage,
                    "policy_changes": policy_changes,
                    "worst_risk": worst_risk
                })
            
            # Current scenario with user's settings
            current_emv = baseline_emv * (1 + abs(reliability_shift) * 0.8 + deception_inflation * 1.2)
            current_scenario = {
                "reliability_shift": reliability_shift,
                "deception_inflation": deception_inflation,
                "emv": current_emv,
                "low_risk_sources": max(1, int(len(ml_policy) * (0.3 - reliability_shift - deception_inflation * 0.5))),
                "coverage": int((70 + reliability_shift * 50 - deception_inflation * 40) * (1 + coverage_priority * 0.3)),
                "policy_changes": int(abs(reliability_shift) * 15 + deception_inflation * 20),
                "worst_risk": min(0.95, baseline_emv + abs(reliability_shift) * 0.5 + deception_inflation * 0.6)
            }
            
            st.session_state["current_scenario"] = current_scenario
            
            # ========== OUTPUT METRICS ==========
            st.markdown("**📊 Stress Test Results**")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                delta_emv = current_emv - baseline_emv
                st.metric(
                    "EMV (Operational Risk)",
                    f"{current_emv:.3f}",
                    delta=f"{delta_emv:+.3f}",
                    delta_color="inverse"
                )
            
            with metric_col2:
                st.metric(
                    "Low-Risk Sources",
                    current_scenario["low_risk_sources"],
                    help="Sources below risk threshold"
                )
            
            with metric_col3:
                st.metric(
                    "Task Coverage",
                    f"{current_scenario['coverage']}%",
                    help="Percentage of tasks assigned"
                )
            
            with metric_col4:
                st.metric(
                    "Policy Changes",
                    current_scenario["policy_changes"],
                    delta="vs baseline",
                    help="Number of assignment changes"
                )
            
            st.divider()
            
            # ========== A. EMV RESPONSE CURVE ==========
            st.markdown("**📈 EMV Response Curve (Primary Sensitivity)**")
            
            tab_rel, tab_dec = st.tabs(["Reliability Sensitivity", "Deception Sensitivity"])
            
            with tab_rel:
                rel_scenarios = [s for s in stress_scenarios if s["deception_inflation"] == 0]
                
                fig_rel_curve = go.Figure()
                
                fig_rel_curve.add_trace(go.Scatter(
                    x=[s["reliability_shift"] for s in rel_scenarios],
                    y=[s["emv"] for s in rel_scenarios],
                    mode='lines+markers',
                    name='EMV Response',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=8, color='#1e40af'),
                    hovertemplate='<b>Reliability Shift: %{x:.2f}</b><br>EMV: %{y:.3f}<extra></extra>'
                ))
                
                # Mark current scenario
                fig_rel_curve.add_trace(go.Scatter(
                    x=[reliability_shift],
                    y=[current_emv],
                    mode='markers',
                    name='Current',
                    marker=dict(size=15, color='#ef4444', symbol='star'),
                    hovertemplate='<b>Current Scenario</b><br>Shift: %{x:.2f}<br>EMV: %{y:.3f}<extra></extra>'
                ))
                
                # Add baseline line
                fig_rel_curve.add_hline(
                    y=baseline_emv,
                    line_dash='dash',
                    line_color='#10b981',
                    opacity=0.6,
                    annotation_text="Baseline EMV"
                )
                
                fig_rel_curve.update_layout(
                    height=300,
                    xaxis_title="Reliability Shift",
                    yaxis_title="EMV (Expected Risk)",
                    hovermode='x',
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='white',
                    plot_bgcolor='#fafafa',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_rel_curve, use_container_width=True, key="stress_rel_curve")
            
            with tab_dec:
                dec_scenarios = [s for s in stress_scenarios if s["reliability_shift"] == 0]
                
                fig_dec_curve = go.Figure()
                
                fig_dec_curve.add_trace(go.Scatter(
                    x=[s["deception_inflation"] for s in dec_scenarios],
                    y=[s["emv"] for s in dec_scenarios],
                    mode='lines+markers',
                    name='EMV Response',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=8, color='#991b1b'),
                    hovertemplate='<b>Deception Inflation: %{x:.2f}</b><br>EMV: %{y:.3f}<extra></extra>'
                ))
                
                # Mark current scenario
                fig_dec_curve.add_trace(go.Scatter(
                    x=[deception_inflation],
                    y=[current_emv],
                    mode='markers',
                    name='Current',
                    marker=dict(size=15, color='#7c3aed', symbol='star'),
                    hovertemplate='<b>Current Scenario</b><br>Inflation: %{x:.2f}<br>EMV: %{y:.3f}<extra></extra>'
                ))
                
                # Add baseline line
                fig_dec_curve.add_hline(
                    y=baseline_emv,
                    line_dash='dash',
                    line_color='#10b981',
                    opacity=0.6,
                    annotation_text="Baseline EMV"
                )
                
                fig_dec_curve.update_layout(
                    height=300,
                    xaxis_title="Deception Risk Inflation",
                    yaxis_title="EMV (Expected Risk)",
                    hovermode='x',
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='white',
                    plot_bgcolor='#fafafa',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_dec_curve, use_container_width=True, key="stress_dec_curve")
            
            st.divider()
            
            # ========== B. COVERAGE VS RISK FRONTIER ==========
            col_scatter, col_stability = st.columns(2)
            
            with col_scatter:
                st.markdown("**🎯 Coverage vs Risk Frontier**")
                
                fig_frontier = go.Figure()
                
                # Plot all scenarios
                colors_map = {
                    "Normal Intelligence Environment": "#10b981",
                    "High Threat Environment": "#ef4444",
                    "Denied/Contested Environment": "#3b82f6"
                }
                
                for s in stress_scenarios:
                    scenario_type = "Normal Intelligence Environment" if s["reliability_shift"] == 0 and s["deception_inflation"] == 0 else "High Threat Environment" if s["reliability_shift"] < 0 or s["deception_inflation"] > 0 else "Denied/Contested Environment"
                    
                    fig_frontier.add_trace(go.Scatter(
                        x=[s["emv"]],
                        y=[s["coverage"]],
                        mode='markers',
                        marker=dict(size=6, color=colors_map.get(scenario_type, "#6b7280"), opacity=0.3),
                        showlegend=False,
                        hovertemplate=f'<b>{s["scenario"]}</b><br>Risk: %{{x:.3f}}<br>Coverage: %{{y}}%<extra></extra>'
                    ))
                
                # Highlight current scenario
                fig_frontier.add_trace(go.Scatter(
                    x=[current_emv],
                    y=[current_scenario["coverage"]],
                    mode='markers',
                    name='Current',
                    marker=dict(size=15, color='#7c3aed', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate='<b>Current Scenario</b><br>Risk: %{x:.3f}<br>Coverage: %{y}%<extra></extra>'
                ))
                
                # Highlight baseline
                fig_frontier.add_trace(go.Scatter(
                    x=[baseline_emv],
                    y=[70],
                    mode='markers',
                    name='Baseline',
                    marker=dict(size=12, color='#10b981', symbol='diamond'),
                    hovertemplate='<b>Baseline</b><br>Risk: %{x:.3f}<br>Coverage: %{y}%<extra></extra>'
                ))
                
                fig_frontier.update_layout(
                    height=300,
                    xaxis_title="Expected Risk (EMV)",
                    yaxis_title="Coverage (%)",
                    hovermode='closest',
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='white',
                    plot_bgcolor='#fafafa',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig_frontier, use_container_width=True, key="stress_frontier")
            
            # ========== C. ASSIGNMENT STABILITY BAR ==========
            with col_stability:
                st.markdown("**🔄 Assignment Stability**")
                
                # Create stability comparison data
                stability_data = []
                
                for s in [s for s in stress_scenarios if s["scenario"] in [f"Rel {reliability_shift:.2f}", f"Dec +{deception_inflation:.2f}", "Rel 0.00"]]:
                    stability_data.append({
                        "Scenario": s["scenario"],
                        "Changes": s["policy_changes"]
                    })
                
                # Add current scenario
                stability_data.append({
                    "Scenario": "Current",
                    "Changes": current_scenario["policy_changes"]
                })
                
                fig_stability = go.Figure()
                
                colors = ['#10b981' if d["Changes"] <= 5 else '#f59e0b' if d["Changes"] <= 10 else '#ef4444' for d in stability_data]
                
                fig_stability.add_trace(go.Bar(
                    x=[d["Scenario"] for d in stability_data],
                    y=[d["Changes"] for d in stability_data],
                    marker=dict(color=colors),
                    text=[d["Changes"] for d in stability_data],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Policy Changes: %{y}<extra></extra>'
                ))
                
                fig_stability.update_layout(
                    height=300,
                    xaxis_title="Scenario",
                    yaxis_title="Task Changes",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='white',
                    plot_bgcolor='#fafafa',
                    showlegend=False
                )
                
                st.plotly_chart(fig_stability, use_container_width=True, key="stress_stability")
            
            st.divider()
            
            # ========== D. RISK STATE DISTRIBUTION ==========
            st.markdown("**📊 Risk State Distribution**")
            
            # Calculate risk state distributions for different scenarios
            risk_dist_data = []
            
            for scenario_name in ["Normal Intelligence Environment", "High Threat Environment", "Denied/Contested Environment", "Current"]:
                if scenario_name == "Normal Intelligence Environment":
                    low, med, high, crit = 8, 6, 3, 1
                elif scenario_name == "High Threat Environment":
                    low, med, high, crit = 3, 5, 7, 4
                elif scenario_name == "Denied/Contested Environment":
                    low, med, high, crit = 12, 4, 2, 0
                else:  # Current
                    # Derive from current scenario
                    total = len(ml_policy)
                    low = current_scenario["low_risk_sources"]
                    crit = max(0, int(abs(reliability_shift) * 5 + deception_inflation * 8))
                    high = max(0, int(total * 0.2 - low * 0.1))
                    med = max(0, total - low - high - crit)
                
                risk_dist_data.append({
                    "Scenario": scenario_name,
                    "Low": low,
                    "Medium": med,
                    "High": high,
                    "Critical": crit
                })
            
            fig_risk_dist = go.Figure()
            
            fig_risk_dist.add_trace(go.Bar(
                name='Low',
                x=[d["Scenario"] for d in risk_dist_data],
                y=[d["Low"] for d in risk_dist_data],
                marker=dict(color='#10b981'),
                hovertemplate='<b>%{x}</b><br>Low Risk: %{y}<extra></extra>'
            ))
            
            fig_risk_dist.add_trace(go.Bar(
                name='Medium',
                x=[d["Scenario"] for d in risk_dist_data],
                y=[d["Medium"] for d in risk_dist_data],
                marker=dict(color='#f59e0b'),
                hovertemplate='<b>%{x}</b><br>Medium Risk: %{y}<extra></extra>'
            ))
            
            fig_risk_dist.add_trace(go.Bar(
                name='High',
                x=[d["Scenario"] for d in risk_dist_data],
                y=[d["High"] for d in risk_dist_data],
                marker=dict(color='#f97316'),
                hovertemplate='<b>%{x}</b><br>High Risk: %{y}<extra></extra>'
            ))
            
            fig_risk_dist.add_trace(go.Bar(
                name='Critical',
                x=[d["Scenario"] for d in risk_dist_data],
                y=[d["Critical"] for d in risk_dist_data],
                marker=dict(color='#ef4444'),
                hovertemplate='<b>%{x}</b><br>Critical Risk: %{y}<extra></extra>'
            ))
            
            fig_risk_dist.update_layout(
                barmode='stack',
                height=280,
                xaxis_title="Scenario",
                yaxis_title="Source Count",
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig_risk_dist, use_container_width=True, key="stress_risk_dist")
            
            st.divider()
            
            # ========== KEY FINDINGS PANEL ==========
            st.markdown("**🔍 Key Findings**")
            
            # Calculate break-even threshold
            break_even_rel = -baseline_emv / 0.8 if baseline_emv > 0 else 0.45
            break_even_rel = max(-0.30, min(0.15, break_even_rel))
            
            # EMV sensitivity
            emv_range_min = min([s["emv"] for s in stress_scenarios])
            emv_range_max = max([s["emv"] for s in stress_scenarios])
            
            # Find deception threshold
            high_sensitivity_dec = 0.30
            
            # Coverage collapse analysis
            baseline_low_risk = next((s["low_risk_sources"] for s in stress_scenarios if s["reliability_shift"] == 0 and s["deception_inflation"] == 0), 8)
            adversarial_low_risk = next((s["low_risk_sources"] for s in stress_scenarios if s["reliability_shift"] == -0.20 and s["deception_inflation"] == 0), 3)
            
            findings = [
                {
                    "icon": "⚠️",
                    "title": "Break-even reliability threshold",
                    "value": f"{break_even_rel:.2f}",
                    "detail": "Below this level, risk increases sharply",
                    "severity": "high" if abs(reliability_shift - break_even_rel) < 0.05 else "medium"
                },
                {
                    "icon": "📉",
                    "title": "EMV sensitivity band",
                    "value": f"{emv_range_min:.2f} → {emv_range_max:.2f}",
                    "detail": f"High sensitivity to deception inflation above {high_sensitivity_dec:.0%}",
                    "severity": "high" if deception_inflation > high_sensitivity_dec else "low"
                },
                {
                    "icon": "🎯",
                    "title": "Low-risk coverage",
                    "value": f"{adversarial_low_risk} → {baseline_low_risk} sources",
                    "detail": f"Coverage collapses from {baseline_low_risk} → {adversarial_low_risk} under adversarial stress",
                    "severity": "critical" if current_scenario["low_risk_sources"] < 4 else "medium"
                },
                {
                    "icon": "🔄",
                    "title": "Policy stability",
                    "value": f"{current_scenario['policy_changes']} changes",
                    "detail": "Moderate" if current_scenario['policy_changes'] < 10 else "High instability detected",
                    "severity": "high" if current_scenario['policy_changes'] > 10 else "low"
                },
                {
                    "icon": "⚖️",
                    "title": "Risk-coverage trade-off",
                    "value": f"{current_scenario['coverage']}% @ {current_emv:.2f}",
                    "detail": f"Tolerance level: {risk_tolerance:.1f} (0=Conservative, 1=Aggressive)",
                    "severity": "medium"
                }
            ]
            
            for finding in findings:
                severity_colors = {
                    "critical": "#ef4444",
                    "high": "#f97316",
                    "medium": "#f59e0b",
                    "low": "#10b981"
                }
                color = severity_colors.get(finding["severity"], "#6b7280")
                
                st.markdown(f"""
                <div style='background: white; padding: 0.9rem; border-radius: 8px; 
                            border-left: 4px solid {color}; margin-bottom: 0.6rem;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <div style='display: flex; align-items: start;'>
                        <span style='font-size: 18px; margin-right: 0.8rem;'>{finding['icon']}</span>
                        <div style='flex: 1;'>
                            <p style='margin: 0; font-size: 11px; font-weight: 700; color: {color}; text-transform: uppercase;'>
                                {finding['title']}
                            </p>
                            <p style='margin: 0.3rem 0; font-size: 15px; font-weight: 700; color: #1e3a8a;'>
                                {finding['value']}
                            </p>
                            <p style='margin: 0; font-size: 10px; color: #6b7280;'>
                                {finding['detail']}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # ========== DECISION-FOCUSED SYNTHESIS SUMMARY ==========
            st.markdown("**💡 Stress Test Synthesis & Actionable Next Steps**")
            
            # Calculate key synthesis metrics
            emv_deterioration_pct = ((current_emv - baseline_emv) / baseline_emv * 100) if baseline_emv > 0 else 0
            coverage_decline = 70 - current_scenario["coverage"]  # Baseline coverage assumed at 70%
            low_risk_collapse_pct = ((baseline_low_risk - current_scenario["low_risk_sources"]) / baseline_low_risk * 100) if baseline_low_risk > 0 else 0
            policy_instability = current_scenario["policy_changes"]
            is_near_breakeven = abs(reliability_shift - break_even_rel) < 0.08
            is_high_deception = deception_inflation > high_sensitivity_dec
            
            # Determine overall stress condition
            if emv_deterioration_pct > 40 or current_scenario["low_risk_sources"] < 4:
                stress_level = "Critical"
                stress_color = "#ef4444"
                stress_icon = "🚨"
            elif emv_deterioration_pct > 20 or policy_instability > 10:
                stress_level = "High"
                stress_color = "#f97316"
                stress_icon = "⚠️"
            elif emv_deterioration_pct > 10 or is_near_breakeven:
                stress_level = "Moderate"
                stress_color = "#f59e0b"
                stress_icon = "⚡"
            else:
                stress_level = "Acceptable"
                stress_color = "#10b981"
                stress_icon = "✅"
            
            # Build dynamic narrative based on stress conditions
            if stress_level == "Critical":
                narrative = f"""Current stress parameters have induced <strong>critical operational degradation</strong>. 
                EMV has deteriorated by <strong>{emv_deterioration_pct:.1f}%</strong> from baseline, and low-risk source availability 
                has collapsed to <strong>{current_scenario['low_risk_sources']} sources</strong> (down {low_risk_collapse_pct:.0f}%). 
                Task coverage has fallen to <strong>{current_scenario['coverage']}%</strong>."""
                
                action = f"""<strong>Immediate Actions Required:</strong> Do not proceed with current ML-TSSP policy under these conditions—risk exposure is unacceptable. Revert to conservative baseline policy or suspend high-stakes tasking until conditions stabilize. Activate enhanced source vetting protocols for all {len(ml_policy)} sources and escalate to command authority, as this stress scenario exceeds acceptable risk tolerance ({risk_tolerance:.1f})."""
                
            elif stress_level == "High":
                if is_high_deception:
                    narrative = f"""Current scenario indicates <strong>high adversarial pressure</strong> with deception risk 
                    at {deception_inflation:.0%}. Policy has undergone <strong>{policy_instability} assignment changes</strong>, 
                    signaling significant instability. EMV degradation: <strong>{emv_deterioration_pct:.1f}%</strong>. 
                    Coverage maintained at {current_scenario['coverage']}% but with {current_scenario['low_risk_sources']} low-risk sources 
                    (vs {baseline_low_risk} at baseline)."""
                    
                    action = f"""<strong>Priority Actions:</strong> Implement enhanced counterintelligence screening immediately—deception threshold ({high_sensitivity_dec:.0%}) has been exceeded. Focus tasking on the {current_scenario['low_risk_sources']} validated low-risk sources for critical operations. Deploy corroboration protocols for sources exhibiting EVPI greater than 0.30, and consider reliability uplift interventions or source rotation to restore operational margin."""
                else:
                    narrative = f"""Reliability degradation to <strong>{reliability_shift:+.2f}</strong> has triggered substantial policy instability 
                    ({policy_instability} changes). Low-risk source pool reduced to <strong>{current_scenario['low_risk_sources']}</strong> 
                    ({low_risk_collapse_pct:.0f}% decline). EMV increase: <strong>{emv_deterioration_pct:.1f}%</strong>. 
                    Coverage: {current_scenario['coverage']}%."""
                    
                    action = f"""<strong>Priority Actions:</strong> Re-assess source reliability calibration as you are approaching the break-even threshold ({break_even_rel:.2f}). Prioritize resource allocation to maintain the {current_scenario['low_risk_sources']} remaining low-risk sources. Increase monitoring frequency for sources near disengagement threshold and prepare contingency tasking with conservative assignments if conditions worsen."""
                    
            elif stress_level == "Moderate":
                if is_near_breakeven:
                    narrative = f"""Scenario operates <strong>near critical break-even point</strong> (reliability: {reliability_shift:+.2f} vs 
                    break-even: {break_even_rel:.2f}). EMV deterioration: <strong>{emv_deterioration_pct:.1f}%</strong>. 
                    {current_scenario['low_risk_sources']} low-risk sources available. Policy changes: {policy_instability}. 
                    Small additional degradation could trigger non-linear risk escalation."""
                    
                    action = f"""<strong>Monitoring & Preparation:</strong> Establish tripwire alerts to monitor for reliability drift below {break_even_rel:.2f}. Maintain current policy but prepare fallback assignments for approximately {int(len(ml_policy) * 0.3)} highest-risk sources. Increase source validation cadence to detect early warning indicators and document decision rationale for audit trail—you are operating in constrained margin."""
                else:
                    narrative = f"""ML-TSSP policy demonstrates <strong>moderate resilience</strong> under current stress. 
                    EMV increase limited to {emv_deterioration_pct:.1f}%. {current_scenario['low_risk_sources']} low-risk sources 
                    (vs {baseline_low_risk} baseline). Coverage: {current_scenario['coverage']}%. Policy adjustments: {policy_instability}."""
                    
                    action = f"""<strong>Operational Guidance:</strong> Proceed with ML-TSSP recommendations—risk remains within tolerance ({risk_tolerance:.1f}). Maintain standard monitoring protocols for the {current_scenario['low_risk_sources']} core low-risk sources and document scenario assumptions for operational record. Consider incremental collection investment if deception risk escalates beyond {high_sensitivity_dec:.0%}."""
            else:  # Acceptable
                narrative = f"""Policy demonstrates <strong>robust performance</strong> under stress scenario. 
                EMV increase minimal (<strong>{emv_deterioration_pct:.1f}%</strong>). Low-risk source availability: 
                <strong>{current_scenario['low_risk_sources']}/{baseline_low_risk}</strong>. Coverage sustained at {current_scenario['coverage']}%. 
                Policy changes limited to {policy_instability} adjustments—indicating stable optimization."""
                
                action = f"**Execute with Confidence:** ML-TSSP policy has been validated under stress—proceed with recommended tasking assignments. Maintain routine monitoring with no immediate intervention required. Stress margins are comfortable with a {abs(reliability_shift - break_even_rel):.2f} buffer from break-even threshold. Allocate resources to operational execution rather than additional vetting."
            
            # Render dynamic summary panel using containers
            summary_container = st.container()
            with summary_container:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%); 
                            padding: 1.3rem; border-radius: 10px; 
                            border: 2px solid {stress_color}; 
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            margin-bottom: 1rem;'>
                    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                        <span style='font-size: 28px; margin-right: 1rem;'>{stress_icon}</span>
                        <div style='flex: 1;'>
                            <h5 style='margin: 0; color: {stress_color}; font-size: 15px; font-weight: 700; text-transform: uppercase;'>
                                Stress Level: {stress_level}
                            </h5>
                            <p style='margin: 0.3rem 0 0 0; font-size: 11px; color: #6b7280;'>
                                Scenario: Reliability {reliability_shift:+.2f} | Deception {deception_inflation:+.0%} | Risk Tolerance {risk_tolerance:.1f}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Operational Assessment
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 6px; border-left: 3px solid {stress_color}; margin-bottom: 1rem;'>
                    <p style='margin: 0; font-size: 12px; color: #1f2937; line-height: 1.7;'>
                        <strong>Operational Assessment:</strong> {narrative}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommended Actions
                st.markdown("""
                <div style='background: #f0f9ff; padding: 1rem; border-radius: 6px; border-left: 3px solid #3b82f6; margin-bottom: 1rem;'>
                    <p style='margin: 0 0 0.6rem 0; font-size: 11px; font-weight: 700; color: #1e40af; text-transform: uppercase;'>
                        Recommended Actions
                    </p>
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='margin: 0; font-size: 11px; color: #1e3a8a; line-height: 1.8;'>{action}</p></div>", unsafe_allow_html=True)


def _render_audit_governance_section():
    """
    Comprehensive Audit and Governance section with:
    1. Decision Timeline (filterable log)
    2. Decision Record Viewer (detailed view on click)
    3. Governance Controls (approval, locking, export)
    4. Versioning & Drift Alerts
    """
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); 
                padding: 1.5rem; border-radius: 12px; border-left: 5px solid #6366f1; 
                margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>
        <h4 style='margin: 0; color: #1e293b; font-size: 18px; font-weight: 700;'>
            🧑‍⚖️ Audit & Governance Dashboard
        </h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 13px; color: #64748b; line-height: 1.6;'>
            Complete decision accountability: timeline tracking, model evidence, policy compliance, and governance controls.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== INITIALIZE AUDIT LOG ==========
    if "audit_log" not in st.session_state:
        _initialize_audit_log()
    
    # ========== DRIFT & RISK ALERTS BANNER ==========
    _render_alert_banner()
    
    st.divider()
    
    # ========== DECISION TIMELINE (PRIMARY LOG) ==========
    st.markdown("### 🧾 Decision Timeline")
    
    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        date_filter = st.date_input("Date Range", value=datetime.now().date(), key="audit_date_filter")
    
    with filter_col2:
        mode_filter = st.multiselect("Operational Mode", ["Conservative", "Balanced", "Aggressive", "Custom", "All"], default=["All"], key="audit_mode_filter")
    
    with filter_col3:
        risk_filter = st.multiselect("Risk Posture", ["Low", "Medium", "High", "Critical", "All"], default=["All"], key="audit_risk_filter")
    
    with filter_col4:
        reviewer_filter = st.multiselect("Reviewer", ["System", "Analyst A", "Analyst B", "Commander", "All"], default=["All"], key="audit_reviewer_filter")
    
    # Apply filters
    audit_log = st.session_state["audit_log"]
    filtered_log = _apply_filters(audit_log, mode_filter, risk_filter, reviewer_filter)
    
    # Display timeline table
    if len(filtered_log) > 0:
        st.markdown(f"*Showing {len(filtered_log)} of {len(audit_log)} decision events*")
        
        # Create interactive dataframe
        timeline_df = pd.DataFrame(filtered_log)
        
        # Display selection-enabled table
        event_selection = st.dataframe(
            timeline_df[["Time", "Operation", "Sources", "Mode", "Risk Posture", "Outcome", "Confidence", "Reviewer"]],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="audit_timeline_table"
        )
        
        # ========== DECISION RECORD VIEWER ==========
        if event_selection and len(event_selection.get("selection", {}).get("rows", [])) > 0:
            selected_idx = event_selection["selection"]["rows"][0]
            selected_record = filtered_log[selected_idx]
            
            st.divider()
            st.markdown("### 🔍 Decision Record Viewer")
            
            _render_decision_record(selected_record)
    else:
        st.info("No decision events match the selected filters.")
    
    st.divider()
    
    # ========== GOVERNANCE CONTROLS ==========
    st.markdown("### 🧑‍⚖️ Governance Controls")
    
    gov_col1, gov_col2, gov_col3, gov_col4 = st.columns(4)
    
    with gov_col1:
        require_approval = st.checkbox("Require Human Approval", value=False, key="require_approval", 
                                       help="All optimization decisions require analyst approval before execution")
    
    with gov_col2:
        lock_decision = st.button("🔒 Lock Current Decision", key="lock_decision",
                                   help="Prevent further modifications to the active policy")
    
    with gov_col3:
        if st.button("📄 Export Audit Report", key="export_audit"):
            _export_audit_report()
    
    with gov_col4:
        if st.button("🚩 Flag for Review", key="flag_review"):
            st.success("Decision flagged for commander review")
    
    # Display governance status
    if require_approval:
        st.info("🔐 Governance Mode: Human-in-the-Loop ACTIVE. All decisions require analyst approval.")
    
    if lock_decision:
        st.session_state["decision_locked"] = True
        st.success("🔒 Current decision locked. No further modifications permitted without unlock.")


def _initialize_audit_log():
    """Initialize synthetic audit log with realistic decision events."""
    from datetime import datetime, timedelta
    
    audit_entries = []
    
    # Generate last 10 decision events
    base_time = datetime.now()
    
    operations = [
        "Optimization Run", "Manual Override", "Threshold Adjustment", 
        "Source Reassignment", "Policy Comparison", "Stress Test", 
        "Drift Detection", "Risk Recalibration"
    ]
    
    modes = ["Conservative", "Balanced", "Aggressive", "Custom"]
    risk_postures = ["Low", "Medium", "High", "Critical"]
    reviewers = ["System", "Analyst A", "Analyst B", "Commander"]
    
    for i in range(10):
        time_offset = timedelta(minutes=i * 15 + np.random.randint(1, 10))
        timestamp = base_time - time_offset
        
        operation = np.random.choice(operations)
        mode = np.random.choice(modes)
        risk_posture = np.random.choice(risk_postures)
        reviewer = "System" if operation == "Optimization Run" else np.random.choice(reviewers)
        
        if operation == "Optimization Run":
            sources = np.random.randint(15, 30)
            outcome = f"{np.random.randint(6, 12)} Assigned"
            confidence = np.random.uniform(0.75, 0.95)
        elif operation == "Manual Override":
            sources = f"SRC_{np.random.randint(1, 80):03d}"
            outcome = "Reassigned"
            confidence = np.random.uniform(0.55, 0.75)
        else:
            sources = np.random.randint(10, 25)
            outcome = "Completed"
            confidence = np.random.uniform(0.65, 0.90)
        
        # Create detailed record
        record = {
            "Time": timestamp.strftime("%H:%M"),
            "Timestamp": timestamp,
            "Operation": operation,
            "Sources": sources,
            "Mode": mode,
            "Risk Posture": risk_posture,
            "Outcome": outcome,
            "Confidence": f"{confidence:.2f}",
            "Reviewer": reviewer,
            
            # Detailed fields for record viewer
            "operation_id": f"OP_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            "mission_objective": "Intelligence Collection" if i % 2 == 0 else "Force Protection",
            "model_version": "ML-TSSP v2.1",
            "data_version": "Dataset v1.4",
            "policy_version": "Policy v1.7",
            "threshold_violations": np.random.choice([True, False], p=[0.2, 0.8]),
            "risk_constraints_satisfied": np.random.choice([True, False], p=[0.85, 0.15]),
            "manual_overrides": np.random.choice([0, 1, 2], p=[0.7, 0.25, 0.05]),
            "sources_tasked": np.random.randint(5, 12),
            "sources_excluded": np.random.randint(3, 8),
            "expected_mission_success": np.random.uniform(0.70, 0.95),
            "risk_exposure_score": np.random.uniform(0.15, 0.45),
            "shap_drivers": _generate_shap_summary(),
            "reliability_forecast": np.random.uniform(0.65, 0.90),
            "deception_indicators": np.random.randint(0, 3)
        }
        
        audit_entries.append(record)
    
    # Sort by timestamp (most recent first)
    audit_entries.sort(key=lambda x: x["Timestamp"], reverse=True)
    
    st.session_state["audit_log"] = audit_entries


def _generate_shap_summary():
    """Generate abbreviated SHAP driver summary."""
    features = ["task_success_rate", "corroboration_score", "reliability_trend", "report_timeliness"]
    selected_features = np.random.choice(features, size=2, replace=False)
    
    summary = []
    for feat in selected_features:
        impact = np.random.choice(["High", "Medium", "Low"])
        direction = np.random.choice(["increases", "reduces"])
        summary.append(f"{feat.replace('_', ' ').title()} ({impact}, {direction})")
    
    return " | ".join(summary)


def _apply_filters(audit_log, mode_filter, risk_filter, reviewer_filter):
    """Apply selected filters to audit log."""
    filtered = audit_log.copy()
    
    if "All" not in mode_filter:
        filtered = [r for r in filtered if r["Mode"] in mode_filter]
    
    if "All" not in risk_filter:
        filtered = [r for r in filtered if r["Risk Posture"] in risk_filter]
    
    if "All" not in reviewer_filter:
        filtered = [r for r in filtered if r["Reviewer"] in reviewer_filter]
    
    return filtered


def _render_alert_banner():
    """Render drift and risk alert banner if conditions are met."""
    
    # Check for alerts
    alerts = []
    
    # Simulate alert conditions
    if np.random.random() > 0.7:
        alerts.append({
            "type": "drift",
            "severity": "warning",
            "message": "Behavior distribution shift detected: 3 sources transitioned from Cooperative → Uncertain in last 24h"
        })
    
    if np.random.random() > 0.8:
        alerts.append({
            "type": "forecast",
            "severity": "critical",
            "message": "Reliability forecast degradation: SRC_004, SRC_012, SRC_019 trending below threshold"
        })
    
    if np.random.random() > 0.85:
        alerts.append({
            "type": "deception",
            "severity": "critical",
            "message": "Deception risk spike: 2 sources flagged with elevated deception indicators (>0.75)"
        })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert["severity"] == "critical":
                icon = "🚨"
                bg_color = "#fef2f2"
                border_color = "#ef4444"
                text_color = "#991b1b"
            else:
                icon = "⚠️"
                bg_color = "#fffbeb"
                border_color = "#f59e0b"
                text_color = "#92400e"
            
            st.markdown(f"""
            <div style='background: {bg_color}; padding: 1rem; border-radius: 8px; 
                        border-left: 4px solid {border_color}; margin-bottom: 1rem;'>
                <p style='margin: 0; font-size: 13px; color: {text_color}; font-weight: 600;'>
                    {icon} <strong>{alert["type"].upper()} ALERT:</strong> {alert["message"]}
                </p>
            </div>
            """, unsafe_allow_html=True)


def _render_decision_record(record):
    """Render detailed decision record viewer."""
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🧠 Model Evidence", "✅ Compliance", "📊 Outcome"])
    
    with tab1:
        st.markdown("#### Decision Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown(f"""
            **Operation ID:** `{record['operation_id']}`  
            **Timestamp:** {record['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
            **Analyst/System:** {record['Reviewer']}
            """)
        
        with summary_col2:
            st.markdown(f"""
            **Operational Mode:** {record['Mode']}  
            **Risk Posture:** {record['Risk Posture']}  
            **Mission Objective:** {record['mission_objective']}
            """)
        
        with summary_col3:
            st.markdown(f"""
            **Model Version:** {record['model_version']}  
            **Data Version:** {record['data_version']}  
            **Policy Version:** {record['policy_version']}
            """)
    
    with tab2:
        st.markdown("#### Model Evidence Snapshot")
        
        st.markdown(f"""
        **Key SHAP Drivers:**  
        {record['shap_drivers']}
        
        **Reliability Forecast:** {record['reliability_forecast']:.2f}  
        **Deception Indicators:** {record['deception_indicators']} source(s) flagged
        
        **Optimization Objective:** Maximize Expected Mission Value (EMV)
        """)
    
    with tab3:
        st.markdown("#### Policy Compliance Check")
        
        compliance_col1, compliance_col2 = st.columns(2)
        
        with compliance_col1:
            threshold_status = "❌ Violated" if record['threshold_violations'] else "✅ Satisfied"
            risk_status = "✅ Satisfied" if record['risk_constraints_satisfied'] else "❌ Violated"
            
            st.markdown(f"""
            **Threshold Compliance:** {threshold_status}  
            **Risk Constraints:** {risk_status}
            """)
        
        with compliance_col2:
            override_status = "Yes" if record['manual_overrides'] > 0 else "No"
            st.markdown(f"""
            **Manual Overrides:** {override_status}  
            {f"*{record['manual_overrides']} override(s) applied*" if record['manual_overrides'] > 0 else ""}
            """)
        
        # Compliance summary
        if record['threshold_violations'] or not record['risk_constraints_satisfied']:
            st.warning("⚠️ This decision contains compliance violations. Review required.")
        else:
            st.success("✅ Full policy compliance achieved.")
    
    with tab4:
        st.markdown("#### Final Outcome")
        
        outcome_col1, outcome_col2 = st.columns(2)
        
        with outcome_col1:
            st.metric("Sources Tasked", record['sources_tasked'])
            st.metric("Expected Mission Success", f"{record['expected_mission_success']:.1%}")
        
        with outcome_col2:
            st.metric("Sources Excluded", record['sources_excluded'])
            st.metric("Risk Exposure Score", f"{record['risk_exposure_score']:.2f}")
        
        # Visual risk gauge
        st.markdown("**Risk Exposure Level:**")
        risk_val = record['risk_exposure_score']
        if risk_val < 0.25:
            risk_label = "Low"
            risk_color = "#10b981"
        elif risk_val < 0.40:
            risk_label = "Medium"
            risk_color = "#f59e0b"
        else:
            risk_label = "High"
            risk_color = "#ef4444"
        
        st.progress(risk_val, text=f"{risk_label} Risk ({risk_val:.2%})")


def _export_audit_report():
    """Export audit log to CSV."""
    audit_log = st.session_state.get("audit_log", [])
    
    if not audit_log:
        st.error("No audit data to export.")
        return
    
    # Convert to DataFrame for export
    export_df = pd.DataFrame(audit_log)
    export_df = export_df[[
        "Timestamp", "Operation", "Sources", "Mode", "Risk Posture", 
        "Outcome", "Confidence", "Reviewer", "operation_id", 
        "threshold_violations", "risk_constraints_satisfied", "manual_overrides"
    ]]
    
    # Generate CSV
    csv_data = export_df.to_csv(index=False)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_report_{timestamp}.csv"
    
    st.download_button(
        label="⬇️ Download Audit Report (CSV)",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key="download_audit_csv"
    )
    
    st.success(f"✅ Audit report prepared: {filename}")


def _render_drift_section():
    """
    Comprehensive drift monitoring with reliability/deception trajectories,
    risk state transitions, task assignment changes, and model integrity indicators.
    
    This timeline tracks evolving source behavior, model confidence, and resulting 
    tasking adjustments to ensure early risk detection and adaptive intelligence management.
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #f1fdf8 100%); 
                padding: 1.2rem; border-radius: 10px; border-left: 4px solid #3b82f6; 
                margin-bottom: 1.5rem;'>
        <h4 style='margin: 0; color: #1e3a8a; font-size: 16px; font-weight: 700;'>
            📡 Behavioral Drift & Risk Escalation Monitor
        </h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 12px; color: #6b7280; line-height: 1.6;'>
            This timeline tracks evolving source behavior, model confidence, and resulting tasking 
            adjustments to ensure early risk detection and adaptive intelligence management.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SOURCE SELECTOR ==========
    if st.session_state.get("results"):
        ml_policy = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
        source_ids = [p["source_id"] for p in ml_policy if "source_id" in p]
        if not source_ids:
            st.warning("No sources available. Please run optimization first.")
            return
    else:
        st.warning("Please run optimization to generate drift data.")
        return
    
    selected_source = st.selectbox(
        "Select Source for Drift Analysis",
        source_ids,
        key="drift_source_selector",
        help="Choose a source to view behavioral drift timeline"
    )
    
    # Get source policy assignment
    source_policy = next((p for p in ml_policy if p["source_id"] == selected_source), None)
    if not source_policy:
        st.error(f"No policy data found for {selected_source}")
        return
    
    st.divider()
    
    # ========== METRICS CARDS ABOVE CHART ==========
    st.markdown("**📊 Current Risk State & Trends**")
    
    card_col1, card_col2, card_col3, card_col4, card_col5 = st.columns(5)
    
    # Generate synthetic drift data (in production, this would come from actual ML model)
    np.random.seed(hash(selected_source) % 2**32)
    periods = 30  # 30 time periods
    
    # Base reliability and deception from source features
    base_reliability = source_policy.get("reliability_forecast", 0.65)
    base_deception = 1.0 - source_policy.get("corroboration_score", 0.7)
    
    # Generate trajectories with drift
    reliability_drift = -0.008 if base_reliability > 0.6 else -0.004  # Degradation
    deception_drift = 0.006 if base_deception > 0.4 else 0.003  # Escalation
    
    reliability_trajectory = []
    deception_trajectory = []
    risk_states = []
    
    for t in range(periods):
        # Reliability with noise
        rel = np.clip(base_reliability + reliability_drift * t + np.random.normal(0, 0.03), 0.2, 0.95)
        reliability_trajectory.append(rel)
        
        # Deception with noise
        dec = np.clip(base_deception + deception_drift * t + np.random.normal(0, 0.02), 0.05, 0.85)
        deception_trajectory.append(dec)
        
        # Risk state calculation
        risk_score = (1 - rel) * 0.6 + dec * 0.4
        if risk_score > 0.7:
            risk_states.append("Critical")
        elif risk_score > 0.5:
            risk_states.append("High")
        elif risk_score > 0.3:
            risk_states.append("Medium")
        else:
            risk_states.append("Low")
    
    # Calculate metrics
    current_reliability = reliability_trajectory[-1]
    current_deception = deception_trajectory[-1]
    current_risk_state = risk_states[-1]
    
    # Trend calculation (last 7 periods)
    rel_trend_val = reliability_trajectory[-1] - reliability_trajectory[-7]
    dec_trend_val = deception_trajectory[-1] - deception_trajectory[-7]
    
    rel_trend = "↑" if rel_trend_val > 0.02 else "↓" if rel_trend_val < -0.02 else "→"
    dec_trend = "↑" if dec_trend_val > 0.02 else "↓" if dec_trend_val < -0.02 else "→"
    
    # Task change simulation (changes when risk state changes significantly)
    task_changes = []
    last_state = risk_states[0]
    for t, state in enumerate(risk_states):
        if state != last_state:
            task_changes.append(t)
            last_state = state
    
    days_since_change = periods - task_changes[-1] if task_changes else periods
    
    # Monitoring level (based on current risk)
    risk_score_current = (1 - current_reliability) * 0.6 + current_deception * 0.4
    if risk_score_current > 0.6:
        monitoring_level = "Elevated"
    elif risk_score_current > 0.4:
        monitoring_level = "Standard"
    else:
        monitoring_level = "Routine"
    
    # Display cards
    with card_col1:
        state_color = {
            "Low": "#10b981",
            "Medium": "#f59e0b",
            "High": "#f97316",
            "Critical": "#ef4444"
        }[current_risk_state]
        st.markdown(f"""
        <div style='background: white; padding: 0.8rem; border-radius: 8px; 
                    border-left: 4px solid {state_color}; text-align: center;'>
            <p style='margin: 0; font-size: 10px; color: #6b7280; font-weight: 600;'>CURRENT RISK STATE</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 18px; font-weight: 700; color: {state_color};'>{current_risk_state}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with card_col2:
        rel_color = "#10b981" if current_reliability > 0.6 else "#f59e0b" if current_reliability > 0.4 else "#ef4444"
        st.markdown(f"""
        <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center;'>
            <p style='margin: 0; font-size: 10px; color: #6b7280; font-weight: 600;'>RELIABILITY TREND</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 18px; font-weight: 700; color: {rel_color};'>{rel_trend} {current_reliability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with card_col3:
        dec_color = "#ef4444" if current_deception > 0.6 else "#f59e0b" if current_deception > 0.3 else "#10b981"
        st.markdown(f"""
        <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center;'>
            <p style='margin: 0; font-size: 10px; color: #6b7280; font-weight: 600;'>DECEPTION TREND</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 18px; font-weight: 700; color: {dec_color};'>{dec_trend} {current_deception:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with card_col4:
        st.markdown(f"""
        <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center;'>
            <p style='margin: 0; font-size: 10px; color: #6b7280; font-weight: 600;'>DAYS SINCE TASK CHANGE</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 18px; font-weight: 700; color: #3b82f6;'>{days_since_change}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with card_col5:
        mon_color = "#ef4444" if monitoring_level == "Elevated" else "#3b82f6"
        st.markdown(f"""
        <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center;'>
            <p style='margin: 0; font-size: 10px; color: #6b7280; font-weight: 600;'>MONITORING LEVEL</p>
            <p style='margin: 0.3rem 0 0 0; font-size: 16px; font-weight: 700; color: {mon_color};'>{monitoring_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== MAIN DRIFT TIMELINE WITH RISK BANDS ==========
    st.markdown("**📈 Reliability & Deception Risk Timeline**")
    
    # Calculate smoothed trajectories
    window = 5
    reliability_smooth = pd.Series(reliability_trajectory).rolling(window, min_periods=1).mean().tolist()
    deception_smooth = pd.Series(deception_trajectory).rolling(window, min_periods=1).mean().tolist()
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add risk state background bands
    state_colors = {
        "Low": "rgba(16, 185, 129, 0.1)",
        "Medium": "rgba(245, 158, 11, 0.1)",
        "High": "rgba(249, 115, 22, 0.15)",
        "Critical": "rgba(239, 68, 68, 0.2)"
    }
    
    current_state = risk_states[0]
    start_idx = 0
    for i in range(1, len(risk_states) + 1):
        if i == len(risk_states) or risk_states[i] != current_state:
            fig.add_shape(
                type="rect",
                x0=start_idx, x1=i-1,
                y0=0, y1=1,
                fillcolor=state_colors[current_state],
                line=dict(width=0),
                layer="below",
                secondary_y=False
            )
            if i < len(risk_states):
                current_state = risk_states[i]
                start_idx = i
    
    # Add reliability trajectory (raw)
    fig.add_trace(
        go.Scatter(
            x=list(range(periods)),
            y=reliability_trajectory,
            mode='lines+markers',
            name='Reliability (Raw)',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=5, opacity=0.6),
            hovertemplate='<b>Period %{x}</b><br>Reliability: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add reliability smoothed
    fig.add_trace(
        go.Scatter(
            x=list(range(periods)),
            y=reliability_smooth,
            mode='lines',
            name='Reliability (Smoothed)',
            line=dict(color='#1e40af', width=3, dash='dash'),
            hovertemplate='<b>Period %{x}</b><br>Smoothed: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add deception trajectory (raw)
    fig.add_trace(
        go.Scatter(
            x=list(range(periods)),
            y=deception_trajectory,
            mode='lines+markers',
            name='Deception Risk (Raw)',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=5, opacity=0.6),
            hovertemplate='<b>Period %{x}</b><br>Deception: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add deception smoothed
    fig.add_trace(
        go.Scatter(
            x=list(range(periods)),
            y=deception_smooth,
            mode='lines',
            name='Deception (Smoothed)',
            line=dict(color='#991b1b', width=3, dash='dash'),
            hovertemplate='<b>Period %{x}</b><br>Smoothed: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add threshold lines
    fig.add_hline(y=0.5, line_dash='dot', line_color='#6b7280', opacity=0.6,
                  annotation_text="Reliability Threshold", annotation_position="left",
                  secondary_y=False)
    fig.add_hline(y=0.6, line_dash='dot', line_color='#dc2626', opacity=0.6,
                  annotation_text="Deception Alert", annotation_position="right",
                  secondary_y=True)
    
    # Add task change annotations
    for change_idx in task_changes:
        fig.add_vline(
            x=change_idx,
            line_dash='dash',
            line_color='#7c3aed',
            opacity=0.7,
            annotation_text=f"Task Change",
            annotation_position="top"
        )
    
    # Update layout
    fig.update_xaxes(title_text="Time Period", gridcolor='#f3f4f6')
    fig.update_yaxes(title_text="<b>Reliability Score</b>", secondary_y=False,
                     range=[0, 1], gridcolor='#f3f4f6')
    fig.update_yaxes(title_text="<b>Deception Risk</b>", secondary_y=True,
                     range=[0, 1])
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='white',
        plot_bgcolor='#fafafa'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"drift_timeline_{selected_source}")
    
    st.divider()
    
    # ========== RECOURSE ACTIVATION INTENSITY ==========
    st.markdown("**💰 Recourse Activation Intensity**")
    st.caption("Aggregate monitoring and intervention costs over time")
    
    # Calculate recourse intensity based on risk
    recourse_intensity = []
    for t in range(periods):
        risk_val = (1 - reliability_trajectory[t]) * 0.6 + deception_trajectory[t] * 0.4
        # Higher risk = more recourse needed
        intensity = np.clip(risk_val * 1.5, 0.1, 1.0)
        recourse_intensity.append(intensity)
    
    fig_recourse = go.Figure()
    
    # Area plot for recourse intensity
    fig_recourse.add_trace(go.Scatter(
        x=list(range(periods)),
        y=recourse_intensity,
        mode='lines',
        name='Recourse Cost',
        fill='tozeroy',
        fillcolor='rgba(124, 58, 237, 0.3)',
        line=dict(color='#7c3aed', width=2),
        hovertemplate='<b>Period %{x}</b><br>Intensity: %{y:.3f}<extra></extra>'
    ))
    
    fig_recourse.update_layout(
        height=200,
        xaxis_title="Time Period",
        yaxis_title="Recourse Intensity",
        hovermode='x',
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='white',
        plot_bgcolor='#fafafa',
        yaxis=dict(range=[0, 1.2], gridcolor='#f3f4f6')
    )
    
    st.plotly_chart(fig_recourse, use_container_width=True, key=f"recourse_{selected_source}")
    
    st.divider()
    
    # ========== DRIFT ALERTS & FLAGS ==========
    st.markdown("**🚨 Behavioral Drift Alerts**")
    
    # Generate alerts based on thresholds
    alerts = []
    
    # Check reliability drop
    for t in range(7, periods):
        rel_change = reliability_trajectory[t] - reliability_trajectory[t-7]
        if rel_change < -0.15:
            alerts.append({
                "period": t,
                "type": "Reliability Degradation",
                "severity": "High",
                "message": f"Reliability dropped {abs(rel_change):.2%} over 7 periods"
            })
    
    # Check deception increase
    for t in range(7, periods):
        dec_change = deception_trajectory[t] - deception_trajectory[t-7]
        if dec_change > 0.12:
            alerts.append({
                "period": t,
                "type": "Deception Escalation",
                "severity": "Critical",
                "message": f"Deception risk increased {dec_change:.2%} over 7 periods"
            })
    
    # Check state transitions to high/critical
    for t in range(1, periods):
        if risk_states[t] in ["High", "Critical"] and risk_states[t-1] not in ["High", "Critical"]:
            alerts.append({
                "period": t,
                "type": "Risk Escalation",
                "severity": "High",
                "message": f"Risk state elevated to {risk_states[t]}"
            })
    
    if alerts:
        for alert in alerts[-5:]:  # Show last 5 alerts
            severity_colors = {
                "Critical": "#ef4444",
                "High": "#f97316",
                "Medium": "#f59e0b"
            }
            color = severity_colors.get(alert["severity"], "#6b7280")
            icon = "🔴" if alert["severity"] == "Critical" else "🟠" if alert["severity"] == "High" else "🟡"
            
            st.markdown(f"""
            <div style='background: white; padding: 0.8rem; border-radius: 8px; 
                        border-left: 4px solid {color}; margin-bottom: 0.5rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <p style='margin: 0; font-size: 11px; font-weight: 700; color: {color};'>
                            {icon} {alert['type']} — Period {alert['period']}
                        </p>
                        <p style='margin: 0.3rem 0 0 0; font-size: 10px; color: #6b7280;'>
                            {alert['message']}
                        </p>
                    </div>
                    <span style='padding: 0.2rem 0.6rem; background: {color}; color: white; 
                                 border-radius: 12px; font-size: 9px; font-weight: 700;'>
                        {alert['severity']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical drift alerts detected")
    
    st.divider()
    
    # ========== DYNAMIC DRIFT-BASED RECOMMENDATIONS ==========
    st.markdown("**💡 Operational Recommendations**")
    
    # Generate recommendations based on drift patterns
    if current_risk_state == "Critical":
        drift_rec = f"⚠️ **Critical Action Required**: {selected_source} shows critical risk state with reliability at {current_reliability:.2f} and deception at {current_deception:.2f}. Recommend immediate task reassignment and enhanced verification protocols. Consider suspending source pending investigation."
        drift_box = "error-box"
    elif current_risk_state == "High":
        drift_rec = f"⚠️ **Elevated Monitoring**: {selected_source} in high-risk state (reliability: {current_reliability:.2f}, deception: {current_deception:.2f}). Increase corroboration requirements, reduce task complexity, and activate enhanced oversight. Task change occurred {days_since_change} periods ago."
        drift_box = "warning-box"
    elif rel_trend == "↓" and current_reliability < 0.5:
        drift_rec = f"⚡ **Reliability Decline Detected**: {selected_source} shows downward reliability trend reaching {current_reliability:.2f}. Implement enhanced quality controls and consider rotating to lower-criticality tasks until trend stabilizes."
        drift_box = "warning-box"
    elif dec_trend == "↑" and current_deception > 0.4:
        drift_rec = f"⚡ **Deception Risk Rising**: {selected_source} exhibits increasing deception indicators ({current_deception:.2f}). Activate cross-validation protocols and increase spot-check frequency. Review recent reporting for anomalies."
        drift_box = "warning-box"
    elif monitoring_level == "Elevated":
        drift_rec = f"📊 **Maintain Elevated Monitoring**: {selected_source} requires continued oversight (reliability: {current_reliability:.2f}, deception: {current_deception:.2f}). Current risk state is {current_risk_state}. Sustain standard verification procedures."
        drift_box = "insight-box"
    elif days_since_change > 20:
        drift_rec = f"✅ **Stable Performance**: {selected_source} shows {days_since_change} periods without task change, indicating consistent behavior (reliability: {current_reliability:.2f}, deception: {current_deception:.2f}). Maintain routine monitoring, consider for higher-value assignments if sustained."
        drift_box = "success-box"
    else:
        drift_rec = f"✅ **Normal Operations**: {selected_source} operating within acceptable parameters (reliability: {current_reliability:.2f}, deception: {current_deception:.2f}, {current_risk_state.lower()} risk). Continue standard monitoring protocols."
        drift_box = "success-box"
    
    st.markdown(f"""
    <div class="{drift_box}" style="margin-top: 1rem;">
        <p style="margin:0; line-height: 1.6;">{drift_rec}</p>
    </div>
    """, unsafe_allow_html=True)

HEADER_IMAGE_PATH = Path(r"D:\FINAL HUMINT DASH\background-logo.png")

def _load_header_background() -> str:
    try:
        with HEADER_IMAGE_PATH.open("rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
            return (
                "linear-gradient(120deg, rgba(15,23,42,0.88), rgba(30,64,175,0.75)), "
                f"url('data:image/png;base64,{encoded}')"
            )
    except FileNotFoundError:
        return "linear-gradient(120deg, rgba(15,23,42,0.88), rgba(30,64,175,0.75))"


def __get_base64_image(image_path):
    """Convert image to base64 for embedding."""
    import base64
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""


def _render_login_page():
    """Render authentication login page with professional design."""
    
    # Initialize session state for login form visibility
    if "show_login_form" not in st.session_state:
        st.session_state.show_login_form = False
    
    # Custom CSS for login page with gradient background
    st.markdown("""
    <style>
    /* Full screen gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Overlay pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .login-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 1rem 1.5rem 1.5rem 1.5rem;
        position: relative;
        z-index: 1;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding-top: 2rem;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }
    
    .logo-container {
        margin-bottom: 0.5rem;
        margin-top: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    
    .logo-container:hover {
        transform: scale(1.05);
    }
    
    .logo-container img {
        max-width: 100px;
        max-height: 80px;
        height: auto;
        opacity: 0.9;
        transition: opacity 0.3s ease;
        object-fit: contain;
    }
    
    .logo-container img:hover {
        opacity: 1;
    }
    
    .login-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .login-subtitle {
        font-size: 14px;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.4;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .login-box {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.5) inset;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .login-form-header {
        text-align: center;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .login-form-header h3 {
        color: #1e40af;
        font-size: 16px;
        font-weight: 600;
        margin: 0;
    }
    
    .login-footer {
        text-align: center;
        margin-top: 0.6rem;
        font-size: 10px;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .demo-credentials-box {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Tab positioning - hidden */
    .stTabs {
        background: transparent;
        margin-top: -1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        display: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        display: none;
    }
    
    .stTabs [aria-selected="true"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Professional Logo and Header section with image
    # Display the OptiSource logo - clickable to show/hide login form
    try:
        # Get the directory where dashboard.py is located
        script_dir = Path(__file__).parent
        logo_path = script_dir / "OptiSource.jpeg"
        
        # Check if logo exists, otherwise try alternative names
        if not logo_path.exists():
            # Try other possible names
            for name in ["OptiSource.jpg", "optisource.jpeg", "logo.png", "OptiSource.png"]:
                alt_path = script_dir / name
                if alt_path.exists():
                    logo_path = alt_path
                    break
        
        if logo_path.exists():
            st.markdown('<div class="login-header">', unsafe_allow_html=True)
            
            # Center the logo - clickable to toggle login form
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Display logo with custom styling
                st.markdown('<div class="logo-container">', unsafe_allow_html=True)
                st.image(str(logo_path), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clickable area below logo
                if st.button("🔐 Click to Access", key="show_login", use_container_width=True, type="primary"):
                    st.session_state.show_login_form = not st.session_state.show_login_form
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Fallback to emoji logo if image not found
            st.markdown("""
            <div class="login-header">
                <div class="logo-container">
                    <div style="
                        width: 25px;
                        height: 25px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                        opacity: 0.85;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto 0.8rem auto;
                        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4), 0 0 0 8px rgba(255, 255, 255, 0.3);
                        position: relative;
                        animation: pulse 3s ease-in-out infinite;
                    ">
                        <div style="
                            width: 85px;
                            height: 85px;
                            background: white;
                            border-radius: 50%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 40px;
                        ">
                            🛡️
                        </div>
                        <div style="
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            font-size: 30px;
                            z-index: 2;
                        ">
                            👁️
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        # Fallback to emoji logo on any error
        st.markdown("""
        <div class="login-header">
            <div class="logo-container">
                <div style="
                    width: 80px;
                    height: 80px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                    opacity: 0.85;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 0.3rem auto;
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4), 0 0 0 6px rgba(255, 255, 255, 0.3);
                    position: relative;
                    animation: pulse 3s ease-in-out infinite;
                ">
                    <div style="
                        width: 20px;
                        height: 20px;
                        background: white;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 32px;
                    ">
                        🛡️
                    </div>
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        font-size: 24px;
                        z-index: 2;
                    ">
                        👁️
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Tagline
    st.markdown("""<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<div style="font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 600; margin-top: 0.5rem; margin-bottom: 1.2rem; line-height: 1.4; text-align: center; letter-spacing: 0.3px; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); color: #ffffff;">
Risk-Aware Intelligence Source Optimization for Strategic Decision Superiority
</div>
</div>
<style>
@keyframes pulse {
0%, 100% { box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4), 0 0 0 6px rgba(255, 255, 255, 0.3); }
50% { box-shadow: 0 12px 40px rgba(102, 126, 234, 0.6), 0 0 0 9px rgba(255, 255, 255, 0.4); }
}
</style>""", unsafe_allow_html=True)
    
    # Show login form only if button was clicked
    if st.session_state.show_login_form:
        # Login box with tab positioning
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # Predefined credentials (in production, use proper authentication)
        CREDENTIALS = {
            "admin": "admin123",
            "analyst": "analyst123",
            "commander": "command123",
            "operator": "ops123"
        }
        
        # Login form with header
        with st.form("login_form", clear_on_submit=True):
            st.markdown("""
            <div class="login-form-header">
                <h3>🔐 Secure Authentication</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([0.5, 3, 0.5])
            with col2:
                username = st.text_input(
                    "👤 Username",
                    placeholder="Enter your username",
                    key="login_username"
                )
                
                password = st.text_input(
                    "🔑 Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password"
                )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button("🚀 Sign In", use_container_width=True, type="primary")
            
            if submit:
                if username in CREDENTIALS and password == CREDENTIALS[username]:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    st.success(f"✅ Welcome back, {username.title()}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check your username and password.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login-box
        
        # Demo credentials info - more compact
        st.markdown("""
        <div class="login-footer">
            <p style="margin: 0 0 0.3rem 0; font-weight: 600; font-size: 10px;">📋 Demo Credentials:</p>
            <div class="demo-credentials-box" style="padding: 0.4rem;">
                <p style="margin: 0.15rem 0; font-family: 'Roboto Mono', monospace; font-size: 10px; color: #1e40af; line-height: 1.5;">
                    <strong>Admin:</strong> admin / admin123<br>
                    <strong>Analyst:</strong> analyst / analyst123<br>
                    <strong>Commander:</strong> commander / command123<br>
                    <strong>Operator:</strong> operator / ops123
                </p>
            </div>
            <p style="margin-top: 0.4rem; font-size: 9px; opacity: 0.8;">
                © 2026 ML-TSSP HUMINT Dashboard | 🛡️ Classified Intelligence System
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close login-container


# ======================================================
# SOURCE DATA INPUT SYSTEM
# ======================================================

def _validate_source_schema(data_dict):
    """Validate source data schema and return clean dict or error."""
    required_fields = ["source_id", "task_success_rate", "corroboration_score", "report_timeliness",
                      "handler_confidence", "deception_score", "ci_flag"]
    optional_fields = ["behavior_category", "access_level", "handler_notes"]
    
    errors = []
    clean_data = {}
    
    # Check required fields
    for field in required_fields:
        if field not in data_dict:
            errors.append(f"Missing required field: {field}")
        else:
            clean_data[field] = data_dict[field]
    
    # Validate numeric ranges for continuous features (0.0 to 1.0)
    numeric_fields = ["task_success_rate", "corroboration_score", "report_timeliness",
                     "handler_confidence", "deception_score"]
    
    for field in numeric_fields:
        if field in clean_data:
            try:
                val = float(clean_data[field])
                if not (0.0 <= val <= 1.0):
                    errors.append(f"{field} must be between 0.0 and 1.0")
                clean_data[field] = val
            except (ValueError, TypeError):
                errors.append(f"{field} must be a numeric value")
    
    # Validate ci_flag (binary: 0 or 1)
    if "ci_flag" in clean_data:
        try:
            val = int(clean_data["ci_flag"])
            if val not in [0, 1]:
                errors.append("ci_flag must be 0 or 1")
            clean_data["ci_flag"] = val
        except (ValueError, TypeError):
            errors.append("ci_flag must be 0 or 1")
    
    # Add optional fields if present
    for field in optional_fields:
        if field in data_dict:
            clean_data[field] = data_dict[field]
    
    if errors:
        return None, errors
    return clean_data, None


def _process_single_source(source_data, recourse_rules):
    """
    Process a single source through ML-TSSP pipeline.
    Pipeline: Input → Preprocessing → ML Score → TSSP Decision → Explanation
    """
    try:
        # ===== STAGE 1: INPUT & PREPROCESSING =====
        features = {
            "task_success_rate": source_data["task_success_rate"],
            "corroboration_score": source_data["corroboration_score"],
            "report_timeliness": source_data["report_timeliness"],
            "handler_confidence": source_data["handler_confidence"],
            "deception_score": source_data["deception_score"],
            "ci_flag": source_data["ci_flag"]
        }
        
        source_id = source_data["source_id"]
        
        # Initialize result structure
        source_result = {
            "source_id": source_id,
            "features": features,
            "recourse_rules": recourse_rules,
            "pipeline_stages": {
                "1_preprocessing": "complete",
                "2_ml_scoring": "pending",
                "3_tssp_decision": "pending",
                "4_explanation": "pending"
            }
        }
        
        # ===== STAGE 2: ML SCORING =====
        # Calculate ML reliability and deception scores using training formula
        tsr = features["task_success_rate"]
        cor = features["corroboration_score"]
        time = features["report_timeliness"]
        handler = features["handler_confidence"]
        dec_score = features["deception_score"]
        ci = features["ci_flag"]
        
        # Check if all inputs are zero - if so, return zero metrics
        if tsr == 0.0 and cor == 0.0 and time == 0.0 and handler == 0.0 and dec_score == 0.0 and ci == 0:
            reliability = 0.0
            deception = 0.0
        else:
            # Use deterministic seed for reproducibility
            rng = np.random.default_rng(hash(source_id) % (2**32))
            
            # ML Reliability Score (exact training formula)
            # reliability_score = 0.30 * TSR + 0.25 * COR + 0.20 * TIME + 0.15 * HANDLER - 0.15 * DEC - 0.10 * CI
            reliability = np.clip(
                0.30 * tsr + 0.25 * cor + 0.20 * time + 0.15 * handler 
                - 0.15 * dec_score - 0.10 * ci + 0.05 * rng.normal(0, 0.03),
                0.0, 1.0
            )
            
            # Deception Confidence (inverse of reliability indicators + pattern noise)
            deception = np.clip(
                0.30 * dec_score + 0.25 * ci + 0.20 * (1 - cor) + 0.15 * (1 - handler) 
                + 0.10 * rng.beta(2, 5),
                0.0, 1.0
            )
        
        source_result["ml_reliability"] = float(reliability)
        source_result["deception_confidence"] = float(deception)
        source_result["pipeline_stages"]["2_ml_scoring"] = "complete"
        
        # ===== STAGE 3: TSSP DECISION =====
        # Apply decision thresholds
        rel_disengage = recourse_rules.get("rel_disengage", 0.35)
        rel_flag = recourse_rules.get("rel_ci_flag", 0.50)
        dec_disengage = recourse_rules.get("dec_disengage", 0.75)
        dec_flag = recourse_rules.get("dec_ci_flag", 0.60)
        
        # Special case: all inputs zero means source has no data
        if tsr == 0.0 and cor == 0.0 and time == 0.0:
            decision = "disengage"
            action_reason = "No source data available (all features are zero)"
            task_assigned = None
            is_taskable = False
            expected_risk = 0.0
            optimization_score = 0.0
        else:
            # Use rng for risk calculations
            rng = np.random.default_rng(hash(source_id) % (2**32))
            
            # Decision logic with priority order
            if deception >= dec_disengage:
                decision = "disengage"
                action_reason = f"High deception risk ({deception:.3f} ≥ {dec_disengage})"
                task_assigned = None
                is_taskable = False
            elif reliability < rel_disengage:
                decision = "disengage"
                action_reason = f"Low reliability ({reliability:.3f} < {rel_disengage})"
                task_assigned = None
                is_taskable = False
            elif deception >= dec_flag:
                decision = "flag_for_ci"
                action_reason = f"Elevated deception risk ({deception:.3f} ≥ {dec_flag})"
                task_assigned = rng.choice(TASK_ROSTER)
                is_taskable = True
            elif reliability < rel_flag:
                decision = "flag_and_task"
                action_reason = f"Below optimal reliability ({reliability:.3f} < {rel_flag})"
                task_assigned = rng.choice(TASK_ROSTER)
                is_taskable = True
            else:
                decision = "task"
                action_reason = f"Meets operational standards (rel: {reliability:.3f}, dec: {deception:.3f})"
                task_assigned = rng.choice(TASK_ROSTER)
                is_taskable = True
            
            # Calculate risk metrics
            expected_risk = np.clip(
                0.30 * (1 - reliability) + 0.40 * deception + 0.30 * rng.beta(2, 3),
                0.0, 1.0
            )
            
            # Calculate optimization score
            optimization_score = reliability * (1 - deception) * (1 - expected_risk)
        
        source_result["decision"] = decision
        source_result["action_reason"] = action_reason
        source_result["is_taskable"] = is_taskable
        source_result["tssp_allocation"] = [task_assigned] if task_assigned else []
        source_result["expected_risk"] = float(expected_risk)
        source_result["optimization_score"] = float(optimization_score)
        source_result["pipeline_stages"]["3_tssp_decision"] = "complete"
        
        # ===== STAGE 4: GENERATE EXPLANATION =====
        # Handle zero-input case for all metrics
        if tsr == 0.0 and cor == 0.0 and time == 0.0:
            emv_ml = 0.0
            emv_deterministic = 0.0
            emv_uniform = 0.0
            optimization_score = 0.0
            expected_risk = 0.0
            source_result["optimization_score"] = 0.0
            source_result["expected_risk"] = 0.0
        else:
            # Calculate EMV (Expected Monetary Value)
            emv_ml = optimization_score
            emv_deterministic = emv_ml * 0.85
            emv_uniform = emv_ml * 0.70
        
        source_result["emv"] = {
            "ml_tssp": float(emv_ml),
            "deterministic": float(emv_deterministic),
            "uniform": float(emv_uniform)
        }
        
        # Add confidence metrics
        source_result["confidence_metrics"] = {
            "reliability_confidence": "high" if reliability >= 0.7 else "medium" if reliability >= 0.5 else "low",
            "deception_confidence": "low_risk" if deception < 0.3 else "moderate_risk" if deception < 0.6 else "high_risk",
            "overall_confidence": "operational" if is_taskable else "non_operational"
        }
        
        source_result["pipeline_stages"]["4_explanation"] = "complete"
        source_result["api_method"] = "local_ml_pipeline"
        
        return source_result, None
        
    except Exception as e:
        import traceback
        return None, f"Pipeline error: {str(e)}\n{traceback.format_exc()}"


def _process_batch_sources(df, recourse_rules):
    """Process multiple sources from DataFrame."""
    results = []
    errors = []
    sources = []
    
    # First pass: validate and build sources list
    for idx, row in df.iterrows():
        # Convert row to dict with all 6 input features
        source_data = {
            "source_id": row.get("source_id", f"SRC_{idx+1:03d}"),
            "task_success_rate": row.get("task_success_rate", 0.0),
            "corroboration_score": row.get("corroboration_score", 0.0),
            "report_timeliness": row.get("report_timeliness", 0.0),
            "handler_confidence": row.get("handler_confidence", 0.0),
            "deception_score": row.get("deception_score", 0.0),
            "ci_flag": row.get("ci_flag", 0)
        }
        
        # Validate
        clean_data, validation_errors = _validate_source_schema(source_data)
        if validation_errors:
            errors.append({"row": idx + 1, "source_id": source_data["source_id"], "errors": validation_errors})
            continue
        
        # Build source structure matching model training format with all 6 features
        features = {
            "task_success_rate": clean_data["task_success_rate"],
            "corroboration_score": clean_data["corroboration_score"],
            "report_timeliness": clean_data["report_timeliness"],
            "handler_confidence": clean_data["handler_confidence"],
            "deception_score": clean_data["deception_score"],
            "ci_flag": clean_data["ci_flag"]
        }
        
        # Match the exact structure used in main optimization (line 5525-5540)
        sources.append({
            "source_id": clean_data["source_id"],
            "features": features,
            "reliability_series": [],  # Required by model
            "recourse_rules": recourse_rules  # Include decision thresholds
        })
    
    # Run batch optimization if we have valid sources
    if sources:
        try:
            payload = {
                "sources": sources,
                "seed": 42
            }
            
            result = run_optimization(payload)
            ml_policy = result.get("policies", {}).get("ml_tssp", [])
            
            # Extract individual source results matching single source format
            for policy_item in ml_policy:
                # Handle both 'task' (fallback) and 'tasks' (API) formats
                task_data = policy_item.get("tasks") or policy_item.get("task")
                if task_data and not isinstance(task_data, list):
                    task_data = [task_data]
                elif not task_data:
                    task_data = []
                
                source_result = {
                    "source_id": policy_item.get("source_id", "UNKNOWN"),
                    "ml_reliability": float(policy_item.get("reliability", 0.0)),
                    "deception_confidence": float(policy_item.get("deception", 0.0)),
                    "decision": policy_item.get("action", "unknown"),
                    "tssp_allocation": task_data,
                    "expected_risk": float(policy_item.get("expected_risk", 0.0)),
                    "optimization_score": float(policy_item.get("score", 0.0)),
                    "features": sources[len(results)]["features"] if len(results) < len(sources) else {},
                    "recourse_rules": recourse_rules,
                    "api_method": "api" if not result.get("_using_fallback") else "fallback",
                    "is_taskable": policy_item.get("action", "unknown") in ["task", "flag_and_task", "flag_for_ci"]
                }
                results.append(source_result)
        except Exception as e:
            errors.append({"row": "batch", "source_id": "ALL", "errors": [f"Batch processing error: {str(e)}"]})
    
    return results, errors


def render_streamlit_app():
    """Main Streamlit application with left-side controls."""
    _init_streamlit()
    
    # ======================================================
    # STARTUP STATUS BANNER
    # ======================================================
    startup_placeholder = st.empty()
    with startup_placeholder.container():
        st.info("🚀 **Loading ML-TSSP Dashboard...** Initializing components...")
    
    try:
        # ======================================================
        # HEALTH CHECK STATUS
        # ======================================================
        with st.expander("🔍 System Health Check", expanded=False):
            health_status = _check_system_health()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Core Packages", 
                    f"{health_status['packages_loaded']}/{health_status['packages_total']}",
                    "✅ OK" if health_status['packages_ok'] else "⚠️ Issues"
                )
            with col2:
                st.metric(
                    "API Status",
                    "Available" if health_status['api_available'] else "Fallback",
                    "✅" if health_status['api_available'] else "⚠️"
                )
            with col3:
                st.metric(
                    "Model Files",
                    f"{health_status['models_found']}/{health_status['models_expected']}",
                    "✅ OK" if health_status['models_ok'] else "⚠️ Missing"
                )
            
            if health_status['warnings']:
                st.warning("\n".join(health_status['warnings']))
            
            with st.expander("📋 Detailed Status"):
                st.json(health_status['details'])
        
        # Clear startup message
        startup_placeholder.empty()
        
    except Exception as e:
        startup_placeholder.error(f"❌ Startup Error: {str(e)}")
        st.exception(e)
    
    # ======================================================
    # AUTHENTICATION CHECK
    # ======================================================
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        _render_login_page()
        return
    
    # ======================================================
    # LOGOUT BUTTON (Top Right - Professional) - STICKY
    # ======================================================
    st.markdown("""
    <style>
    /* Sticky header container */
    .sticky-header-wrapper {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #e0e7ff 0%, #dbeafe 50%, #e0f2fe 100%);
        padding: 0.5rem 0;
        margin: -1rem -1rem 1rem -1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .sticky-logout-bar {
        max-width: 100%;
        padding: 0 1rem;
    }
    
    .sticky-hero {
        position: sticky;
        top: 60px;
        z-index: 998;
        margin-bottom: 1rem;
    }
    
    /* Professional logout button styling */
    .stButton button[kind="secondary"] {
        height: 40px !important;
        min-height: 40px !important;
        padding: 0 20px !important;
        font-size: 14px !important;
        font-weight: 200 !important;
        border-radius: 9px !important;
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(100, 116, 139, 0.25) !important;
        background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
    }
    
    .stButton button[kind="secondary"]:active {
        transform: scale(0.97) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
    }
    
    .logout-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0;
        margin-bottom: 0.8rem;
    }
    
    .user-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.9rem;
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(8px);
        border-radius: 8px;
        border: 1px solid rgba(203, 213, 225, 0.6);
        font-size: 13px;
        color: #475569;
        font-weight: 500;
    }
    
    .user-badge .username {
        color: #1e40af;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sticky header wrapper start
    st.markdown('<div class="sticky-header-wrapper"><div class="sticky-logout-bar">', unsafe_allow_html=True)
    
    # Top-right user info and logout
    st.markdown(f"""
    <div class="logout-container">
        <div class="user-badge">
            <span>👤</span>
            <span>Logged in as <span class="username">{st.session_state.username.title()}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_spacer, col_logout = st.columns([5, 1])
    with col_logout:
        if st.button("🚪 Logout", type="secondary", use_container_width=True, key="logout_btn_top"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)  # Close sticky header wrapper
    
    # ======================================================
    # HEADER (STICKY HERO)
    # ======================================================
    
    hero_bg = _load_header_background()
    st.markdown(f"""
    <div class="sticky-hero">
        <div style="
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.1);
            background-image:
                {hero_bg};
            background-size: cover;
            background-position: center;
        ">
            <div style="padding: 2rem 2rem; text-align: center; color: #f8fafc;">
                <h1 style="
                    margin: 0;
                    font-size: 26px;
                    font-weight: 800;
                    letter-spacing: -0.5px;
                    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.45);
                ">
                    🛰️ Hybrid HUMINT Sources Performance Optimization Engine
                </h1>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-frame">', unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f4f7fb 0%, #e5e9f1 100%);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        box-shadow: inset 0 1px 4px rgba(15, 23, 42, 0.08);">
        <p style="
            font-size:16px;
            margin:0;
            text-align:center;
            line-height:1.75;
            font-weight:500;
            color:#0F2A44;">
            Supports intelligence operations through a unified framework integrating XGBoost-based behavioral classification,
            GRU-driven forecasting of source reliability and deception, and two-stage stochastic optimization for risk-aware
            resource allocation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ======================================================
    # OPERATIONAL OVERVIEW
    # ======================================================
    st.markdown("""
    <div style="
        background: linear-gradient(118deg, #0b1736 0%, #15306c 45%, #1d4ad1 100%);
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 24px rgba(8, 15, 35, 0.55);
        padding: 1rem 1.7rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 16px 16px;">
        <p style="margin: 0; font-size: 11.5px; font-weight: 650; color: #e3e9ff; text-transform: uppercase; letter-spacing: 0.6px;">
            📊 Operational Overview
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if "sources_count" not in st.session_state:
        st.session_state.sources_count = 3
    if "results" not in st.session_state:
        st.session_state.results = None
    
    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
    with ov_col1:
        render_kpi_indicator("🧠 Total Sources", st.session_state.get("sources_count", 0), key="kpi_total_sources_overview")
    with ov_col2:
        ml_policy = st.session_state.get("results", {}).get("policies", {}).get("ml_tssp", []) if st.session_state.get("results") else []
        avg_risk = np.mean([a.get("expected_risk", 0.5) for a in ml_policy]) if ml_policy else None
        render_kpi_indicator("📉 Avg Risk", avg_risk, suffix="", key="kpi_avg_risk_overview")
    with ov_col3:
        high_risk = sum(1 for a in ml_policy if a.get("expected_risk", 0) > 0.6) if ml_policy else None
        render_kpi_indicator("⚠️ High Risk", high_risk, note="Sources > 0.6 risk", key="kpi_high_risk_overview")
    with ov_col4:
        render_kpi_indicator("🎯 Tasks", len(TASK_ROSTER), note="Available slots", key="kpi_tasks_overview")
    
    st.divider()
    
    # ======================================================
    # TWO-COLUMN LAYOUT: LEFT CONTROLS + RIGHT CONTENT
    # ======================================================
    nav_labels = [
        "📋 Source Profiles",
        "📈 Policy Insights",
        "💰 EVPI Focus",
        "🔬 Stress Lab"
    ]
    nav_lookup = {
        "📋 Source Profiles": "profiles",
        "📈 Policy Insights": "policies",
        "💰 EVPI Focus": "evpi",
        "🔬 Stress Lab": "stress"
    }
    nav_choice = st.radio("Navigate dashboard", nav_labels, horizontal=True, key="nav_pills",
                          label_visibility="hidden")
    nav_key = nav_lookup[nav_choice]
    
    with st.container():
        filt1, filt2, filt3 = st.columns([1.2, 1, 1])
        with filt1:
            scenario_preset = st.selectbox(
                "Scenario preset",
                ["Normal Intelligence Environment", "High Threat Environment", "Denied/Contested Environment"],
                key="scenario_preset")
        with filt2:
            review_horizon = st.slider("Review horizon (days)", 14, 180, 60, key="review_horizon")
        with filt3:
            priority_tag = st.multiselect("Priority tags", ["SIGINT", "CI", "Liaison"], default=["SIGINT"],
                                          key="priority_tags")
        st.session_state["scenario_filters"] = {
            "preset": scenario_preset,
            "horizon": review_horizon,
            "tags": priority_tag
        }
    
    with st.sidebar:
        st.markdown("""
        <div class="control-panel">
            <div class="control-panel-header" style="color: #6b21a8;">⚙️ Configuration</div>
        """, unsafe_allow_html=True)
        
        # ========== OPERATIONAL MODE PRESETS ==========
        with st.expander("🎯 OPERATIONAL MODE", expanded=False):
            preset_mode = st.radio(
            "Select policy mode",
            ["🟢 Conservative", "🟡 Balanced", "🔴 Aggressive", "⚙️ Custom"],
            index=1,
            key="preset_mode",
            label_visibility="collapsed",
            horizontal=False
        )
        
        # Set defaults based on mode
        if preset_mode == "🟢 Conservative":
            default_rel_disengage, default_rel_flag = 0.45, 0.60
            default_dec_disengage, default_dec_escalate = 0.65, 0.50
            default_sources = 50  # Conservative approach with moderate source pool
        elif preset_mode == "🟡 Balanced":
            default_rel_disengage, default_rel_flag = 0.35, 0.50
            default_dec_disengage, default_dec_escalate = 0.75, 0.60
            default_sources = 65  # Balanced approach with substantial source pool
        elif preset_mode == "🔴 Aggressive":
            default_rel_disengage, default_rel_flag = 0.25, 0.40
            default_dec_disengage, default_dec_escalate = 0.85, 0.70
            default_sources = 80  # Aggressive approach utilizing full source capacity
        else:  # Custom
            default_rel_disengage = st.session_state.get("rel_disengage_slider", 0.35)
            default_rel_flag = st.session_state.get("rel_ci_flag_slider", 0.50)
            default_dec_disengage = st.session_state.get("dec_disengage_slider", 0.75)
            default_dec_escalate = st.session_state.get("dec_ci_flag_slider", 0.60)
            default_sources = st.session_state.sources_count
        
        # ========== SOURCE DATA INPUT SECTION ==========
        with st.expander("📥 SOURCE DATA INPUT", expanded=False):
            # Initialize session state for input mode
            if "input_mode" not in st.session_state:
                st.session_state.input_mode = "Single Source"
            if "custom_sources" not in st.session_state:
                st.session_state.custom_sources = []
            
            # Tab/Mode selection
            input_mode = st.radio(
                "Input mode",
                ["🔬 Single Source Test", "📊 Batch Upload"],
                key="input_mode_selector",
                label_visibility="collapsed",
                horizontal=True,
                help="Choose single source for testing or batch upload for multiple sources"
            )
            
            st.markdown("<div style='margin: 0.8rem 0;'></div>", unsafe_allow_html=True)
            
            # ========== SINGLE SOURCE ENTRY MODE ==========
            if input_mode == "🔬 Single Source Test":
                st.markdown("""
                <p style='font-size: 10px; color: #92400e; margin: 0 0 0.6rem 0; font-style: italic;'>
                    Manual entry for testing and "what-if" scenarios
                </p>
                """, unsafe_allow_html=True)
                
                with st.form("single_source_form"):
                    # Source ID at the top
                    source_id = st.text_input(
                        "Source ID",
                        value=f"SRC_TEST_{len(st.session_state.custom_sources)+1:03d}",
                        help="Unique identifier for the source"
                    )
                    
                    st.markdown("<div style='margin: 0.4rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Balanced 2-column layout for the 6 input features
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<p style='font-size: 10px; font-weight: 600; color: #1e40af; margin: 0 0 0.3rem 0;'>Core Performance Metrics</p>", unsafe_allow_html=True)
                        
                        task_success = st.slider(
                            "Task Success Rate",
                            0.0, 1.0, 0.0, 0.05,
                            help="Historical success rate in completing assigned tasks"
                        )
                        
                        corroboration = st.slider(
                            "Corroboration Score",
                            0.0, 1.0, 0.0, 0.05,
                            help="Degree to which source reports are independently verified"
                        )
                        
                        timeliness = st.slider(
                            "Report Timeliness",
                            0.0, 1.0, 0.0, 0.05,
                            help="Consistency in delivering reports on time"
                        )
                    
                    with col2:
                        st.markdown("<p style='font-size: 10px; font-weight: 600; color: #1e40af; margin: 0 0 0.3rem 0;'>Risk & Confidence Indicators</p>", unsafe_allow_html=True)
                        
                        handler_conf = st.slider(
                            "Handler Confidence",
                            0.0, 1.0, 0.0, 0.05,
                            help="Handler's confidence in the source based on experience"
                        )
                        
                        deception = st.slider(
                            "Deception Score",
                            0.0, 1.0, 0.0, 0.05,
                            help="Indicators of potential deception or manipulation"
                        )
                        
                        ci_flag = st.selectbox(
                            "CI Flag",
                            [0, 1],
                            index=0,
                            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
                            help="Counterintelligence concern flag (0=No, 1=Yes)"
                        )
                    
                    # Optional fields
                    with st.expander("📝 Optional Fields"):
                        behavior = st.selectbox(
                            "Behavior Category",
                            ["Unknown", "Cooperative", "Uncertain", "Coerced"],
                            help="Behavioral classification of the source"
                        )
                        
                        access = st.selectbox(
                            "Access Level",
                            ["Unknown", "Limited", "Moderate", "Extensive"],
                            help="Source's access to target information"
                        )
                        
                        notes = st.text_area(
                            "Handler Notes",
                            placeholder="Additional context or observations...",
                            height=80
                        )
                    
                    col_submit, col_clear = st.columns(2)
                    with col_submit:
                        submit_single = st.form_submit_button("▶ Run Single Source", type="primary", use_container_width=True)
                    with col_clear:
                        clear_results = st.form_submit_button("↺ Clear Results", use_container_width=True)
                
                if clear_results:
                    if "single_source_result" in st.session_state:
                        del st.session_state.single_source_result
                    st.rerun()
                
                if submit_single:
                    # Build source data with all 6 input features
                    source_data = {
                        "source_id": source_id,
                        "task_success_rate": task_success,
                        "corroboration_score": corroboration,
                        "report_timeliness": timeliness,
                        "handler_confidence": handler_conf,
                        "deception_score": deception,
                        "ci_flag": ci_flag
                    }
                    
                    if behavior != "Unknown":
                        source_data["behavior_category"] = behavior
                    if access != "Unknown":
                        source_data["access_level"] = access
                    if notes:
                        source_data["handler_notes"] = notes
                    
                    # Validate
                    clean_data, errors = _validate_source_schema(source_data)
                    
                    if errors:
                        st.error("**Validation Errors:**")
                        for err in errors:
                            st.error(f"• {err}")
                    else:
                        # Build recourse rules from sliders (will be defined below)
                        recourse_rules = {
                            "rel_disengage": st.session_state.get("rel_disengage_slider", 0.35),
                            "rel_ci_flag": st.session_state.get("rel_ci_flag_slider", 0.50),
                            "dec_disengage": st.session_state.get("dec_disengage_slider", 0.75),
                            "dec_ci_flag": st.session_state.get("dec_ci_flag_slider", 0.60)
                        }
                        
                        with st.spinner("🔄 Processing source through ML-TSSP pipeline..."):
                            result, error = _process_single_source(clean_data, recourse_rules)
                        
                        if error:
                            st.error(f"**Processing Error:** {error}")
                        else:
                            st.session_state.single_source_result = result
                            st.success("✅ Source processed successfully!")
                            st.rerun()
                
                # Display results if available
                if "single_source_result" in st.session_state:
                    result = st.session_state.single_source_result
                    
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); 
                                border-radius: 12px; padding: 1.2rem; margin: 1rem 0; 
                                border-left: 5px solid #10b981; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <h3 style='margin: 0; font-size: 12px; font-weight: 700; color: #047857;'>
                            ✅ Source Analysis Complete
                        </h3>
                        <p style='margin: 0.3rem 0 0 0; font-size: 8px; color: #065f46; opacity: 0.9;'>
                            ML-TSSP Pipeline • Real-time Decision Support
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key Metrics Dashboard
                    st.markdown("""
                    <style>
                    /* Metric card visual hierarchy */
                    [data-testid="stMetricValue"] {
                        font-size: 12px;
                    }
                    [data-testid="stMetricLabel"] {
                        font-size: 8px;
                    }
                    /* Primary metric (Reliability) - stronger */
                    div[data-testid="column"]:nth-child(1) [data-testid="stMetricLabel"] {
                        font-weight: 600;
                        color: #1e3a8a;
                        font-size: 8px;
                    }
                    div[data-testid="column"]:nth-child(1) [data-testid="stMetricValue"] {
                        font-weight: 700;
                        font-size: 14px;
                    }
                    /* Secondary metrics - softer */
                    div[data-testid="column"]:nth-child(2) [data-testid="stMetricLabel"],
                    div[data-testid="column"]:nth-child(3) [data-testid="stMetricLabel"],
                    div[data-testid="column"]:nth-child(4) [data-testid="stMetricLabel"] {
                        font-weight: 400;
                        color: #6b7280;
                        font-size: 8px;
                    }
                    div[data-testid="column"]:nth-child(2) [data-testid="stMetricValue"],
                    div[data-testid="column"]:nth-child(3) [data-testid="stMetricValue"],
                    div[data-testid="column"]:nth-child(4) [data-testid="stMetricValue"] {
                        font-weight: 500;
                        font-size: 12px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        rel_val = result.get('ml_reliability', 0.0)
                        rel_delta = rel_val - 0.5
                        st.metric(
                            "Reliability", 
                            f"{rel_val:.1%}",
                            delta=f"{rel_delta:+.1%}",
                            help="ML-estimated source reliability score"
                        )
                    
                    with metric_col2:
                        dec_val = result.get('deception_confidence', 0.0)
                        dec_delta = dec_val - 0.5
                        st.metric(
                            "Deception Risk", 
                            f"{dec_val:.1%}",
                            delta=f"{dec_delta:+.1%}",
                            delta_color="inverse",
                            help="Estimated probability of deceptive reporting"
                        )
                    
                    with metric_col3:
                        decision = result.get('decision', 'unknown')
                        decision_display = {
                            'task': 'Task',
                            'flag_and_task': 'Flag & Task',
                            'flag_for_ci': 'CI Review',
                            'disengage': 'Disengage'
                        }.get(decision, 'Unknown')
                        st.metric(
                            "Decision", 
                            decision_display,
                            help="TSSP operational decision"
                        )
                    
                    with metric_col4:
                        tasks = result.get('tssp_allocation', [])
                        task_count = len(tasks) if tasks else 0
                        st.metric(
                            "Tasks", 
                            str(task_count),
                            help=f"Assigned: {', '.join(tasks[:3])}..." if task_count > 3 else f"Assigned: {', '.join(tasks)}" if task_count > 0 else "No tasks assigned"
                        )
                    
                    # Add to simulation button
                    st.markdown("---")
                    col_add, col_export = st.columns(2)
                    with col_add:
                        if st.button("➕ Add to Main Simulation", use_container_width=True, help="Add this source to the main simulation pool"):
                            # Add to custom sources list
                            if "custom_sources_pool" not in st.session_state:
                                st.session_state.custom_sources_pool = []
                            
                            # Create source entry
                            custom_source = {
                                "source_id": result.get('source_id'),
                                "features": result.get('features'),
                                "ml_reliability": result.get('ml_reliability'),
                                "deception_confidence": result.get('deception_confidence'),
                                "decision": result.get('decision')
                            }
                            
                            # Check if already exists
                            existing_ids = [s.get('source_id') for s in st.session_state.custom_sources_pool]
                            if custom_source['source_id'] not in existing_ids:
                                st.session_state.custom_sources_pool.append(custom_source)
                                st.success(f"✅ {custom_source['source_id']} added to simulation pool!")
                                st.info(f"📊 Pool now contains {len(st.session_state.custom_sources_pool)} custom sources")
                            else:
                                st.warning("⚠️ Source already in simulation pool")
                    
                    with col_export:
                        # Export single source as CSV
                        export_df = pd.DataFrame([{
                            "source_id": result.get('source_id'),
                            "task_success_rate": result.get('features', {}).get('task_success_rate'),
                            "corroboration_score": result.get('features', {}).get('corroboration_score'),
                            "report_timeliness": result.get('features', {}).get('report_timeliness'),
                            "ml_reliability": result.get('ml_reliability'),
                            "deception_confidence": result.get('deception_confidence'),
                            "decision": result.get('decision')
                        }])
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export as CSV",
                            data=csv,
                            file_name=f"{result.get('source_id', 'source')}_result.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Enhanced detailed explanation
                    with st.expander("🔍 Detailed Analysis & Insights", expanded=False):
                        
                        # Add custom CSS for calmer tabs
                        st.markdown("""
                        <style>
                        /* Calmer tab styling */
                        .stTabs [data-baseweb="tab-list"] {
                            gap: 8px;
                            border-bottom: 2px solid #e5e7eb;
                            padding-bottom: 0;
                        }
                    .stTabs [data-baseweb="tab"] {
                        background-color: #f9fafb;
                        color: #6b7280;
                        border-radius: 8px 8px 0 0;
                        padding: 0.5rem 1rem;
                        font-weight: 500;
                        border: 1px solid #e5e7eb;
                        border-bottom: none;
                    }
                    .stTabs [aria-selected="true"] {
                        background-color: white;
                        color: #1e3a8a;
                        font-weight: 600;
                        border-color: #e5e7eb;
                        border-bottom: 2px solid white;
                        margin-bottom: -2px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Subtle divider before tabs
                    st.markdown("<div style='margin: 1rem 0 1.5rem 0; border-top: 1px solid #e5e7eb;'></div>", unsafe_allow_html=True)
                    
                    # Tab-based navigation for cleaner organization
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Breakdown", "🎯 Decision Logic", "📈 Performance", "🔧 Technical"])
                    
                    with tab1:
                        st.markdown("<h3 style='font-size: 12px; font-weight: 700; color: #1e3a8a;'>📊 Score Analysis</h3>", unsafe_allow_html=True)
                        
                        features = result.get('features', {})
                        rel = result.get('ml_reliability', 0.0)
                        dec = result.get('deception_confidence', 0.0)
                        
                        # Interactive gauges side by side
                        gauge_col1, gauge_col2 = st.columns(2)
                    
                        with gauge_col1:
                            # Reliability gauge with cleaner design
                            fig_rel = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=rel * 100,
                                number={'suffix': '%', 'font': {'size': 16}},
                                title={'text': "<b>Reliability Score</b><br><span style='font-size:8px; color:#6b7280'>ML Assessment</span>", 'font': {'size': 12}},

                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'ticksuffix': '%'},
                                    'bar': {'color': "#10b981", 'thickness': 0.8},
                                    'bgcolor': "white",
                                    'steps': [
                                        {'range': [0, 35], 'color': "#fee2e2"},
                                        {'range': [35, 50], 'color': "#fef3c7"},
                                        {'range': [50, 100], 'color': "#dcfce7"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#dc2626", 'width': 3},
                                        'thickness': 0.8,
                                        'value': result.get('recourse_rules', {}).get('rel_ci_flag', 0.50) * 100
                                    }
                                }
                            ))
                            fig_rel.update_layout(
                                height=250,
                                margin=dict(l=20, r=20, t=60, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'family': "system-ui, -apple-system, sans-serif"}
                            )
                            st.plotly_chart(fig_rel, use_container_width=True)
                            
                            # Classification badge
                            rel_class = result.get('confidence_metrics', {}).get('reliability_confidence', 'medium')
                            class_color = {'high': '#10b981', 'medium': '#f59e0b', 'low': '#ef4444'}.get(rel_class, '#6b7280')
                            st.markdown(f"""
                            <div style='text-align: center; padding: 0.5rem; background: {class_color}20; 
                                        border-radius: 6px; border: 1px solid {class_color};'>
                                <span style='color: {class_color}; font-weight: 600; font-size: 8px;'>
                                    {rel_class.upper()} CONFIDENCE
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with gauge_col2:
                            # Deception risk gauge with cleaner design
                            fig_dec = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=dec * 100,
                                number={'suffix': '%', 'font': {'size': 32}},
                                title={'text': "<b>Deception Risk</b><br><span style='font-size:12px; color:#6b7280'>ML Assessment</span>", 'font': {'size': 16}},

                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'ticksuffix': '%'},
                                    'bar': {'color': "#ef4444", 'thickness': 0.8},
                                    'bgcolor': "white",
                                    'steps': [
                                        {'range': [0, 30], 'color': "#dcfce7"},
                                        {'range': [30, 60], 'color': "#fef3c7"},
                                        {'range': [60, 100], 'color': "#fee2e2"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#dc2626", 'width': 3},
                                        'thickness': 0.8,
                                        'value': result.get('recourse_rules', {}).get('dec_ci_flag', 0.60) * 100
                                    }
                                }
                            ))
                            fig_dec.update_layout(
                                height=250,
                                margin=dict(l=20, r=20, t=60, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'family': "system-ui, -apple-system, sans-serif"}
                            )
                            st.plotly_chart(fig_dec, use_container_width=True)
                            
                            # Risk level badge
                            dec_class = result.get('confidence_metrics', {}).get('deception_confidence', 'medium_risk')
                            risk_display = dec_class.replace('_', ' ').upper()
                            risk_color = {'low': '#10b981', 'low_risk': '#10b981', 'medium': '#f59e0b', 'medium_risk': '#f59e0b', 'high': '#ef4444', 'high_risk': '#ef4444'}.get(dec_class, '#6b7280')
                            st.markdown(f"""
                            <div style='text-align: center; padding: 0.5rem; background: {risk_color}20; 
                                        border-radius: 6px; border: 1px solid {risk_color};'>
                                <span style='color: {risk_color}; font-weight: 600; font-size: 8px;'>
                                    {risk_display}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Feature contribution visualization
                        st.markdown("---")
                        st.markdown("**Feature Impact Analysis**")
                        
                        tsr = features.get('task_success_rate', 0)
                        cor = features.get('corroboration_score', 0)
                        tim = features.get('report_timeliness', 0)
                        
                        fig_features = go.Figure()
                        
                        feature_data = [
                            ('Task Success Rate', tsr, 0.30),
                            ('Corroboration', cor, 0.35),
                            ('Timeliness', tim, 0.25)
                        ]
                        
                        for fname, fval, weight in feature_data:
                            contribution = fval * weight
                            color = '#10b981' if fval >= 0.7 else '#f59e0b' if fval >= 0.5 else '#ef4444'
                            
                            fig_features.add_trace(go.Bar(
                                y=[fname],
                                x=[contribution],
                                orientation='h',
                                name=fname,
                                marker=dict(color=color),
                                text=f"{fval:.1%} × {weight:.0%} = {contribution:.3f}",
                                textposition='outside',
                                hovertemplate=f"<b>{fname}</b><br>Value: {fval:.1%}<br>Weight: {weight:.0%}<br>Contribution: {contribution:.3f}<extra></extra>"
                            ))
                        
                        fig_features.update_layout(
                            title="Weighted Contributions to Reliability Score",
                            xaxis_title="Contribution to Final Score",
                            height=220,
                            margin=dict(l=20, r=20, t=40, b=40),
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(gridcolor='#e5e7eb'),
                            font={'family': "system-ui, -apple-system, sans-serif", 'size': 8}
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    with tab2:
                        st.markdown("<h3 style='font-size: 12px; font-weight: 700; color: #1e3a8a;'>🎯 Decision Logic</h3>", unsafe_allow_html=True)
                        
                        decision = result.get('decision', 'unknown')
                        action_reason = result.get('action_reason', 'No reason provided')
                        tasks = result.get('tssp_allocation', [])
                        
                        # Decision card
                        decision_config = {
                            'task': {
                                'title': '✅ CLEARED FOR TASKING',
                                'color': '#10b981',
                                'bg': '#ecfdf5',
                                'action': 'Source fully operational under standard protocols',
                                'icon': '✅'
                            },
                            'flag_and_task': {
                                'title': '⚠️ TASKABLE WITH OVERSIGHT',
                                'color': '#f59e0b',
                                'bg': '#fffbeb',
                                'action': 'Source operational but requires enhanced monitoring',
                                'icon': '⚠️'
                            },
                            'flag_for_ci': {
                                'title': '🚩 FLAGGED FOR CI REVIEW',
                                'color': '#f59e0b',
                                'bg': '#fffbeb',
                                'action': 'Enhanced counterintelligence review required',
                                'icon': '🚩'
                            },
                            'disengage': {
                                'title': '⛔ DISENGAGED',
                                'color': '#ef4444',
                                'bg': '#fef2f2',
                                'action': 'Source removed from operational consideration',
                                'icon': '⛔'
                            }
                        }
                        
                        config = decision_config.get(decision, {
                            'title': '❓ UNKNOWN STATUS',
                            'color': '#6b7280',
                            'bg': '#f3f4f6',
                            'action': 'Decision could not be determined',
                            'icon': '❓'
                        })
                        
                        st.markdown(f"""
                        <div style='background: {config['bg']}; border-radius: 12px; padding: 1.5rem; 
                                    border-left: 5px solid {config['color']}; margin-bottom: 1rem;'>
                            <h3 style='color: {config['color']}; margin: 0 0 0.5rem 0; font-size: 12px;'>
                                {config['icon']} {config['title']}
                            </h3>
                            <p style='margin: 0; color: #374151; font-size: 8px; line-height: 1.6;'>
                                <b>Rationale:</b> {action_reason}
                            </p>
                            <p style='margin: 0.5rem 0 0 0; color: #374151; font-size: 8px;'>
                                <b>Action:</b> {config['action']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Task allocation
                        if tasks:
                            st.markdown(f"**📋 Assigned Tasks ({len(tasks)})**")
                            task_display = ', '.join(tasks[:8])
                            if len(tasks) > 8:
                                task_display += f" +{len(tasks)-8} more"
                            st.info(task_display)
                        else:
                            st.markdown("**📋 Task Assignment:** None")
                        
                        # Threshold comparison visual
                        st.markdown("---")
                        st.markdown("**Threshold Analysis**")
                        
                        rules = result.get('recourse_rules', {})
                        
                        # Create threshold comparison chart
                        fig_thresh = go.Figure()
                        
                        # Add reliability bar
                        fig_thresh.add_trace(go.Bar(
                            y=['Reliability'],
                            x=[rel],
                            name='Current Value',
                            orientation='h',
                            marker=dict(color='#10b981'),
                            text=f"{rel:.2%}",
                            textposition='inside'
                        ))
                        
                        # Add deception bar
                        fig_thresh.add_trace(go.Bar(
                            y=['Deception Risk'],
                            x=[dec],
                            name='Current Value',
                            orientation='h',
                            marker=dict(color='#ef4444'),
                            text=f"{dec:.2%}",
                            textposition='inside',
                            showlegend=False
                        ))
                        
                        # Add threshold lines
                        fig_thresh.add_vline(x=rules.get('rel_ci_flag', 0.50), line_dash="dash", 
                                            line_color="#dc2626", annotation_text="Rel Flag Threshold")
                        fig_thresh.add_vline(x=rules.get('dec_ci_flag', 0.60), line_dash="dash", 
                                            line_color="#dc2626", annotation_text="Dec Flag Threshold")
                        
                        fig_thresh.update_layout(
                            title="Score vs. Decision Thresholds",
                            xaxis_title="Score",
                            xaxis=dict(range=[0, 1], tickformat=".0%"),
                            height=200,
                            margin=dict(l=20, r=20, t=40, b=40),
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'family': "system-ui, -apple-system, sans-serif", 'size': 8}
                        )
                        st.plotly_chart(fig_thresh, use_container_width=True)
                    
                    with tab3:
                        st.markdown("<h3 style='font-size: 12px; font-weight: 700; color: #1e3a8a;'>📈 Performance Metrics</h3>", unsafe_allow_html=True)
                        
                        # Key performance indicators
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        
                        risk = result.get('expected_risk', 0.5)
                        opt_score = result.get('optimization_score', 0)
                        
                        with perf_col1:
                            st.metric(
                                "Expected Risk",
                                f"{risk:.1%}",
                                delta=f"{(0.5-risk)*100:+.1f}%",
                                delta_color="inverse",
                                help="Predicted operational risk level"
                            )
                        
                        with perf_col2:
                            st.metric(
                                "Optimization Score",
                                f"{opt_score:.3f}",
                                help="TSSP optimization objective value"
                            )
                        
                        with perf_col3:
                            is_taskable = result.get('is_taskable', False)
                            st.metric(
                                "Operational Status",
                                "✅ Taskable" if is_taskable else "⛔ Not Taskable",
                                help="Whether source is cleared for task assignment"
                            )
                        
                        # EMV comparison
                        if result.get('emv'):
                            st.markdown("---")
                            st.markdown("**Expected Monetary Value (EMV) Comparison**")
                            st.caption("Comparing ML-TSSP optimization against baseline methods")
                            
                            emv = result.get('emv', {})
                            ml_val = emv.get('ml_tssp', 0)
                            det_val = emv.get('deterministic', 0)
                            uni_val = emv.get('uniform', 0)
                            
                            fig_emv = go.Figure()
                            
                            methods = ['ML-TSSP', 'Deterministic', 'Uniform']
                            values = [ml_val, det_val, uni_val]
                            colors = ['#10b981', '#f59e0b', '#6b7280']
                            
                            fig_emv.add_trace(go.Bar(
                                x=methods,
                                y=values,
                                marker=dict(color=colors),
                                text=[f"{v:.4f}" for v in values],
                                textposition='outside'
                            ))
                            
                            fig_emv.update_layout(
                                title="EMV Method Comparison",
                                yaxis_title="Expected Monetary Value",
                                height=280,
                                margin=dict(l=20, r=20, t=40, b=40),
                                showlegend=False,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                yaxis=dict(gridcolor='#e5e7eb'),
                                font={'family': "system-ui, -apple-system, sans-serif", 'size': 8}
                            )
                            st.plotly_chart(fig_emv, use_container_width=True)
                            
                            # Performance delta
                            ml_vs_det = ((ml_val - det_val) / det_val * 100) if det_val != 0 else 0
                            ml_vs_uni = ((ml_val - uni_val) / uni_val * 100) if uni_val != 0 else 0
                            
                            delta_col1, delta_col2 = st.columns(2)
                            with delta_col1:
                                st.metric("vs. Deterministic", f"{ml_vs_det:+.1f}%")
                            with delta_col2:
                                st.metric("vs. Uniform", f"{ml_vs_uni:+.1f}%")
                    
                    with tab4:
                        st.markdown("<h3 style='font-size: 12px; font-weight: 700; color: #1e3a8a;'>🔧 Technical Details</h3>", unsafe_allow_html=True)
                        
                        # Source inputs summary
                        st.markdown("**Input Features**")
                        features = result.get('features', {})
                        
                        feat_df_data = []
                        for k, v in features.items():
                            feat_df_data.append({
                                'Feature': k.replace('_', ' ').title(),
                                'Value': f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                            })
                        
                        if feat_df_data:
                            feat_df = pd.DataFrame(feat_df_data)
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)
                        
                        # Processing info
                        st.markdown("---")
                        st.markdown("**Processing Information**")
                        
                        api_method = result.get('api_method', 'unknown')
                        method_display = {
                            'local_ml_pipeline': 'Local ML Pipeline',
                            'fallback': 'Local Fallback',
                            'api': 'Backend API'
                        }.get(api_method, 'Unknown')
                        
                        st.info(f"Processing Method: **{method_display}**")
                        st.markdown(f"Source ID: `{result.get('source_id', 'Unknown')}`")
                        
                        # Confidence metrics
                        conf_metrics = result.get('confidence_metrics', {})
                        if conf_metrics:
                            st.markdown("---")
                            st.markdown("**Confidence Assessment**")
                            
                            rel_conf = conf_metrics.get('reliability_confidence', 'unknown').replace('_', ' ').title()
                            dec_conf = conf_metrics.get('deception_confidence', 'unknown').replace('_', ' ').title()
                            overall = conf_metrics.get('overall_confidence', 'unknown').replace('_', ' ').title()
                            
                            st.markdown(f"Reliability: **{rel_conf}**")
                            st.markdown(f"Deception Assessment: **{dec_conf}**")
                            st.markdown(f"Overall Status: **{overall}**")
                        
                        # Recourse rules
                        rules = result.get('recourse_rules', {})
                        if rules:
                            st.markdown("---")
                            st.markdown("**Decision Thresholds**")
                            
                            thresh_col1, thresh_col2 = st.columns(2)
                            with thresh_col1:
                                st.markdown("*Reliability Thresholds:*")
                                st.markdown(f"Disengage: `{rules.get('rel_disengage', 0):.2f}`")
                                st.markdown(f"CI Flag: `{rules.get('rel_ci_flag', 0):.2f}`")
                            with thresh_col2:
                                st.markdown("*Deception Thresholds:*")
                                st.markdown(f"CI Flag: `{rules.get('dec_ci_flag', 0):.2f}`")
                                st.markdown(f"Disengage: `{rules.get('dec_disengage', 0):.2f}`")
                    
                        # Raw JSON export
                        st.markdown("---")
                        st.markdown("**Export Full Result**")
                        
                        try:
                            def serialize_value(val):
                                """Convert value to JSON-serializable format."""
                                if isinstance(val, (np.integer, np.floating)):
                                    return float(val)
                                elif isinstance(val, np.ndarray):
                                    return val.tolist()
                                elif isinstance(val, dict):
                                    return {k: serialize_value(v) for k, v in val.items()}
                                elif isinstance(val, (list, tuple)):
                                    return [serialize_value(v) for v in val]
                                elif val is None:
                                    return None
                                elif isinstance(val, (str, int, float, bool)):
                                    return val
                                else:
                                    return str(val)
                            
                            clean_result = {}
                            for k, v in result.items():
                                if k not in ['full_policy']:  # Skip large nested objects
                                    try:
                                        clean_result[k] = serialize_value(v)
                                    except Exception:
                                        clean_result[k] = str(v)
                            
                            # Export options
                            json_col1, json_col2 = st.columns(2)
                            
                            with json_col1:
                                # View JSON
                                if st.button("👁️ View JSON", use_container_width=True):
                                    st.json(clean_result)
                            
                            with json_col2:
                                # Download JSON
                                json_str = json.dumps(clean_result, indent=2)
                                st.download_button(
                                    label="💾 Download JSON",
                                    data=json_str,
                                    file_name=f"{result.get('source_id', 'source')}_result.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            
                        except Exception as e:
                            st.error(f"**JSON Rendering Error:** {str(e)}")
                            
                            # Show error details
                            with st.expander("🔍 Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
                            
                            st.markdown("**Fallback - Raw String Representation:**")
                            st.code(str(result), language="python")
                            
                            # Try to show at least the keys
                            try:
                                st.markdown("**Available Keys:**")
                                for key in result.keys():
                                    value_type = type(result[key]).__name__
                                    value_preview = str(result[key])[:50] + "..." if len(str(result[key])) > 50 else str(result[key])
                                    st.markdown(f"- `{key}` ({value_type}): {value_preview}")
                            except:
                                pass
            
            # ========== BATCH UPLOAD MODE ==========
            else:  # Batch Upload
                st.markdown("""
                <p style='font-size: 11px; color: #92400e; margin: 0 0 0.8rem 0; font-style: italic;'>
                    Upload Excel/CSV file for multiple sources at once
                </p>
                """, unsafe_allow_html=True)
                
                # File upload
                uploaded_file = st.file_uploader(
                    "Upload source data file",
                    type=["csv", "xlsx", "xls"],
                    help="File must contain columns: source_id, task_success_rate, corroboration_score, report_timeliness",
                    label_visibility="collapsed"
                )
                
                # Template download
                col_template, col_limit = st.columns(2)
                with col_template:
                    template_csv = "source_id,task_success_rate,corroboration_score,report_timeliness,handler_confidence,deception_score,ci_flag\nSRC_001,0.85,0.75,0.90,0.80,0.20,0\nSRC_002,0.65,0.60,0.70,0.65,0.35,0\nSRC_003,0.90,0.85,0.95,0.88,0.15,1"
                    st.download_button(
                        label="📥 Download Template",
                        data=template_csv,
                        file_name="source_template.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_limit:
                    max_rows = st.number_input(
                        "Max rows",
                        min_value=1,
                        max_value=100,
                        value=50,
                        help="Limit number of sources to process (demo safety)"
                    )
                
                if uploaded_file is not None:
                    try:
                        # Read file
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Limit rows
                        if len(df) > max_rows:
                            st.warning(f"⚠️ File contains {len(df)} rows. Processing first {max_rows} rows only.")
                            df = df.head(max_rows)
                        
                        st.info(f"📋 Loaded {len(df)} sources from file")
                        
                        # Preview
                        with st.expander("👁️ Preview Data"):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Process button
                        if st.button("▶ Process Batch", type="primary", use_container_width=True):
                            recourse_rules = {
                                "rel_disengage": st.session_state.get("rel_disengage_slider", 0.35),
                                "rel_ci_flag": st.session_state.get("rel_ci_flag_slider", 0.50),
                                "dec_disengage": st.session_state.get("dec_disengage_slider", 0.75),
                                "dec_ci_flag": st.session_state.get("dec_ci_flag_slider", 0.60)
                            }
                            
                            with st.spinner(f"🔄 Processing {len(df)} sources through ML-TSSP pipeline..."):
                                results, errors = _process_batch_sources(df, recourse_rules)
                            
                            st.session_state.batch_results = results
                            st.session_state.batch_errors = errors
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"**File Error:** {str(e)}")
                
                # Display batch results
                if "batch_results" in st.session_state and st.session_state.batch_results:
                    results = st.session_state.batch_results
                    errors = st.session_state.batch_errors
                    
                    st.markdown("""
                    <div style='background: #ecfdf5; border-radius: 8px; padding: 0.8rem; margin-top: 1rem; border: 2px solid #10b981;'>
                        <p style='margin: 0; font-size: 12px; font-weight: 700; color: #047857;'>
                            📊 BATCH PROCESSING RESULTS
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processed", len(results))
                    with col2:
                        st.metric("Errors", len(errors))
                    with col3:
                        taskable = sum(1 for r in results if r['decision'] in ["task", "flag_and_task"])
                        st.metric("Taskable", taskable)
                    
                    # Summary table
                    summary_df = pd.DataFrame([{
                        "Source ID": r["source_id"],
                        "ML Reliability": f"{r['ml_reliability']:.3f}",
                        "Deception Risk": f"{r['deception_confidence']:.3f}",
                        "Decision": r["decision"],
                        "Tasks Assigned": len(r.get("tssp_allocation", []))
                    } for r in results])
                    
                    st.dataframe(summary_df, use_container_width=True, height=300)
                    
                    # Download results
                    output_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=output_csv,
                        file_name="batch_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show errors if any
                    if errors:
                        with st.expander(f"⚠️ Errors ({len(errors)})"):
                            for err in errors:
                                st.error(f"Row {err['row']} ({err['source_id']}): {', '.join(err['errors'])}")
        
        
        # ========== SIMULATION SCOPE CARD ==========
        with st.expander("🧮 SIMULATION SCOPE", expanded=True):
            num_sources = st.slider(
                "Number of sources", 
                1, 80,  # Full source capacity restored
                default_sources if preset_mode != "⚙️ Custom" else min(st.session_state.sources_count, 80),
                key="num_sources_slider",
                help="Total intelligence sources in optimization pool (80 sources with full behavioral distribution)"
            )
            st.markdown("<p style='font-size: 10px; color: #6b7280; margin: -0.5rem 0 0.8rem 0; font-style: italic;'>Total sources in the optimization pool</p>", unsafe_allow_html=True)
            
            st.session_state.sources_count = num_sources
            source_ids = [f"SRC_{k + 1:03d}" for k in range(num_sources)]
            jump_source_id = st.selectbox(
                "Jump to source",
                source_ids,
                index=None,
                key="jump_source",
                placeholder="Type or select a source",
                help="Quick navigation to specific source profile"
            )
        
        # ========== DECISION THRESHOLDS CARD ==========
        with st.expander("⚖️ DECISION THRESHOLDS", expanded=True):
            rel_cols = st.columns(2)
            with rel_cols[0]:
                rel_disengage = st.slider(
                    "Reliability disengage", 
                    0.0, 1.0, 
                    default_rel_disengage,
                    0.05,
                    key="rel_disengage_slider",
                    help="Below this score, source is removed from tasking"
                )
                st.markdown("<p style='font-size: 9px; color: #6b7280; margin: -0.5rem 0 0.5rem 0; line-height: 1.3;'>Below this score, source is automatically removed from tasking</p>", unsafe_allow_html=True)
                
            with rel_cols[1]:
                rel_ci_flag = st.slider(
                    "Reliability flag", 
                    0.0, 1.0, 
                    default_rel_flag,
                    0.05,
                    key="rel_ci_flag_slider",
                    help="Triggers enhanced monitoring and verification"
                )
                st.markdown("<p style='font-size: 9px; color: #6b7280; margin: -0.5rem 0 0.5rem 0; line-height: 1.3;'>Triggers enhanced monitoring and CI review</p>", unsafe_allow_html=True)
            
            dec_cols = st.columns(2)
            with dec_cols[0]:
                dec_disengage = st.slider(
                    "Deception reject", 
                    0.0, 1.0, 
                    default_dec_disengage,
                    0.05,
                    key="dec_disengage_slider",
                    help="High deception confidence triggers full rejection"
                )
                st.markdown("<p style='font-size: 9px; color: #6b7280; margin: -0.5rem 0 0.5rem 0; line-height: 1.3;'>High confidence deception triggers full source rejection</p>", unsafe_allow_html=True)
                
            with dec_cols[1]:
                dec_ci_flag = st.slider(
                    "Deception escalate", 
                    0.0, 1.0, 
                    default_dec_escalate,
                    0.05,
                    key="dec_ci_flag_slider",
                    help="Moderate deception escalates to CI investigation"
                )
                st.markdown("<p style='font-size: 9px; color: #6b7280; margin: -0.5rem 0 0.5rem 0; line-height: 1.3;'>Moderate deception risk escalates to CI investigation</p>", unsafe_allow_html=True)
            
            # Check for threshold conflicts
            if dec_ci_flag > dec_disengage:
                st.markdown("""
                <div style='background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px; 
                            padding: 0.5rem; margin: 0.5rem 0;'>
                    <p style='margin: 0; font-size: 10px; color: #991b1b;'>
                        ⚠️ <strong>Policy conflict:</strong> Escalate threshold exceeds reject threshold
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        st.session_state.recourse_rules = {
            "rel_disengage": float(rel_disengage),
            "rel_ci_flag": float(rel_ci_flag),
            "dec_disengage": float(dec_disengage),
            "dec_ci_flag": float(dec_ci_flag),
        }
        
        # ========== SCENARIO SUMMARY (LIVE FEEDBACK ENGINE) ==========
        # Calculate risk posture
        avg_rel_threshold = (rel_disengage + rel_ci_flag) / 2
        avg_dec_threshold = (dec_disengage + dec_ci_flag) / 2
        
        if avg_rel_threshold > 0.5 and avg_dec_threshold < 0.65:
            risk_posture = "🟢 Conservative"
            posture_color = "#10b981"
            policy_mode = "High Assurance Intelligence"
        elif avg_rel_threshold > 0.35 and avg_dec_threshold < 0.75:
            risk_posture = "🟡 Balanced"
            posture_color = "#f59e0b"
            policy_mode = "Standard Operations"
        else:
            risk_posture = "🔴 Aggressive"
            posture_color = "#ef4444"
            policy_mode = "High Risk Collection"
        
        # Calculate mission confidence (based on thresholds strictness)
        strictness_score = (avg_rel_threshold * 0.6 + (1 - avg_dec_threshold) * 0.4)
        mission_confidence = min(0.95, 0.65 + strictness_score * 0.35)
        
        confidence_color = "#10b981" if mission_confidence > 0.8 else "#f59e0b" if mission_confidence > 0.7 else "#ef4444"
        
        # Calculate additional metrics
        expected_high_risk = int(num_sources * (1 - strictness_score) * 0.3)
        expected_moderate_risk = int(num_sources * 0.4)
        expected_low_risk = num_sources - expected_high_risk - expected_moderate_risk
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
                    border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem; 
                    border: 2px solid {posture_color}; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
            <p style='margin: 0 0 0.8rem 0; font-size: 14px; font-weight: 700; 
                      color: #047857; text-transform: uppercase; letter-spacing: 0.5px; text-align: center;'>
                📋 SCENARIO SUMMARY
            </p>""", unsafe_allow_html=True)
        
        # Use native Streamlit components for better reliability
        st.markdown(f"""
            <div style='background: white; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;'>
                <p style='margin: 0; font-size: 11px; color: #6b7280; font-weight: 600;'>Risk Posture</p>
                <p style='margin: 0.2rem 0 0 0; font-size: 16px; font-weight: 700; color: {posture_color};'>
                    {risk_posture}
                </p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: white; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;'>
                <p style='margin: 0; font-size: 11px; color: #6b7280; font-weight: 600;'>Mission Confidence</p>
                <p style='margin: 0.2rem 0 0 0; font-size: 18px; font-weight: 700; color: {confidence_color};'>
                    {mission_confidence:.2f}
                </p>""", unsafe_allow_html=True)
        st.progress(mission_confidence)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: white; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem;'>
                <p style='margin: 0; font-size: 11px; color: #6b7280; font-weight: 600;'>Policy Mode</p>
                <p style='margin: 0.2rem 0 0 0; font-size: 12px; font-weight: 600; color: #1e40af;'>
                    {policy_mode}
                </p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='border-top: 1px solid #d1d5db; padding-top: 0.8rem; margin-bottom: 0.8rem;'>
                <p style='margin: 0 0 0.3rem 0; font-size: 10px; color: #6b7280;'>
                    <strong>Sources:</strong> <span style='color: #1e40af; font-weight: 700;'>{num_sources}</span>
                </p>
                <p style='margin: 0 0 0.3rem 0; font-size: 10px; color: #6b7280;'>
                    <strong>Review load:</strong> <span style='color: #1e40af; font-weight: 700;'>~{int(num_sources * 0.3)}</span>
                </p>
                <p style='margin: 0 0 0.5rem 0; font-size: 10px; color: #6b7280;'>
                    <strong>Est. runtime:</strong> <span style='color: #10b981; font-weight: 700;'>&lt; 2s</span>
                </p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: rgba(59, 130, 246, 0.05); border-radius: 6px; padding: 0.6rem; border: 1px solid #dbeafe;'>
                <p style='margin: 0 0 0.4rem 0; font-size: 10px; color: #1e40af; font-weight: 700;'>
                    Expected Risk Distribution
                </p>
                <p style='margin: 0; font-size: 9px; color: #6b7280;'>
                    🟢 Low: ~{expected_low_risk} | 🟡 Med: ~{expected_moderate_risk} | 🔴 High: ~{expected_high_risk}
                </p>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Close control panel
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ======================================================
    # 1. DECISION OPTIMIZATION ENGINE
    # ======================================================
    sources = []
    
    # First, collect source data from profiles with all 6 features
    source_ids = [f"SRC_{k + 1:03d}" for k in range(num_sources)]
    for i in range(num_sources):
        rng = np.random.default_rng(i + 1)
        tsr_default = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
        cor_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
        time_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
        handler_default = float(np.clip(rng.beta(4, 3), 0.0, 1.0))
        dec_default = float(np.clip(rng.beta(2, 5), 0.0, 0.8))
        ci_default = int(rng.choice([0, 1], p=[0.88, 0.12]))
        
        features = {
            "task_success_rate": float(tsr_default),
            "corroboration_score": float(cor_default),
            "report_timeliness": float(time_default),
            "handler_confidence": float(handler_default),
            "deception_score": float(dec_default),
            "ci_flag": int(ci_default)
        }
        
        sources.append({
            "source_id": f"SRC_{i + 1:03d}",
            "features": features,
            "reliability_series": [],
            "recourse_rules": {}
        })
    
    # Show Decision Optimization Engine only on Source Profiles tab
    if nav_key == "profiles":
        with st.expander("🧠 Decision Optimization Engine", expanded=True):
            st.markdown("""
            <div style="background:linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);border-radius:15px;padding:1.8rem;
                        box-shadow:0 4px 15px rgba(0,0,0,0.12);border:1px solid #cbd5e1;
                        border-top:4px solid #10b981;">
                <h3 class="section-header" style="margin-top:0;color:#047857;">🧠 Decision Optimization Engine</h3>
                <p style="text-align:center;color:#475569;font-size:13px;margin:0 0 1.2rem 0;">
                    Configure parameters and execute the ML–TSSP optimization algorithm
                </p>
            """, unsafe_allow_html=True)
        
        # ========== OPTIMIZATION CONTROL PANEL ==========
        st.markdown('<h4 style="color: #1e3a8a; margin-bottom: 1rem;">🧪 Optimization Control Panel</h4>', unsafe_allow_html=True)
        
        col_run, col_reset = st.columns([2, 1])
        with col_run:
            run_button_right = st.button("▶ Execute Optimization", type="primary", use_container_width=True, key="run_opt_btn_right", help="Execute ML–TSSP with current configuration")
        with col_reset:
            reset_button_right = st.button("↺ Reset Configuration", use_container_width=True, key="reset_btn_right", help="Clear configuration and results")
        
        if reset_button_right:
            st.session_state.results = None
            st.rerun()
        
        st.divider()
        
        # ========== EXECUTION FEEDBACK & STATUS CONSOLE ==========
        if st.session_state.results is None:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #f1fdf8 100%); padding: 1.5rem; border-radius: 12px; border: 2px dashed #bfdbfe; text-align: center;">
                <p style="margin: 0; font-size: 14px; color: #1e3a8a; font-weight: 600;">⏳ Ready for Optimization</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 12px; color: #6b7280;">Click <strong>Execute Optimization</strong> to run the ML–TSSP algorithm</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #a7f3d0;">
                <p style="margin: 0; font-size: 14px; color: #15803d; font-weight: 600;">✅ Optimization Complete</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 12px; color: #1f2937;">Results ready for analysis. Review decision summary below.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ========== EXECUTIVE DECISION SUMMARY ==========
        st.markdown('<h4 style="color: #1e3a8a; margin-bottom: 1rem;">📊 Executive Decision Summary</h4>', unsafe_allow_html=True)
        
        if st.session_state.results is None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sources Configured", len(sources))
            with col2:
                st.metric("Expected Risk", "—")
            with col3:
                st.metric("Improvement vs Uniform", "—")
            
            # Explanation for pre-optimization state
            st.markdown(f"""
            <div style='background: #f8fafc; border-left: 3px solid #94a3b8; padding: 0.8rem 1rem; border-radius: 6px; margin-top: 1rem;'>
                <p style='margin: 0; font-size: 12px; color: #475569; line-height: 1.6;'>
                    <strong>Status:</strong> {len(sources)} sources configured and awaiting optimization. 
                    Execute the ML-TSSP algorithm to determine optimal task assignments based on reliability thresholds 
                    (disengage: {st.session_state.recourse_rules['rel_disengage']:.2f}, flag: {st.session_state.recourse_rules['rel_ci_flag']:.2f}) 
                    and deception constraints (reject: {st.session_state.recourse_rules['dec_disengage']:.2f}, escalate: {st.session_state.recourse_rules['dec_ci_flag']:.2f}).
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            results = st.session_state.results
            ml_emv = results.get("emv", {}).get("ml_tssp", 0)
            uni_emv = results.get("emv", {}).get("uniform", 0)
            det_emv = results.get("emv", {}).get("deterministic", 0)
            risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0.0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_indicator("Total Sources", len(sources), note="All assigned", key="kpi_total_sources_exec")
            with col2:
                render_kpi_indicator("Risk (EMV)", ml_emv, reference=uni_emv, note="vs Uniform", key="kpi_risk_exec")
            with col3:
                low_risk = sum(1 for a in results.get("policies", {}).get("ml_tssp", []) if a.get("expected_risk", 0) < 0.3)
                render_kpi_indicator("Low Risk Sources", low_risk, suffix=f" / {len(sources)}", key="kpi_low_risk_exec")
            with col4:
                render_kpi_indicator("Improvement", risk_reduction, suffix="%", note="Vs baseline", key="kpi_improvement_exec")
            
            # Dynamic explanation based on optimization results
            ml_policy = results.get("policies", {}).get("ml_tssp", [])
            medium_risk = sum(1 for a in ml_policy if 0.3 <= a.get("expected_risk", 0) < 0.6)
            high_risk = sum(1 for a in ml_policy if a.get("expected_risk", 0) >= 0.6)
            
            # Determine outcome characterization
            if risk_reduction > 10:
                outcome_desc = "substantive reduction in expected mission value loss"
            elif risk_reduction > 5:
                outcome_desc = "measurable reduction in expected mission value loss"
            elif risk_reduction > 0:
                outcome_desc = "marginal reduction in expected mission value loss"
            else:
                outcome_desc = "no reduction in expected mission value loss relative to the uniform baseline"
            
            # Risk distribution assessment
            risk_distribution = f"{low_risk} low-risk ({low_risk/len(sources)*100:.1f}%), {medium_risk} medium-risk ({medium_risk/len(sources)*100:.1f}%), and {high_risk} high-risk ({high_risk/len(sources)*100:.1f}%)"
            
            st.markdown(f"""
            <div style='background: #f8fafc; border-left: 3px solid #3b82f6; padding: 0.8rem 1rem; border-radius: 6px; margin-top: 1rem;'>
                <p style='margin: 0 0 0.6rem 0; font-size: 12px; color: #475569; line-height: 1.6;'>
                    <strong>Outcome:</strong> The ML-TSSP optimization assigns all {len(sources)} sources to tasks with an expected mission value (EMV) loss of {ml_emv:.2f}, 
                    representing a {outcome_desc} compared to the uniform assignment baseline (EMV: {uni_emv:.2f}) and deterministic policy (EMV: {det_emv:.2f}). 
                    The source portfolio distributes as {risk_distribution} under current reliability (disengage ≤{st.session_state.recourse_rules['rel_disengage']:.2f}, 
                    flag ≥{st.session_state.recourse_rules['rel_ci_flag']:.2f}) and deception (reject ≥{st.session_state.recourse_rules['dec_disengage']:.2f}, 
                    escalate ≥{st.session_state.recourse_rules['dec_ci_flag']:.2f}) constraints.
                </p>
                <p style='margin: 0; font-size: 12px; color: #64748b; line-height: 1.6;'>
                    <strong>Constraint Impact:</strong> The configured thresholds filtered sources based on behavioral reliability and deception indicators, 
                    with {high_risk} sources ({high_risk/len(sources)*100:.1f}%) exceeding acceptable risk levels despite optimization. 
                    Adjusting constraint parameters or reviewing high-risk source assignments may yield further risk mitigation within operational requirements.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== RUN OPTIMIZATION EXECUTION ==========
        if run_button_right:
            payload = {
                "sources": sources,
                "seed": 42
            }

            try:
                with st.spinner("🔄 Running optimization…"):
                    result = run_optimization(payload)
                    if isinstance(result, dict) and isinstance(result.get("policies"), dict):
                        # Create sources_map for dynamic score calculation
                        sources_map = {s.get("source_id"): s for s in sources}
                        
                        for pkey in ["ml_tssp", "deterministic", "uniform"]:
                            plist = result["policies"].get(pkey) or []
                            fixed = enforce_assignment_constraints(plist, sources_map)
                            result["policies"][pkey] = fixed
                            result.setdefault("emv", {})[pkey] = compute_emv(fixed)
                    st.session_state.results = result
                    st.session_state.sources = sources
                    st.success("✅ Optimization complete! Review decision summary above.")
                    st.session_state.show_results_popup = True
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")

    # ======================================================
    # 2. DECISION INTELLIGENCE SUITE
    # ======================================================
    results = st.session_state.results

    # Show Policy Insights tab content
    if nav_key == "policies" and results is not None:
        st.markdown('<div class="section-frame">', unsafe_allow_html=True)
        st.markdown("""<h3 class="section-header">📈 Policy Insights - Comparative Policy Evaluation</h3>
        <p style="text-align:center;color:#6b7280;font-size:13px;margin:0 0 1rem 0;">
            Comprehensive analysis of ML–TSSP optimization results with policy comparisons and sensitivity assessments.
        </p>""", unsafe_allow_html=True)
        ml_policy = results.get("policies", {}).get("ml_tssp", [])
        det_policy = results.get("policies", {}).get("deterministic", [])
        uni_policy = results.get("policies", {}).get("uniform", [])
        ml_emv = results.get("emv", {}).get("ml_tssp", 0)
        det_emv = results.get("emv", {}).get("deterministic", 0)
        uni_emv = results.get("emv", {}).get("uniform", 0)
        risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0.0
        
        # ========== COMPARATIVE POLICY EVALUATION ==========
        with st.expander("🧭 Comparative Policy Evaluation", expanded=True):
            _render_comparative_policy_section(results, ml_policy, det_policy, uni_policy, ml_emv, det_emv, uni_emv, risk_reduction)
        with st.expander("🧠 SHAP Explanations", expanded=False):
            _render_shap_section(num_sources)
        with st.expander("📡 Source Drift Monitoring (Reliability & Deception)", expanded=False):
            _render_drift_section()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show EVPI Focus tab content
    elif nav_key == "evpi" and results is not None:
        st.markdown('<div class="section-frame">', unsafe_allow_html=True)
        st.markdown("""<h3 class="section-header">💰 EVPI Focus - Expected Value of Perfect Information</h3>
        <p style="text-align:center;color:#6b7280;font-size:13px;margin:0 0 1rem 0;">
            Quantify the marginal value of eliminating source behavior uncertainty through perfect information acquisition.
        </p>
        <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                    padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; 
                    margin-bottom: 1.5rem;'>
            <p style='margin: 0; font-size: 12px; color: #1e3a8a; line-height: 1.6;'>
                <strong>Decision Context:</strong> EVPI measures the maximum justifiable cost of obtaining perfect foreknowledge 
                about each source's true reliability and deception risk. High EVPI indicates that current uncertainty 
                materially degrades decision quality—these sources warrant enhanced vetting, corroboration, or collection investment. 
                Low EVPI suggests the ML model has already extracted actionable signal from available data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        ml_policy = results.get("policies", {}).get("ml_tssp", [])
        uni_policy = results.get("policies", {}).get("uniform", [])
        
        with st.expander("💰 Expected Value of Perfect Information (EVPI)", expanded=True):
            _render_evpi_section(ml_policy, uni_policy)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show Stress Lab tab content
    elif nav_key == "stress" and results is not None:
        st.markdown('<div class="section-frame">', unsafe_allow_html=True)
        st.markdown("""<h3 class="section-header">🔬 Stress Lab - Behavioral Uncertainty & What-If Analysis</h3>
        <p style="text-align:center;color:#6b7280;font-size:13px;margin:0 0 1rem 0;">
            Test system resilience under various behavioral scenarios and stress conditions.
        </p>""", unsafe_allow_html=True)
        ml_policy = results.get("policies", {}).get("ml_tssp", [])
        ml_emv = results.get("emv", {}).get("ml_tssp", 0)
        det_emv = results.get("emv", {}).get("deterministic", 0)
        uni_emv = results.get("emv", {}).get("uniform", 0)
        risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0.0
        
        with st.expander("🔬 Behavioral Uncertainty & Stress Analysis (What-If)", expanded=True):
            _render_stress_section(ml_policy, ml_emv, det_emv, uni_emv, risk_reduction)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ======================================================
    # 3. SOURCE PROFILES AND TASKING
    # ======================================================
    # Show Source Profiles only on Source Profiles tab
    if nav_key == "profiles":
        with st.expander("📋 Source Profiles & Tasking", expanded=False):
            st.markdown("""
            <div style="background:linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);border-radius:15px;padding:1.8rem;
                        box-shadow:0 4px 15px rgba(0,0,0,0.12);border:1px solid #cbd5e1;
                        border-top:4px solid #3b82f6;">
                <h3 class="section-header" style="margin-top:0;color:#1e40af;">📋 Source Profiles & Detailed Analysis</h3>
                <p style="text-align:center;color:#475569;font-size:13px;margin:0 0 1.2rem 0;">
                    Select a source to view and configure its detailed intelligence profile
                </p>
            """, unsafe_allow_html=True)
            
            # Initialize selected source in session state
            if "selected_source_idx" not in st.session_state:
                st.session_state.selected_source_idx = 0
        
        source_selector_col, source_profile_col = st.columns([1.2, 2.8])
        
        # ========== LEFT PANEL: SOURCE SELECTOR CONSOLE ==========
        with source_selector_col:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 1rem; border-radius: 10px; border: 1px solid #cbd5e1; margin-bottom: 1rem; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <p style="margin: 0 0 0.8rem 0; font-weight: 700; color: #1e40af; text-transform: uppercase; letter-spacing: 0.5px;">📑 Source Selection</p>
            """, unsafe_allow_html=True)
            
            # Source selection as clickable buttons
            sources_list = []
            for i in range(num_sources):
                src_id = f"SRC_{i + 1:03d}"
                
                # Calculate risk level based on source data
                rng = np.random.default_rng(i + 1)
                tsr_val = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
                cor_val = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
                risk_score = 1.0 - (tsr_val * 0.6 + cor_val * 0.4)
                
                if risk_score < 0.3:
                    risk_level = "Low"
                    risk_color = "#10b981"
                    status_icon = "🟢"
                elif risk_score < 0.6:
                    risk_level = "Medium"
                    risk_color = "#f59e0b"
                    status_icon = "🟡"
                else:
                    risk_level = "High"
                    risk_color = "#ef4444"
                    status_icon = "🔴"
                
                try:
                    if st.session_state.get("results"):
                        ml_policy = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
                        match = next((a for a in ml_policy if a.get("source_id") == src_id), None)
                        if match:
                            task_assign = str(match.get("task", "—"))
                        else:
                            task_assign = "—"
                    else:
                        task_assign = "—"
                except Exception:
                    task_assign = "—"
                
                sources_list.append({
                    "id": src_id,
                    "index": i,
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "status_icon": status_icon,
                    "task": task_assign,
                    "tsr": tsr_val,
                    "cor": cor_val
                })
            
            # Display clickable source cards
            for src in sources_list[:min(num_sources, 10)]:
                is_selected = st.session_state.selected_source_idx == src["index"]
                
                # Apply different styling for active (selected) vs inactive buttons
                if is_selected:
                    # Active button styling - primary type with visual emphasis
                    button_type = "primary"
                    border_style = f"border: 3px solid #3b82f6;"
                    bg_color = "#dbeafe"
                    card_shadow = "box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);"
                else:
                    # Inactive button styling - secondary type, subtle appearance
                    button_type = "secondary"
                    border_style = "border: 1px solid #cbd5e1;"
                    bg_color = "#f8fafc"
                    card_shadow = "box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);"
                
                col_btn, col_spacer = st.columns([1, 0.05])
                with col_btn:
                    if st.button(
                        f"{src['status_icon']} {src['id']}",
                        key=f"select_src_{src['index']}",
                        type=button_type,
                        use_container_width=True,
                        help=f"Click to view {src['id']} details"
                    ):
                        st.session_state.selected_source_idx = src["index"]
                        st.rerun()
                
                # Display source info card with conditional styling
                st.markdown(f"""
                <div style="background: {bg_color}; border-left: 4px solid {src['risk_color']}; {border_style} {card_shadow} padding: 0.6rem; border-radius: 6px; margin-bottom: 0.6rem; margin-top: -0.5rem; transition: all 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <p style="margin: 0; font-size: 10px; color: {'#1e3a8a' if is_selected else '#475569'}; font-weight: {'700' if is_selected else '500'};">{src['risk_level']} Risk</p>
                        </div>
                        <div style="text-align: right;">
                            <p style="margin: 0; font-size: 10px; font-weight: {'700' if is_selected else '600'}; color: {'#1e40af' if is_selected else '#3b82f6'};">Task: {src['task']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if num_sources > 10:
                st.markdown(f"""
                <div style="background: #dbeafe; padding: 0.6rem; border-radius: 6px; text-align: center; font-size: 10px; color: #1e40af; font-weight: 600; border: 1px solid #93c5fd;">
                    +{num_sources - 10} more sources (scroll or use jump selector)
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== RIGHT PANEL: DETAILED SOURCE PROFILE ==========
        with source_profile_col:
            # Get selected source data
            selected_idx = st.session_state.selected_source_idx
            selected_src_id = f"SRC_{selected_idx + 1:03d}"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 1.2rem; border-radius: 10px; border: 2px solid #60a5fa; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #ffffff;">🔹 {selected_src_id}</h4>
                        <p style="margin: 0.3rem 0 0 0; font-size: 11px; color: #dbeafe;">Source Intelligence Profile & Configuration</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick actions
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                st.button("📋 Copy Data", key=f"copy_src_detail_{selected_idx}", help="Copy source data", use_container_width=True)
            with action_col2:
                st.button("📊 Export Report", key=f"export_src_detail_{selected_idx}", help="Export source report", use_container_width=True)
            with action_col3:
                st.button("🔄 Reset Values", key=f"reset_src_{selected_idx}", help="Reset to defaults", use_container_width=True)
            
            st.divider()
            
            # ========== SOURCE ATTRIBUTE CONTROLS ==========
            rng = np.random.default_rng(selected_idx + 1)
            tsr_default = float(np.clip(rng.beta(5, 3), 0.0, 1.0))
            cor_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))
            time_default = float(np.clip(rng.beta(4, 4), 0.0, 1.0))

            gauge_cols = st.columns(3)
            with gauge_cols[0]:
                st.markdown("**Competence Level**")
                
                fig_comp_mini = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=tsr_default * 100,
                    title={'text': "Task Success Rate %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                    number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#bfdbfe', 'tickfont': {'size': 9}},
                        'bar': {'color': COLORS['baseline'], 'thickness': 0.15},
                        'bgcolor': '#f0f9ff',
                        'borderwidth': 1.5,
                        'bordercolor': '#bfdbfe',
                        'steps': [
                            {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                            {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                            {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                        ],
                        'threshold': {
                            'line': {'color': '#ef4444', 'width': 2},
                            'thickness': 0.7,
                            'value': 50
                        }
                    }
                ))
                fig_comp_mini.update_layout(
                    height=200, 
                    margin=dict(l=5, r=5, t=35, b=5), 
                    paper_bgcolor='white', 
                    font=dict(size=10),
                    hovermode=False,
                    clickmode='event+select'
                )
                st.plotly_chart(fig_comp_mini, use_container_width=True, key=f'gauge_comp_{selected_idx}')
                
                tsr = st.number_input("Adjust Task Success Rate", 0.0, 1.0, tsr_default, step=0.05, key=f"tsr_input_{selected_idx}")
            
            with gauge_cols[1]:
                st.markdown("**Reporting Consistency**")
                
                fig_cons_mini = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cor_default * 100,
                    title={'text': "Corroboration Score %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                    number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#d1fae5', 'tickfont': {'size': 9}},
                        'bar': {'color': COLORS['cooperative'], 'thickness': 0.15},
                        'bgcolor': '#f0fdf4',
                        'borderwidth': 1.5,
                        'bordercolor': '#d1fae5',
                        'steps': [
                            {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                            {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                            {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                        ],
                        'threshold': {
                            'line': {'color': '#ef4444', 'width': 2},
                            'thickness': 0.7,
                            'value': 50
                        }
                    }
                ))
                fig_cons_mini.update_layout(
                    height=200, 
                    margin=dict(l=5, r=5, t=35, b=5), 
                    paper_bgcolor='white', 
                    font=dict(size=10),
                    hovermode=False,
                    clickmode='event+select'
                )
                st.plotly_chart(fig_cons_mini, use_container_width=True, key=f'gauge_cons_{selected_idx}')
                
                cor = st.number_input("Adjust Corroboration Level", 0.0, 1.0, cor_default, step=0.05, key=f"cor_input_{selected_idx}")
            
            with gauge_cols[2]:
                st.markdown("**Report Timeliness**")
                
                fig_time_mini = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=time_default * 100,
                    title={'text': "Report Speed %", 'font': {'size': 12, 'color': '#1e3a8a'}},
                    number={'suffix': "%", 'font': {'size': 16, 'color': '#1e3a8a'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': '#fde68a', 'tickfont': {'size': 9}},
                        'bar': {'color': COLORS['uncertain'], 'thickness': 0.15},
                        'bgcolor': '#fffbeb',
                        'borderwidth': 1.5,
                        'bordercolor': '#fde68a',
                        'steps': [
                            {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.12)'},
                            {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.12)'},
                            {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.12)'}
                        ],
                        'threshold': {
                            'line': {'color': '#ef4444', 'width': 2},
                            'thickness': 0.7,
                            'value': 50
                        }
                    }
                ))
                fig_time_mini.update_layout(
                    height=200, 
                    margin=dict(l=5, r=5, t=35, b=5), 
                    paper_bgcolor='white', 
                    font=dict(size=10),
                    hovermode=False,
                    clickmode='event+select'
                )
                st.plotly_chart(fig_time_mini, use_container_width=True, key=f'gauge_time_{selected_idx}')
                
                time = st.number_input("Adjust Report Speed", 0.0, 1.0, time_default, step=0.05, key=f"time_input_{selected_idx}")
            
            st.markdown("**60-Day Reliability Forecast**")
            st.caption("Expanded horizon to observe medium-term reliability trajectory (60 periods).")
            
            periods = 60
            rng_forecast = np.random.default_rng(10_000 + selected_idx)
            base_rel = np.clip(0.35 + 0.25 * tsr + 0.20 * cor + 0.15 * time, 0.2, 0.9)
            drift = 0.012 + 0.006 * rng_forecast.normal()
            reliability_ts = [np.clip(base_rel + drift * j + rng_forecast.normal(0, 0.02), 0.25, 0.98) for j in range(periods)]
            
            window = 7
            rel_ma = []
            for j in range(len(reliability_ts)):
                start_idx = max(0, j - window + 1)
                window_vals = reliability_ts[start_idx:j + 1]
                rel_ma.append(np.mean(window_vals))
            
            rel_df = pd.DataFrame({
                'period': range(periods),
                'reliability': reliability_ts,
                'ma': rel_ma,
                'upper': [min(r + 0.1, 1.0) for r in reliability_ts],
                'lower': [max(r - 0.1, 0.0) for r in reliability_ts]
            })
            
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['reliability'], mode='lines+markers', name='Predicted', line=dict(color=COLORS['baseline'], width=2.5), marker=dict(size=7), hovertemplate='<b>Period %{x}</b><br>Reliability: %{y:.2f}<extra></extra>'))
            fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['ma'], mode='lines', name='Moving Avg (7)', line=dict(color=COLORS['cooperative'], width=2.5, dash='dash'), hovertemplate='<b>Period %{x}</b><br>MA: %{y:.2f}<extra></extra>'))
            fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_rel.add_trace(go.Scatter(x=rel_df['period'], y=rel_df['lower'], mode='lines', line=dict(width=0), fillcolor='rgba(59, 130, 246, 0.2)', fill='tonexty', showlegend=False, hoverinfo='skip', name='Confidence'))
            fig_rel.add_hline(y=0.5, line_dash='dash', line_color=COLORS['deceptive'], opacity=0.6, annotation_text="Risk Threshold")
            fig_rel.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#f0f9ff', plot_bgcolor='#f8fafc', xaxis_title='Period', yaxis_title='Reliability', showlegend=True, font=dict(size=10), hovermode='x unified', dragmode='zoom')
            st.plotly_chart(fig_rel, use_container_width=True, key=f'rel_chart_{selected_idx}')
            
            st.divider()
            
            avg_rel = np.mean(reliability_ts)
            deception_risk = 1.0 - cor
            
            # ========== SUMMARY METRICS ==========
            st.markdown('<h4 style="color: #1e3a8a; margin: 0.5rem 0 1rem 0;">📊 Assessment Summary</h4>', unsafe_allow_html=True)
            
            met_col1, met_col2, met_col3 = st.columns(3)
            
            with met_col1:
                st.metric(
                    "🎯 Reliability",
                    f"{avg_rel:.2f}",
                    delta=f"{(avg_rel - 0.5) * 100:+.0f}%" if avg_rel >= 0.5 else f"{(avg_rel - 0.5) * 100:.0f}%",
                    delta_color="normal"
                )
            
            with met_col2:
                st.metric(
                    "⚠️ Risk Level",
                    "High" if deception_risk > 0.6 else "Med" if deception_risk > 0.3 else "Low",
                    delta=f"{deception_risk:.2f}",
                    delta_color="inverse"
                )
            
            with met_col3:
                assigned_task_display = "—"
                try:
                    if st.session_state.get("results"):
                        ml_assignments = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
                        my_id = f"SRC_{selected_idx + 1:03d}"
                        match = next((a for a in ml_assignments if a.get("source_id") == my_id), None)
                        if match:
                            assigned_task_display = str(match.get("task") or "—")
                except Exception:
                    pass
                st.metric(
                    "📋 Assignment",
                    assigned_task_display,
                    help="ML-TSSP optimized task"
                )
            
            st.divider()
            
            # ========== DYNAMIC RECOMMENDATION PANEL ==========
            st.markdown('<h4 style="color: #1e3a8a; margin: 0.5rem 0 1rem 0;">💡 AI-Powered Recommendations</h4>', unsafe_allow_html=True)
            
            # Calculate comprehensive metrics for recommendation
            avg_rel = np.mean(reliability_ts)
            deception_risk = 1.0 - cor
            trend_direction = reliability_ts[-1] - reliability_ts[0]
            volatility = np.std(reliability_ts)
            recent_performance = np.mean(reliability_ts[-10:])
            
            # Get ML-TSSP assignment context
            ml_assignment = None
            expected_risk = None
            try:
                if st.session_state.get("results"):
                    ml_assignments = st.session_state["results"].get("policies", {}).get("ml_tssp", [])
                    my_id = f"SRC_{selected_idx + 1:03d}"
                    ml_assignment = next((a for a in ml_assignments if a.get("source_id") == my_id), None)
                    if ml_assignment:
                        expected_risk = ml_assignment.get("expected_risk", 0.5)
            except Exception:
                pass
            
            # Determine recommendation level based on multiple factors
            score = 0
            factors = []
            
            # Factor 1: Average reliability
            if avg_rel > 0.7:
                score += 3
                factors.append(("High reliability trajectory", "🟢"))
            elif avg_rel > 0.5:
                score += 2
                factors.append(("Moderate reliability", "🟡"))
            else:
                score += 0
                factors.append(("Low reliability concern", "🔴"))
            
            # Factor 2: Trend direction
            if trend_direction > 0.1:
                score += 2
                factors.append(("Improving performance trend", "🟢"))
            elif trend_direction > -0.05:
                score += 1
                factors.append(("Stable performance", "🟡"))
            else:
                score += 0
                factors.append(("Declining performance", "🔴"))
            
            # Factor 3: Deception risk
            if deception_risk < 0.3:
                score += 2
                factors.append(("Low deception indicators", "🟢"))
            elif deception_risk < 0.5:
                score += 1
                factors.append(("Moderate deception risk", "🟡"))
            else:
                score += 0
                factors.append(("High deception risk", "🔴"))
            
            # Factor 4: Volatility
            if volatility < 0.05:
                score += 1
                factors.append(("Consistent behavior", "🟢"))
            elif volatility < 0.1:
                score += 0
                factors.append(("Variable behavior", "🟡"))
            else:
                score += 0
                factors.append(("Unstable patterns", "🔴"))
            
            # Factor 5: Task success rate
            if tsr > 0.7:
                score += 2
                factors.append(("Strong task completion", "🟢"))
            elif tsr > 0.5:
                score += 1
                factors.append(("Adequate performance", "🟡"))
            else:
                score += 0
                factors.append(("Poor task success", "🔴"))
            
            # Factor 6: ML-TSSP risk assessment
            if expected_risk is not None:
                if expected_risk < 0.3:
                    score += 2
                    factors.append((f"Low ML-predicted risk ({expected_risk:.2f})", "🟢"))
                elif expected_risk < 0.6:
                    score += 1
                    factors.append((f"Medium ML-predicted risk ({expected_risk:.2f})", "🟡"))
                else:
                    score += 0
                    factors.append((f"High ML-predicted risk ({expected_risk:.2f})", "🔴"))
            
            # Determine recommendation tier (max score = 12)
            if score >= 9:
                rec_tier = "HIGHLY_RECOMMENDED"
                rec_label = "✅ HIGHLY RECOMMENDED"
                rec_color_primary = "#10b981"
                rec_bg_gradient = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
                rec_text_color = "#065f46"
                rec_action = "Prioritize for critical intelligence operations. Suitable for high-stakes missions."
            elif score >= 6:
                rec_tier = "RECOMMENDED"
                rec_label = "✓ RECOMMENDED"
                rec_color_primary = "#3b82f6"
                rec_bg_gradient = "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)"
                rec_text_color = "#1e40af"
                rec_action = "Reliable for standard operations. Monitor for any performance degradation."
            elif score >= 4:
                rec_tier = "CONDITIONAL"
                rec_label = "⚠️ CONDITIONAL APPROVAL"
                rec_color_primary = "#f59e0b"
                rec_bg_gradient = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                rec_text_color = "#92400e"
                rec_action = "Enhanced monitoring required. Recommend pairing with corroborative sources."
            else:
                rec_tier = "NOT_RECOMMENDED"
                rec_label = "❌ NOT RECOMMENDED"
                rec_color_primary = "#ef4444"
                rec_bg_gradient = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                rec_text_color = "#991b1b"
                rec_action = "Immediate review required. Consider suspension pending counterintelligence assessment."
            
            # Interactive recommendation card
            st.markdown(f"""
            <div style="background: {rec_bg_gradient}; 
                        padding: 1rem; 
                        border-radius: 10px; 
                        border-left: 5px solid {rec_color_primary};
                        box-shadow: 0 3px 12px rgba(0,0,0,0.12);
                        margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <p style="margin: 0; font-size: 13px; font-weight: 700; color: {rec_text_color}; text-transform: uppercase; letter-spacing: 0.5px;">
                        {rec_label}
                    </p>
                    <div style="background: {rec_color_primary}; color: white; padding: 0.3rem 0.8rem; border-radius: 6px; font-size: 12px; font-weight: 600;">
                        Score: {score}/12
                    </div>
                </div>
                <p style="margin: 0; font-size: 13px; color: {rec_text_color}; line-height: 1.5;">
                    <strong>Recommendation:</strong> {rec_action}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Expandable factor breakdown
            with st.expander("📊 View Assessment Factors", expanded=False):
                st.markdown("**Contributing Factors to Recommendation:**")
                for idx, (factor, indicator) in enumerate(factors, 1):
                    st.markdown(f"{idx}. {indicator} {factor}")
                
                st.divider()
                st.markdown(f"""
                **Scoring Methodology:**
                - Reliability Score: {avg_rel:.2f} (Target: >0.6)
                - Performance Trend: {'+' if trend_direction > 0 else ''}{trend_direction:.3f}
                - Deception Risk: {deception_risk:.2f} (Target: <0.4)
                - Behavior Volatility: {volatility:.3f} (Target: <0.08)
                - Recent 10-Period Avg: {recent_performance:.2f}
                """)
                
                if ml_assignment:
                    opt_score = ml_assignment.get('score', 0)
                    # Determine score color and label
                    if opt_score >= 70:
                        score_color = "#10b981"  # Green
                        score_label = "Excellent"
                    elif opt_score >= 50:
                        score_color = "#3b82f6"  # Blue
                        score_label = "Good"
                    elif opt_score >= 30:
                        score_color = "#f59e0b"  # Orange
                        score_label = "Fair"
                    else:
                        score_color = "#ef4444"  # Red
                        score_label = "Poor"
                    
                    st.markdown(f"""
                    **ML-TSSP Assignment Context:**
                    - Assigned Task: {ml_assignment.get('task', 'N/A')}
                    - Expected Risk: {expected_risk:.3f}
                    - <span style='color: {score_color}; font-weight: 700;'>Optimization Score: {opt_score}/100 ({score_label})</span>
                    
                    *Score combines: Risk (40%), Task Success (25%), Corroboration (20%), Timeliness (15%)*
                    """, unsafe_allow_html=True)
            
            # Update sources list with current selected source data
            features = {
                "task_success_rate": float(tsr),
                "corroboration_score": float(cor),
                "report_timeliness": float(time)
            }
            
            sources[selected_idx] = {
                "source_id": f"SRC_{selected_idx + 1:03d}",
                "features": features,
                "reliability_series": reliability_ts,
                "recourse_rules": {},
                "recommendation_tier": rec_tier,
                "recommendation_score": score
            }
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ======================================================
    # 4. AUDIT & GOVERNANCE DASHBOARD
    # ======================================================
    with st.expander("🧑‍⚖️ Audit & Governance Dashboard", expanded=False):
        _render_audit_governance_section()

    # ======================================================
    # COPYRIGHT SECTION
    # ======================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border-radius: 10px; margin-top: 30px; border: 1px solid #cbd5e1; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
        <p style='color: #475569; font-size: 14px; margin: 0;'>
            © 2026 Hybrid HUMINT Tasking Dashboard. All Rights Reserved.
        </p>
            <p style='color: #64748b; font-size: 12px; margin-top: 10px;'>
                Prototype Model Developed based on Synthetic Data for Intelligence Source Performance Evaluation and Optimization| Version 1.0
            </p>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# MAIN ENTRY POINT
# ======================================================
if __name__ == "__main__" or MODE == "streamlit":
    render_streamlit_app()





