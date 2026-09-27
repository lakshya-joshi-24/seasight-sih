"""
SeaSight — Intelligent Freight Forecasting & Charter Decision Platform
Team Lanterns | SIH 2026 | Problem Statement SIH26006

Prototype notes for the team:
- All market data below (BDI, VLSFO bunker, FBX, shipment volumes) is SYNTHETIC,
  generated with realistic trend + seasonality + noise so the pipeline behaves
  the way real freight data behaves. Swap `generate_freight_series()` for your
  real Investing.com / Ship&Bunker / Freightos / Volza pulls later — the rest
  of the app (optimizer, matcher, risk engine, dashboard) does not need to change.
- Forecasting model: Holt-Winters Exponential Smoothing (statsmodels). Easy to
  swap for SARIMAX / XGBoost / LightGBM later — just replace `run_forecast()`.
- Everything else (UI, decision logic, explanations) is real, working logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# ============================================================================
# PAGE CONFIG + STYLE
# ============================================================================
st.set_page_config(
    page_title="SeaSight | Team Lanterns — SIH 2026",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1a2f5c"      # navy
ACCENT_ORANGE = "#f7941e"
ACCENT_GREEN = "#2ea86b"
BG_CARD = "#ffffff"

st.markdown(f"""
<style>
    .stApp {{ background-color: #f4f6fb; }}
    #MainMenu, footer {{visibility: hidden;}}

    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #24478f 60%, {ACCENT_ORANGE} 160%);
        padding: 28px 34px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(26,47,92,0.25);
    }}
    .hero h1 {{ margin: 0; font-size: 30px; font-weight: 800; }}
    .hero p {{ margin: 6px 0 0 0; opacity: 0.92; font-size: 15px; }}
    .badge {{
        display:inline-block; background: rgba(255,255,255,0.18);
        padding: 4px 12px; border-radius: 20px; font-size: 12px;
        margin-right: 8px; margin-top: 10px; font-weight: 600; letter-spacing: .3px;
    }}

    .card {{
        background: {BG_CARD}; border-radius: 14px; padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(16,24,64,0.06); border: 1px solid #eef0f6;
        margin-bottom: 14px; color: #1f2937;
    }}
    .card table {{ color: #1f2937; }}
    .card b, .card strong {{ color: #101b36; }}
    .metric-label {{ color:#6b7280; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.4px;}}
    .metric-value {{ font-size: 26px; font-weight:800; color:{PRIMARY}; }}

    .pill-low {{ background:#e6f7ee; color:#177245; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px;}}
    .pill-med {{ background:#fff4e0; color:#a5610b; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px;}}
    .pill-high {{ background:#fde8e8; color:#b42318; padding:4px 14px; border-radius:20px; font-weight:700; font-size:13px;}}

    .decision-box {{
        border-radius: 16px; padding: 26px 30px; color:white; text-align:center;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }}
    .decision-box h2 {{ margin:0; font-size:34px; font-weight:900; letter-spacing:1px;}}
    .decision-box p {{ margin-top:8px; font-size:14px; opacity:0.95;}}

    .explain-tag {{
        display:inline-block; background:#eef1fb; color:{PRIMARY}; padding:3px 10px;
        border-radius:8px; font-size:12px; font-weight:600; margin:3px 4px 3px 0;
    }}
    /* Light sidebar instead of dark: this avoids fighting Streamlit's internal
       dropdown widget styling (which is always dark-text-on-white) entirely,
       so nothing can end up invisible regardless of widget internals. */
    section[data-testid="stSidebar"] {{
        background-color: #eef1fb;
        border-right: 1px solid #dde3f5;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: {PRIMARY} !important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SYNTHETIC DATA GENERATION  (cached — swap this block for real data later)
# ============================================================================
LANES = {
    "Indonesia → Paradip (Coal)": dict(base=1450, vol=0.09, vessel_bias="Panamax"),
    "Indonesia → Visakhapatnam (Coal)": dict(base=1610, vol=0.11, vessel_bias="Supramax"),
    "Australia → Chennai (Coal)": dict(base=1980, vol=0.13, vessel_bias="Capesize"),
    "South Africa → Paradip (Coal)": dict(base=2250, vol=0.10, vessel_bias="Capesize"),
}

VESSELS = pd.DataFrame([
    {"vessel_class": "Handysize", "dwt_range": "25k–40k", "loa_m": 190, "draft_m": 10.5, "typical_rate_idx": 0.55},
    {"vessel_class": "Supramax",  "dwt_range": "50k–60k", "loa_m": 200, "draft_m": 12.2, "typical_rate_idx": 0.75},
    {"vessel_class": "Panamax",   "dwt_range": "65k–80k", "loa_m": 225, "draft_m": 13.8, "typical_rate_idx": 1.00},
    {"vessel_class": "Capesize",  "dwt_range": "150k–180k","loa_m": 300, "draft_m": 17.5, "typical_rate_idx": 1.55},
])

PORTS = pd.DataFrame([
    {"port": "Paradip",         "max_draft_m": 18.0, "max_loa_m": 310, "avg_wait_days": 2.1, "handling_rate": "Good"},
    {"port": "Visakhapatnam",   "max_draft_m": 17.0, "max_loa_m": 300, "avg_wait_days": 1.6, "handling_rate": "Good"},
    {"port": "Krishnapatnam",   "max_draft_m": 18.5, "max_loa_m": 320, "avg_wait_days": 1.2, "handling_rate": "Excellent"},
    {"port": "Ennore (Kamarajar)", "max_draft_m": 13.5, "max_loa_m": 230, "avg_wait_days": 3.4, "handling_rate": "Moderate"},
    {"port": "Haldia",           "max_draft_m": 11.5, "max_loa_m": 200, "avg_wait_days": 4.0, "handling_rate": "Moderate"},
])


@st.cache_data
def generate_freight_series(lane_name: str, days: int = 730, seed: int = 7):
    cfg = LANES[lane_name]
    rng = np.random.default_rng(abs(hash(lane_name)) % (2**32) + seed)
    t = np.arange(days)

    trend = cfg["base"] * (1 + 0.00025 * t)                                   # gentle upward drift
    seasonal = cfg["base"] * 0.07 * np.sin(2 * np.pi * t / 90)                # quarterly seasonality
    weekly = cfg["base"] * 0.015 * np.sin(2 * np.pi * t / 7)                  # weekly micro-noise
    shock = np.zeros(days)
    for _ in range(int(days / 150)):                                         # occasional market shocks
        idx = rng.integers(60, days - 30)
        shock[idx: idx + 25] += rng.choice([-1, 1]) * cfg["base"] * rng.uniform(0.08, 0.18)
    noise = rng.normal(0, cfg["base"] * cfg["vol"] * 0.35, days)

    rate = trend + seasonal + weekly + shock + noise
    rate = np.maximum(rate, cfg["base"] * 0.4)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    df = pd.DataFrame({"date": dates, "rate": rate})

    # correlated auxiliary indices (bunker & container) for the dashboard
    df["vlsfo_bunker"] = 560 + 0.02 * (rate - cfg["base"]) + rng.normal(0, 8, days)
    df["fbx_container"] = 1800 + 0.6 * (rate - cfg["base"]) + rng.normal(0, 40, days)
    return df


# ============================================================================
# FORECASTING ENGINE
# ============================================================================
def fit_holt_winters(train_series: pd.Series, seasonal_periods: int = 90):
    model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add",
        seasonal_periods=seasonal_periods, initialization_method="estimated",
    )
    return model.fit(optimized=True)


def run_forecast(df: pd.DataFrame, horizon: int = 30):
    fit = fit_holt_winters(df["rate"])
    forecast = fit.forecast(horizon)
    resid_std = float(np.std(fit.resid))
    upper = forecast + 1.96 * resid_std
    lower = np.maximum(forecast - 1.96 * resid_std, 0)
    future_dates = pd.date_range(df["date"].iloc[-1] + timedelta(days=1), periods=horizon, freq="D")
    return future_dates, forecast.values, upper.values, lower.values, resid_std


def validate_model(df: pd.DataFrame, test_days: int = 60):
    train, test = df.iloc[:-test_days], df.iloc[-test_days:]
    fit = fit_holt_winters(train["rate"])
    model_pred = fit.forecast(test_days).values
    baseline_pred = np.repeat(train["rate"].iloc[-1], test_days)              # persistence baseline
    actual = test["rate"].values

    def metrics(pred):
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))
        dir_actual = np.sign(np.diff(np.concatenate([[train["rate"].iloc[-1]], actual])))
        dir_pred = np.sign(np.diff(np.concatenate([[train["rate"].iloc[-1]], pred])))
        dir_acc = np.mean(dir_actual == dir_pred) * 100
        return mae, rmse, dir_acc

    m_mae, m_rmse, m_dir = metrics(model_pred)
    b_mae, b_rmse, b_dir = metrics(baseline_pred)
    return dict(test=test, model_pred=model_pred, baseline_pred=baseline_pred,
                model=(m_mae, m_rmse, m_dir), baseline=(b_mae, b_rmse, b_dir))


# ============================================================================
# CHARTER TIMING OPTIMIZER
# ============================================================================
def charter_decision(current_rate, forecast_vals, resid_std):
    pct_change = (np.mean(forecast_vals) - current_rate) / current_rate * 100
    volatility_pct = resid_std / current_rate * 100

    if pct_change <= -3.0:
        label, color = "WAIT", "#177245"
        reason = f"Rates are projected to fall ~{abs(pct_change):.1f}% over the forecast window — delaying entry likely lowers cost."
    elif pct_change >= 3.0:
        label, color = "CHARTER NOW", "#b42318"
        reason = f"Rates are projected to rise ~{pct_change:.1f}% — locking in a charter today avoids higher future cost."
    else:
        label, color = "MONITOR", "#a5610b"
        reason = f"Forecast is roughly flat ({pct_change:+.1f}%) — hold position and re-check as new data arrives."
    return label, color, reason, pct_change, volatility_pct


# ============================================================================
# RISK ENGINE
# ============================================================================
def compute_risk(resid_std, current_rate, port_wait_days):
    cv = resid_std / current_rate
    wait_penalty = min(port_wait_days / 5, 1.0) * 0.02
    score = cv + wait_penalty

    if score < 0.05:
        level, css = "Low", "pill-low"
    elif score < 0.09:
        level, css = "Medium", "pill-med"
    else:
        level, css = "High", "pill-high"

    factors = [
        f"Forecast volatility (CV): {cv*100:.1f}%",
        f"Port congestion contribution: {wait_penalty*100:.1f}%",
    ]
    return level, css, score, factors


# ============================================================================
# SIDEBAR — GLOBAL CONTROLS
# ============================================================================
with st.sidebar:
    st.markdown("### 🚢 SeaSight Controls")
    st.caption("Team Lanterns · SIH26006")
    lane = st.selectbox("Trade Lane", list(LANES.keys()))
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30, step=7)
    port_choice = st.selectbox("Destination Port", PORTS["port"].tolist())

df = generate_freight_series(lane)
current_rate = df["rate"].iloc[-1]
future_dates, fc_vals, fc_upper, fc_lower, resid_std = run_forecast(df, horizon)
port_row = PORTS[PORTS["port"] == port_choice].iloc[0]
decision_label, decision_color, decision_reason, pct_change, vol_pct = charter_decision(current_rate, fc_vals, resid_std)
risk_level, risk_css, risk_score, risk_factors = compute_risk(resid_std, current_rate, port_row["avg_wait_days"])

# ============================================================================
# HERO
# ============================================================================
st.markdown(f"""
<div class="hero">
    <h1>🚢 SeaSight — Intelligent Freight Forecasting Platform</h1>
    <p>From checking the market every day → predicting trends and chartering with confidence.</p>
    <span class="badge">SIH26006</span>
    <span class="badge">Team Lanterns</span>
    <span class="badge">Theme: Transportation &amp; Logistics</span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Overview", "📈 Forecasting Engine", "🧪 Validation", "⏱️ Charter Timing Optimizer",
                "⚓ Vessel & Port Matcher", "⚠️ Risk Engine", "🎯 Recommendation"])

# ---------------------------------------------------------------- OVERVIEW
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Current Rate", f"${current_rate:,.0f}", lane.split("(")[0].strip()),
        (c2, f"{horizon}-Day Forecast Avg", f"${np.mean(fc_vals):,.0f}", f"{pct_change:+.1f}% vs today"),
        (c3, "Model Volatility (σ)", f"{vol_pct:.1f}%", "residual std / current rate"),
        (c4, "Recommended Action", decision_label, ""),
    ]:
        with col:
            st.markdown(f"""<div class="card"><div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div style="color:#9aa2b1;font-size:12px;">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("#### End-to-end pipeline")
    st.markdown("""
    <div class="card">
    Data Sources → Feature Engineering → <b>Forecasting Engine</b> → <b>Charter Timing Optimizer</b>
    → <b>Vessel &amp; Port Matcher</b> → <b>Risk Engine</b> → Manager Dashboard
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"].tail(180), y=df["rate"].tail(180), name="Historical Rate",
                              line=dict(color=PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=fc_vals, name="Forecast", line=dict(color=ACCENT_ORANGE, width=2, dash="dash")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                       title="Freight Rate — last 180 days + forecast")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- FORECASTING
with tabs[1]:
    st.markdown("#### Forecasting Engine — Holt-Winters Exponential Smoothing")
    st.caption("Trained on the full historical window shown below; produces point forecast + 95% confidence band.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rate"], name="Historical", line=dict(color=PRIMARY, width=1.6)))
    fig.add_trace(go.Scatter(x=future_dates, y=fc_vals, name="Forecast", line=dict(color=ACCENT_ORANGE, width=2.4)))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(fc_upper) + list(fc_lower[::-1]),
        fill="toself", fillcolor="rgba(247,148,30,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval"))
    fig.update_layout(height=440, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Auxiliary indices (correlated inputs)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"].tail(180), y=df["vlsfo_bunker"].tail(180), name="VLSFO Bunker ($/t)"))
        fig2.add_trace(go.Scatter(x=df["date"].tail(180), y=df["fbx_container"].tail(180), name="FBX Container Idx"))
        fig2.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.markdown("##### Explainable AI — what's driving this forecast")
        st.caption("Illustrative attribution for the demo model; production version reports true SHAP-style feature importances.")
        drivers = pd.DataFrame({
            "Factor": ["Trend (base drift)", "Seasonal cycle (90-day)", "Weekly micro-pattern", "Bunker cost correlation", "Recent shocks"],
            "Contribution %": [42, 27, 9, 14, 8],
        })
        fig3 = go.Figure(go.Bar(x=drivers["Contribution %"], y=drivers["Factor"], orientation="h",
                                 marker_color=ACCENT_GREEN))
        fig3.update_layout(height=300, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------- VALIDATION
with tabs[2]:
    st.markdown("#### Model Validation — vs. Persistence Baseline")
    st.caption("Mirrors the real evaluation plan: train on the earlier period, hold out the most recent period as a test set, and benchmark against a naive persistence baseline.")
    val = validate_model(df, test_days=60)
    m_mae, m_rmse, m_dir = val["model"]
    b_mae, b_rmse, b_dir = val["baseline"]

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{m_mae:,.1f}", f"{m_mae - b_mae:+.1f} vs baseline", delta_color="inverse")
    c2.metric("RMSE", f"{m_rmse:,.1f}", f"{m_rmse - b_rmse:+.1f} vs baseline", delta_color="inverse")
    c3.metric("Directional Accuracy", f"{m_dir:.1f}%", f"{m_dir - b_dir:+.1f} pts vs baseline")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=val["test"]["date"], y=val["test"]["rate"], name="Actual", line=dict(color=PRIMARY, width=2)))
    fig4.add_trace(go.Scatter(x=val["test"]["date"], y=val["model_pred"], name="Model Forecast", line=dict(color=ACCENT_ORANGE, width=2)))
    fig4.add_trace(go.Scatter(x=val["test"]["date"], y=val["baseline_pred"], name="Persistence Baseline",
                               line=dict(color="#9aa2b1", width=1.6, dash="dot")))
    fig4.update_layout(height=380, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10),
                        title="Held-out test window: model vs. baseline vs. actual")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------- CHARTER TIMING
with tabs[3]:
    st.markdown("#### Charter Timing Optimizer")
    st.markdown(f"""
    <div class="decision-box" style="background:{decision_color};">
        <h2>{decision_label}</h2>
        <p>{decision_reason}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="card"><div class="metric-label">Current Rate</div>
                <div class="metric-value">${current_rate:,.0f}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="card"><div class="metric-label">{horizon}-Day Avg Forecast</div>
                <div class="metric-value">${np.mean(fc_vals):,.0f}</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="card"><div class="metric-label">Projected Change</div>
                <div class="metric-value">{pct_change:+.1f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("##### Decision thresholds (transparent — no black box)")
    st.markdown("""
    <div class="card">
    <span class="explain-tag">Forecast ≤ −3%</span> → <b>WAIT</b> &nbsp;|&nbsp;
    <span class="explain-tag">−3% to +3%</span> → <b>MONITOR</b> &nbsp;|&nbsp;
    <span class="explain-tag">Forecast ≥ +3%</span> → <b>CHARTER NOW</b>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------- VESSEL & PORT MATCHER
with tabs[4]:
    st.markdown(f"#### Vessel & Port Matcher — checking against **{port_choice}**")
    compatible = VESSELS[(VESSELS["draft_m"] <= port_row["max_draft_m"]) & (VESSELS["loa_m"] <= port_row["max_loa_m"])].copy()
    incompatible = VESSELS[~VESSELS["vessel_class"].isin(compatible["vessel_class"])]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Port constraints")
        st.markdown(f"""
        <div class="card">
        <b>Max Draft:</b> {port_row['max_draft_m']} m<br>
        <b>Max LOA:</b> {port_row['max_loa_m']} m<br>
        <b>Avg Waiting Time:</b> {port_row['avg_wait_days']} days<br>
        <b>Cargo Handling Rate:</b> {port_row['handling_rate']}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("##### ✅ Compatible vessel classes")
        st.dataframe(compatible[["vessel_class", "dwt_range", "loa_m", "draft_m"]], use_container_width=True, hide_index=True)
        if len(incompatible):
            st.markdown("##### ❌ Excluded (exceeds port limits)")
            st.dataframe(incompatible[["vessel_class", "dwt_range", "loa_m", "draft_m"]], use_container_width=True, hide_index=True)

    lane_bias = LANES[lane]["vessel_bias"]
    if lane_bias in compatible["vessel_class"].values:
        st.success(f"Recommended class for this lane: **{lane_bias}** — compatible with {port_choice} and typical for this cargo volume.")
    else:
        st.warning(f"This lane typically uses **{lane_bias}**, but that class exceeds {port_choice}'s limits — consider transshipment or an alternate port.")

# ---------------------------------------------------------------- RISK ENGINE
with tabs[5]:
    st.markdown("#### Risk Engine")
    c1, c2 = st.columns([1, 2])
    with c1:
        pill_class = risk_css
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="metric-label">Overall Risk Level</div>
            <div style="margin-top:10px;"><span class="{pill_class}">{risk_level.upper()}</span></div>
            <div style="margin-top:10px;color:#6b7280;font-size:13px;">Composite score: {risk_score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("##### Why this score — contributing factors")
        for f in risk_factors:
            st.markdown(f"- {f}")
        st.caption("Every risk score is traceable to its inputs — no unexplained outputs, matching the 'no black box' principle from the design.")

# ---------------------------------------------------------------- RECOMMENDATION
with tabs[6]:
    st.markdown("#### 🎯 Final Recommendation")
    st.markdown(f"""
    <div class="card">
    <table style="width:100%; font-size:15px;">
    <tr><td style="padding:8px 0;"><b>Trade Lane</b></td><td>{lane}</td></tr>
    <tr><td style="padding:8px 0;"><b>Destination Port</b></td><td>{port_choice}</td></tr>
    <tr><td style="padding:8px 0;"><b>Recommended Vessel Class</b></td><td>{lane_bias if lane_bias in compatible['vessel_class'].values else compatible['vessel_class'].iloc[-1] if len(compatible) else "None compatible"}</td></tr>
    <tr><td style="padding:8px 0;"><b>Charter Decision</b></td><td><b style="color:{decision_color};">{decision_label}</b></td></tr>
    <tr><td style="padding:8px 0;"><b>Risk Level</b></td><td>{risk_level}</td></tr>
    <tr><td style="padding:8px 0;"><b>{horizon}-Day Forecast Range</b></td><td>${fc_lower.mean():,.0f} – ${fc_upper.mean():,.0f}</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
    st.info("💡 This is where the platform ties every engine together into one actionable call for the procurement manager — "
            "forecast, timing, vessel fit, and risk, in one screen instead of four spreadsheets.")

st.markdown("---")
st.caption("SeaSight prototype · Team Lanterns · SIH26006 · Built for internal-round demo. "
           "Market data simulated — real BDI/VLSFO/FBX/Volza integration planned before final submission.")
