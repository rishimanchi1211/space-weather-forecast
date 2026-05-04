"""
Space Weather Operational Forecasting — Streamlit dashboard.

Final project for Time Series Analysis. The dashboard is the communication
tool a non-technical decision maker (satellite-operations engineer, grid-control
dispatcher, airline-ops planner) would actually use to:

    1. pick which series to monitor (Sunspot Number, F10.7, or Ap),
    2. choose a forecast horizon,
    3. compare six models from three model families side by side,
    4. read residual diagnostics to know whether each model is trustworthy,
    5. see the cross-series picture (the system view).

Run locally:    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_loader
import diagnostics as dg
import models as mdl

# --------------------------------------------------------------------------- #
#  Page setup                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Space Weather Forecasting",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .small-note { font-size: 0.85rem; color: #666; }
    .metric-good { color: #2ca02c; font-weight: 600; }
    .metric-bad  { color: #d62728; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  Header                                                                      #
# --------------------------------------------------------------------------- #
st.title("🌞 Space Weather Operational Forecasting")
st.markdown(
    "**Forecasting the Sun-Earth system** — Sunspot Number, 10.7 cm Solar Flux, "
    "and the planetary geomagnetic Ap index — with three families of models "
    "(Exponential Smoothing • Box-Jenkins • Machine Learning). "
    "Built for the people who actually use these forecasts: satellite-operations "
    "engineers, power-grid dispatchers, and polar-flight planners."
)

# --------------------------------------------------------------------------- #
#  Load data                                                                   #
# --------------------------------------------------------------------------- #
try:
    with st.spinner("Loading space-weather data from GFZ Potsdam…"):
        monthly = data_loader.load_monthly()
except Exception as exc:  # noqa: BLE001
    st.error(
        f"**Data load failed:** {exc}\n\n"
        "The dashboard fetches the file `Kp_ap_Ap_SN_F107_since_1932.txt` from "
        "GFZ Potsdam at <https://kp.gfz.de>. Run `python data_loader.py` once "
        "with internet access to populate the local cache, then redeploy."
    )
    st.stop()

if monthly.empty:
    st.error("Loaded data is empty — check `data_loader.py`.")
    st.stop()

# --------------------------------------------------------------------------- #
#  Sidebar — controls                                                          #
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Controls")

series_choice = st.sidebar.selectbox(
    "Target series",
    options=["SN", "F107", "Ap"],
    format_func=lambda k: f"{k} — {data_loader.SERIES_META[k]['label']}",
    index=0,
)

st.sidebar.caption(data_loader.SERIES_META[series_choice]["desc"])

horizon = st.sidebar.slider(
    "Forecast horizon (months ahead)",
    min_value=6, max_value=120, value=36, step=6,
    help="How many months into the future to forecast. The 11-year solar cycle "
         "means horizons up to 132 months show full-cycle behaviour.",
)

train_years = st.sidebar.slider(
    "Years of history to train on",
    min_value=20, max_value=int((monthly.index[-1] - monthly.index[0]).days / 365),
    value=60, step=5,
)

selected_models = st.sidebar.multiselect(
    "Models to fit",
    options=list(mdl.MODEL_REGISTRY.keys()),
    default=list(mdl.MODEL_REGISTRY.keys()),
)

holdout_months = st.sidebar.slider(
    "Hold-out months for accuracy comparison",
    min_value=12, max_value=120, value=36, step=6,
    help="Reserved test set used to compute MAE/RMSE/MAPE. "
         "Forecasts in the main chart are produced from the FULL series.",
)

# Slice the series to the chosen training window
y_full = monthly[series_choice].copy()
start_date = y_full.index[-1] - pd.DateOffset(years=train_years)
y = y_full.loc[y_full.index >= start_date]
unit = data_loader.SERIES_META[series_choice]["unit"]
label = data_loader.SERIES_META[series_choice]["label"]

# Prepare a hold-out split for the accuracy comparison table
y_train_eval = y.iloc[:-holdout_months]
y_test_eval  = y.iloc[-holdout_months:]

# --------------------------------------------------------------------------- #
#  Header KPIs                                                                 #
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest month", y.index[-1].strftime("%b %Y"))
c2.metric(f"Latest {label}", f"{y.iloc[-1]:.1f} {unit}")
c3.metric("Cycle position",
          f"{y.iloc[-12:].mean()/y.max()*100:.0f}% of historical max")
c4.metric("Training points", f"{len(y):,}")

st.divider()

# --------------------------------------------------------------------------- #
#  Tabs                                                                        #
# --------------------------------------------------------------------------- #
tab_data, tab_fc, tab_diag, tab_acc, tab_system, tab_decision = st.tabs([
    "📊 Data & EDA",
    "🔮 Forecasts",
    "🔬 Residual diagnostics",
    "📋 Accuracy comparison",
    "🌐 System view",
    "🎯 Decisions",
])


# =========================================================================== #
#  TAB 1 — DATA & EDA                                                          #
# =========================================================================== #
with tab_data:
    st.subheader(f"Historical {label}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, mode="lines",
        name=label, line=dict(color="#ff8c00", width=1.5),
    ))
    fig.update_layout(
        height=400, xaxis_title="Date",
        yaxis_title=f"{label} ({unit})",
        margin=dict(l=10, r=10, t=30, b=10),
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Decomposition**")
    from statsmodels.tsa.seasonal import seasonal_decompose
    try:
        period = 132 if len(y) >= 264 else 12
        dec = seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
        sub = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        for i, comp in enumerate([dec.observed, dec.trend, dec.seasonal, dec.resid], start=1):
            sub.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines",
                                     line=dict(width=1)), row=i, col=1)
        sub.update_layout(height=600, showlegend=False, template="simple_white",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(sub, use_container_width=True)
        st.caption(
            f"Seasonal period set to **{period} months** "
            f"({'≈ 11-year solar cycle' if period == 132 else 'annual fallback'})."
        )
    except Exception as e:  # noqa: BLE001
        st.info(f"Decomposition unavailable: {e}")

    st.markdown("**ACF & PACF**")
    from statsmodels.tsa.stattools import acf, pacf
    n_lags = min(48, len(y) // 4)
    a_vals = acf(y.dropna(), nlags=n_lags, fft=True)
    p_vals = pacf(y.dropna(), nlags=n_lags)
    bound = 1.96 / np.sqrt(len(y))
    sub = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
    for col, vals in enumerate([a_vals, p_vals], start=1):
        sub.add_trace(go.Bar(x=list(range(len(vals))), y=vals,
                             marker_color="#1f77b4"), row=1, col=col)
        sub.add_hline(y=bound, line_dash="dash", line_color="red", row=1, col=col)
        sub.add_hline(y=-bound, line_dash="dash", line_color="red", row=1, col=col)
    sub.update_layout(height=300, showlegend=False, template="simple_white",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sub, use_container_width=True)


# =========================================================================== #
#  TAB 2 — FORECASTS                                                           #
# =========================================================================== #
with tab_fc:
    st.subheader(f"Forecasts for {label} — next {horizon} months")
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
    else:
        with st.spinner("Fitting models on full history…"):
            results = mdl.run_all(y, horizon, selected=selected_models)

        # Plot
        colours = {
            "Simple Exponential Smoothing": "#1f77b4",
            "Holt":                         "#17becf",
            "Holt-Winters":                 "#2ca02c",
            "ARIMA (auto)":                 "#d62728",
            "Random Forest":                "#9467bd",
            "Neural Network (MLP)":         "#e377c2",
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, mode="lines", name="Observed",
            line=dict(color="black", width=1.5),
        ))
        for name, r in results.items():
            if r["forecast"] is None:
                continue
            c = colours.get(name, "#777")
            # Prediction band
            fig.add_trace(go.Scatter(
                x=list(r["forecast"].index) + list(r["forecast"].index[::-1]),
                y=list(r["upper"].values) + list(r["lower"].values[::-1]),
                fill="toself", fillcolor=c, opacity=0.10,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip", name=f"{name} 95% PI",
            ))
            # Point forecast
            fig.add_trace(go.Scatter(
                x=r["forecast"].index, y=r["forecast"].values,
                mode="lines", name=r["name"],
                line=dict(color=c, width=2, dash="dot"),
            ))
        fig.update_layout(
            height=520, template="simple_white",
            xaxis_title="Date", yaxis_title=f"{label} ({unit})",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Selected hyper-parameters**")
        for name, r in results.items():
            if r.get("error"):
                st.error(f"{name}: {r['error']}")
                continue
            with st.expander(f"⚙️ {r['name']}"):
                params = {k: v for k, v in r["params"].items()
                          if k != "feature_importance"}
                st.json(params, expanded=False)
                if "feature_importance" in r["params"]:
                    fi = pd.Series(r["params"]["feature_importance"]
                                   ).sort_values(ascending=True)
                    fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h"))
                    fig_fi.update_layout(height=300, template="simple_white",
                                         title="Feature importance",
                                         margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_fi, use_container_width=True)


# =========================================================================== #
#  TAB 3 — RESIDUAL DIAGNOSTICS                                                #
# =========================================================================== #
with tab_diag:
    st.subheader("Residual diagnostics")
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
    else:
        # Reuse the results from tab 2 (Streamlit re-runs are cheap because of caches)
        with st.spinner("Fitting models…"):
            results = mdl.run_all(y, horizon, selected=selected_models)

        which = st.selectbox("Inspect which model?",
                             options=[r["name"] for r in results.values()
                                      if r["residuals"] is not None])

        chosen = next(r for r in results.values() if r["name"] == which)
        resid = chosen["residuals"].dropna()

        col1, col2, col3, col4 = st.columns(4)
        lb = dg.ljung_box(resid)
        nm = dg.normality(resid)
        col1.metric("Mean residual", f"{resid.mean():.3f}")
        col2.metric("Std residual", f"{resid.std():.3f}")
        col3.metric("Ljung-Box p", f"{lb['p_value']:.3f}",
                    delta="white-noise ✅" if lb["white_noise"] else "autocorr ❌",
                    delta_color="normal" if lb["white_noise"] else "inverse")
        col4.metric("Jarque-Bera p", f"{nm['jb_p']:.3f}",
                    delta="normal ✅" if nm["normal"] else "non-normal ❌",
                    delta_color="normal" if nm["normal"] else "inverse")

        sub = make_subplots(rows=2, cols=2,
                            subplot_titles=("Residuals over time", "Histogram",
                                            "ACF of residuals", "Q-Q plot"))
        # 1. residuals
        sub.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines",
                                 line=dict(width=1, color="#1f77b4")),
                      row=1, col=1)
        sub.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        # 2. histogram
        sub.add_trace(go.Histogram(x=resid.values, nbinsx=40,
                                   marker_color="#1f77b4"), row=1, col=2)
        # 3. ACF
        acf_df = dg.residual_acf(resid, n_lags=min(36, len(resid)//4))
        sub.add_trace(go.Bar(x=acf_df["lag"], y=acf_df["acf"],
                             marker_color="#1f77b4"), row=2, col=1)
        sub.add_hline(y=acf_df["upper"].iloc[0], line_dash="dash",
                      line_color="red", row=2, col=1)
        sub.add_hline(y=acf_df["lower"].iloc[0], line_dash="dash",
                      line_color="red", row=2, col=1)
        # 4. Q-Q
        from scipy import stats as sps
        qs = np.linspace(0.01, 0.99, 50)
        theo = sps.norm.ppf(qs, loc=resid.mean(), scale=resid.std())
        emp = np.quantile(resid.values, qs)
        sub.add_trace(go.Scatter(x=theo, y=emp, mode="markers",
                                 marker=dict(color="#1f77b4")),
                      row=2, col=2)
        sub.add_trace(go.Scatter(x=theo, y=theo, mode="lines",
                                 line=dict(color="red", dash="dash")),
                      row=2, col=2)
        sub.update_layout(height=600, showlegend=False, template="simple_white",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(sub, use_container_width=True)

        st.info(
            "**How to read these:** "
            "Residuals should look like white noise — no trend, no autocorrelation, "
            "roughly normal. If the Ljung-Box p-value is below 0.05, the model has "
            "left structure on the table and forecasts are likely biased."
        )


# =========================================================================== #
#  TAB 4 — ACCURACY COMPARISON                                                 #
# =========================================================================== #
with tab_acc:
    st.subheader(f"Accuracy on the last {holdout_months} months (hold-out)")
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
    else:
        with st.spinner("Re-fitting on training portion…"):
            eval_results = mdl.run_all(y_train_eval, holdout_months,
                                       selected=selected_models)

        rows = []
        for name, r in eval_results.items():
            if r["forecast"] is None:
                rows.append({"Model": name, **{k: np.nan for k in
                            ["MAE", "RMSE", "MAPE %", "sMAPE %"]}, "n": 0})
                continue
            metrics = dg.evaluate(y_test_eval, r["forecast"])
            rows.append({"Model": r["name"], **metrics})
        comp = pd.DataFrame(rows).set_index("Model")
        comp_sorted = comp.sort_values("RMSE")

        st.dataframe(
            comp_sorted.style.format({
                "MAE": "{:.2f}", "RMSE": "{:.2f}",
                "MAPE %": "{:.1f}", "sMAPE %": "{:.1f}",
            }).highlight_min(subset=["MAE", "RMSE", "MAPE %", "sMAPE %"],
                             color="#cdeac0"),
            use_container_width=True,
        )

        # Visual side-by-side
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_train_eval.index[-60:],
                                 y=y_train_eval.iloc[-60:].values,
                                 mode="lines", name="Train (last 60m)",
                                 line=dict(color="black", width=1)))
        fig.add_trace(go.Scatter(x=y_test_eval.index, y=y_test_eval.values,
                                 mode="lines+markers", name="Actual hold-out",
                                 line=dict(color="black", width=2)))
        for name, r in eval_results.items():
            if r["forecast"] is None:
                continue
            fig.add_trace(go.Scatter(x=r["forecast"].index, y=r["forecast"].values,
                                     mode="lines", name=r["name"],
                                     line=dict(width=1.5, dash="dot")))
        fig.update_layout(
            height=420, template="simple_white",
            xaxis_title="Date", yaxis_title=f"{label} ({unit})",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        best = comp_sorted.index[0]
        st.success(f"**Best model on this hold-out:** {best} "
                   f"(RMSE = {comp_sorted.loc[best, 'RMSE']:.2f} {unit})")


# =========================================================================== #
#  TAB 5 — SYSTEM VIEW                                                         #
# =========================================================================== #
with tab_system:
    st.subheader("The Sun–Earth system: three series, one phenomenon")
    st.markdown(
        "Sunspots (the cause) drive the 10.7 cm radio flux (the proxy for "
        "thermospheric heating) which, when coronal mass ejections hit Earth, "
        "shows up as elevated geomagnetic Ap. Looking at them together is how "
        "operational forecasters actually think about the problem."
    )
    sub = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=tuple(
                            f"{data_loader.SERIES_META[k]['label']} "
                            f"({data_loader.SERIES_META[k]['unit']})"
                            for k in ["SN", "F107", "Ap"]))
    colors = {"SN": "#ff8c00", "F107": "#1f77b4", "Ap": "#d62728"}
    for i, k in enumerate(["SN", "F107", "Ap"], start=1):
        sub.add_trace(go.Scatter(x=monthly.index, y=monthly[k].values,
                                 mode="lines", line=dict(color=colors[k], width=1),
                                 name=k), row=i, col=1)
    sub.update_layout(height=620, showlegend=False, template="simple_white",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sub, use_container_width=True)

    st.markdown("**Cross-correlations (lead/lag in months)**")
    from scipy.signal import correlate
    pairs = [("SN", "F107"), ("SN", "Ap"), ("F107", "Ap")]
    sub = make_subplots(rows=1, cols=3, subplot_titles=tuple(f"{a} vs {b}"
                                                              for a, b in pairs))
    for col, (a, b) in enumerate(pairs, start=1):
        x = (monthly[a] - monthly[a].mean()) / monthly[a].std()
        z = (monthly[b] - monthly[b].mean()) / monthly[b].std()
        c = correlate(z.dropna().values, x.dropna().values, mode="full")
        c = c / max(len(x), len(z))
        lags = np.arange(-len(x) + 1, len(x))
        mid = len(c) // 2
        window = 60
        sub.add_trace(go.Bar(x=lags[mid - window: mid + window],
                             y=c[mid - window: mid + window],
                             marker_color="#1f77b4"), row=1, col=col)
        sub.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=col)
    sub.update_layout(height=320, showlegend=False, template="simple_white",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(sub, use_container_width=True)
    st.caption(
        "A peak at a positive lag means the second series follows the first. "
        "We typically see SN leading F10.7 by 0–1 months and Ap by 1–3 months — "
        "the operational signal that justifies forecasting from the upstream driver."
    )

    st.markdown("**Pearson correlations on monthly means**")
    st.dataframe(monthly.corr().round(3), use_container_width=True)


# =========================================================================== #
#  TAB 6 — DECISIONS                                                           #
# =========================================================================== #
with tab_decision:
    st.subheader("Translating forecasts into decisions")
    st.markdown(
        """
| Stakeholder | Series watched | Threshold (rough) | Action |
|---|---|---|---|
| **Satellite operator** (LEO) | F10.7 | > 150 sfu sustained | Tighter station-keeping budget; review re-entry predictions for end-of-life craft |
| **Power-grid dispatcher** | Ap | > 50 nT | Prepare for Geomagnetically Induced Currents; stage transformers, throttle long lines |
| **Polar-flight planner** | Ap | > 30 nT | Re-route sub-polar flights, reduce cruise altitude, avoid HF-only sectors |
| **GNSS/precision-ag user** | Ap | > 30 nT | Switch to backup positioning; pause RTK-critical operations |
| **Mission planner** (deep space) | SN, F10.7 | cycle phase | Schedule launches around solar minima for sensitive instruments |

The forecasts in this dashboard are *probabilistic*: the prediction interval matters as much as
the point forecast. A point forecast of Ap = 25 with a 95% upper bound of 70 still triggers
the polar-flight protocol — operational decisions are made on the *upper tail*, not the mean.
        """
    )
    st.info(
        "**Why three model families?** ETS handles smooth trend and the 11-year cycle; "
        "ARIMA handles short-run autocorrelation; tree-based and neural models handle "
        "the non-linear, heavy-tailed storm tails that linear models systematically "
        "under-predict. No single model wins on every horizon — that is the whole "
        "reason the comparison table exists."
    )

# --------------------------------------------------------------------------- #
#  Footer                                                                      #
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "Data: GFZ Potsdam — Kp/Ap/SN/F10.7 (since 1932). · "
    "Models: statsmodels, scikit-learn. · "
    "Built as the final project for Time Series Analysis."
)
