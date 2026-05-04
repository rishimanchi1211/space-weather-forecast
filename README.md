# 🌞 Space Weather Operational Forecasting

**Final Project — Time Series Analysis**

A complete applied time-series study of the Sun–Earth system, treating three monitored series both as individual processes and as components of a single operational system. The deliverable is a reproducibility notebook plus an interactive Streamlit dashboard a non-technical decision maker can use to choose a series, choose a horizon, and inspect forecasts and diagnostics from six models in three model families.

---

## 1. Why this project matters

Space weather is the variability of the Sun–Earth electromagnetic environment. It is invisible to the public but it routinely costs money and lives:

| Event | What happened |
|---|---|
| **March 1989** | Geomagnetic storm collapsed the Hydro-Québec grid, 6 million people without power for 9 hours. |
| **October 2003 (Halloween storms)** | Re-routing of polar flights cost airlines tens of millions; satellite anomalies. |
| **February 2022 (Starlink)** | 38 newly-launched Starlink satellites lost to atmospheric drag from a *minor* storm. |
| **May 2024 ("Gannon Storm")** | First G5 storm in 21 years; multiple operational impacts on grids, GNSS, satellites. |

Three publicly-available daily series describe the chain: **Sunspot Number (cause) → F10.7 cm Solar Flux (driver of thermospheric heating) → Ap planetary index (effect on Earth's magnetosphere)**. Operational decision makers — satellite operators, power-grid dispatchers, polar-flight planners, GNSS users — need short-to-medium-horizon forecasts of all three.

## 2. Data

Single combined file from **GFZ German Research Centre for Geosciences (Potsdam)**:
`https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt`

Daily observations from **1932-01-01** onwards. Three series are extracted and resampled to monthly means:

| Code | Series | Unit | Used by |
|---|---|---|---|
| `SN`   | International Sunspot Number       | count | Mission planners, modellers |
| `F107` | 10.7 cm solar radio flux           | sfu   | Satellite-drag models |
| `Ap`   | Planetary geomagnetic Ap index     | nT × 2 | Grid operators, airlines |

## 3. Models compared (six models, three families)

| Family | Model | Why included |
|---|---|---|
| **Exponential smoothing** | Simple ES | Baseline — no trend, no seasonality |
| | Holt | Linear trend |
| | Holt-Winters (s = 132 months) | 11-year solar cycle as additive seasonality |
| **Box-Jenkins** | ARIMA(p,d,q) by AIC grid search | Covers AR, MA, ARMA, ARIMA as special cases |
| **Machine learning** | Random Forest on lagged features | Tree-based; captures non-linear storm tails |
| | MLP neural network on lagged features | Neural network; same feature set, different inductive bias |

ML feature set: lags 1, 2, 3, 6, 12, 24, 132 months + rolling mean & std (windows 3, 12) + sin/cos of month-of-year.

## 4. Repository layout

```
space-weather-forecast/
├── app.py                          # Streamlit dashboard (run this)
├── data_loader.py                  # GFZ Potsdam fetch + monthly resample
├── models.py                       # Six models, unified API
├── diagnostics.py                  # Residual tests + accuracy metrics
├── requirements.txt                # Pinned dependencies
├── reproducibility_notebook.ipynb  # Full analytical narrative
├── .streamlit/config.toml          # Theme
├── .gitignore
└── README.md                       # This file
```

## 5. Run locally

```bash
git clone https://github.com/<your-username>/space-weather-forecast.git
cd space-weather-forecast
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>. The first run downloads ~2 MB of data from GFZ Potsdam and caches it for 24 h.

## 6. Deploy on Streamlit Community Cloud (free, public URL)

1. Push this repository to **GitHub** (public).
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. Click **"New app"** → choose your repo → branch `main` → main file `app.py`.
4. Click **Deploy**. In ~3 minutes you get a public URL like
   `https://your-username-space-weather-forecast.streamlit.app`.

## 7. Dashboard tour

| Tab | What it shows |
|---|---|
| **📊 Data & EDA** | Full historical series, additive decomposition (trend, 11-year seasonal, residual), ACF/PACF |
| **🔮 Forecasts** | All selected models overlaid with 95% prediction bands; selected hyper-parameters; feature importances for tree-based models |
| **🔬 Residual diagnostics** | Residuals over time, histogram, ACF, Q-Q plot; Ljung-Box and Jarque-Bera tests |
| **📋 Accuracy comparison** | MAE, RMSE, MAPE, sMAPE on a sidebar-controlled hold-out; the table highlights the best model per metric |
| **🌐 System view** | All three series stacked, cross-correlation between every pair, correlation matrix |
| **🎯 Decisions** | Threshold table mapping forecast values to operational actions, by stakeholder |

## 8. Sidebar controls

- **Target series** — SN / F10.7 / Ap
- **Forecast horizon** — 6 to 120 months
- **Years of history to train on**
- **Models to fit** — multi-select
- **Hold-out months** — for the accuracy comparison

## 9. Reproducibility

The full statistical narrative — from EDA through model identification, estimation, validation, and forecast communication — is in `reproducibility_notebook.ipynb`. It re-uses the same `data_loader.py`, `models.py`, and `diagnostics.py` modules as the dashboard, so any change to the modelling code is reflected in both.

## 10. Acknowledgements

- **GFZ German Research Centre for Geosciences (Potsdam)** for the combined Kp/Ap/SN/F10.7 archive.
- **SILSO Royal Observatory of Belgium** for upstream sunspot data.
- **NOAA Space Weather Prediction Center** for context on operational thresholds.

---
*Submitted as the final project for Time Series Analysis. Author: [your name]. Cohort of 24.*
