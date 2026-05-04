"""
Forecasting models — three families required by the assignment.

1. Exponential smoothing  : Simple ES, Holt, Holt-Winters (statsmodels)
2. Box-Jenkins            : ARIMA fit by AIC grid search          (statsmodels)
3. Machine Learning       : RandomForest (tree-based) + MLP (neural net) on lagged features (sklearn)

Every fit-function returns a dict with the same keys so the dashboard can
treat them uniformly:

    {
        "name":          str,
        "fitted":        pd.Series   indexed by training dates,
        "forecast":      pd.Series   indexed by future dates,
        "lower":         pd.Series   95% prediction interval lower bound,
        "upper":         pd.Series   95% prediction interval upper bound,
        "residuals":     pd.Series   training residuals,
        "params":        dict        chosen hyper-parameters,
    }
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _future_index(last_date: pd.Timestamp, horizon: int, freq: str = "MS") -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                         periods=horizon, freq=freq)


def _empty_result(name: str) -> dict:
    return {"name": name, "fitted": None, "forecast": None,
            "lower": None, "upper": None, "residuals": None, "params": {}}


# --------------------------------------------------------------------------- #
#  1. Exponential smoothing                                                    #
# --------------------------------------------------------------------------- #
def fit_simple_exp(y: pd.Series, horizon: int) -> dict:
    """Simple Exponential Smoothing — for series with no trend / seasonality."""
    model = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
    fitted = model.fittedvalues
    forecast = model.forecast(horizon)
    forecast.index = _future_index(y.index[-1], horizon)
    resid = (y - fitted).dropna()
    sigma = resid.std()
    z = 1.96
    return {
        "name": "Simple ES",
        "fitted": fitted,
        "forecast": forecast,
        "lower": forecast - z * sigma,
        "upper": forecast + z * sigma,
        "residuals": resid,
        "params": {"alpha": float(model.params["smoothing_level"])},
    }


def fit_holt(y: pd.Series, horizon: int) -> dict:
    """Holt's linear trend method."""
    model = Holt(y, initialization_method="estimated").fit(optimized=True)
    fitted = model.fittedvalues
    forecast = model.forecast(horizon)
    forecast.index = _future_index(y.index[-1], horizon)
    resid = (y - fitted).dropna()
    sigma = resid.std()
    h = np.arange(1, horizon + 1)
    band = 1.96 * sigma * np.sqrt(h)  # widening interval with horizon
    return {
        "name": "Holt",
        "fitted": fitted,
        "forecast": forecast,
        "lower": pd.Series(forecast.values - band, index=forecast.index),
        "upper": pd.Series(forecast.values + band, index=forecast.index),
        "residuals": resid,
        "params": {
            "alpha": float(model.params["smoothing_level"]),
            "beta":  float(model.params["smoothing_trend"]),
        },
    }


def fit_holt_winters(y: pd.Series, horizon: int, season_length: int = 132) -> dict:
    """
    Holt-Winters with the *11-year* solar cycle as seasonality.
    Monthly data ⇒ season length ≈ 11 yrs × 12 ≈ 132 months.
    Falls back to a 12-month seasonality if the series is too short.
    """
    if len(y) < 2 * season_length:
        season_length = 12
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=season_length,
        initialization_method="estimated",
    ).fit(optimized=True)
    fitted = model.fittedvalues
    forecast = model.forecast(horizon)
    forecast.index = _future_index(y.index[-1], horizon)
    resid = (y - fitted).dropna()
    sigma = resid.std()
    h = np.arange(1, horizon + 1)
    band = 1.96 * sigma * np.sqrt(h)
    return {
        "name": f"Holt-Winters (s={season_length})",
        "fitted": fitted,
        "forecast": forecast,
        "lower": pd.Series(forecast.values - band, index=forecast.index),
        "upper": pd.Series(forecast.values + band, index=forecast.index),
        "residuals": resid,
        "params": {
            "alpha": float(model.params["smoothing_level"]),
            "beta":  float(model.params["smoothing_trend"]),
            "gamma": float(model.params["smoothing_seasonal"]),
            "seasonal_periods": season_length,
        },
    }


# --------------------------------------------------------------------------- #
#  2. Box-Jenkins (ARIMA)                                                      #
# --------------------------------------------------------------------------- #
def fit_arima(y: pd.Series, horizon: int,
              max_p: int = 3, max_d: int = 2, max_q: int = 3) -> dict:
    """
    Box-Jenkins ARIMA(p,d,q) selected by AIC grid search.
    Covers AR, MA, ARMA, ARIMA as special cases (p=0, q=0, d=0, etc.).
    """
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_model = None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(y, order=(p, d, q)).fit()
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                        best_model = m
                except Exception:
                    continue

    if best_model is None:
        return _empty_result("ARIMA (failed)")

    fitted = best_model.fittedvalues
    fc_res = best_model.get_forecast(steps=horizon)
    forecast = fc_res.predicted_mean
    forecast.index = _future_index(y.index[-1], horizon)
    ci = fc_res.conf_int(alpha=0.05)
    ci.index = forecast.index
    resid = best_model.resid

    family = "ARIMA"
    p, d, q = best_order
    if d == 0 and q == 0:
        family = f"AR({p})"
    elif d == 0 and p == 0:
        family = f"MA({q})"
    elif d == 0:
        family = f"ARMA({p},{q})"
    else:
        family = f"ARIMA({p},{d},{q})"

    return {
        "name": family,
        "fitted": fitted,
        "forecast": forecast,
        "lower": ci.iloc[:, 0],
        "upper": ci.iloc[:, 1],
        "residuals": resid,
        "params": {"order": best_order, "aic": float(best_aic)},
    }


# --------------------------------------------------------------------------- #
#  3. Machine learning (lagged feature regression)                             #
# --------------------------------------------------------------------------- #
@dataclass
class _LaggedDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def _build_features(y: pd.Series, lags: list[int], windows: list[int]) -> _LaggedDataset:
    """Build a supervised matrix with lags + rolling stats + month-of-year."""
    df = pd.DataFrame({"y": y})
    feats = []
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
        feats.append(f"lag_{L}")
    for w in windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w).std()
        feats.append(f"roll_mean_{w}")
        feats.append(f"roll_std_{w}")
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    feats += ["month_sin", "month_cos"]

    df = df.dropna()
    return _LaggedDataset(
        X=df[feats].values,
        y=df["y"].values,
        feature_names=feats,
    )


def _recursive_forecast(predict_fn, y_hist: pd.Series, horizon: int,
                        lags: list[int], windows: list[int]) -> np.ndarray:
    """Forecast `horizon` steps recursively, feeding predictions back as lags."""
    history = list(y_hist.values)
    months = list(y_hist.index)
    preds = []
    for _ in range(horizon):
        next_month = months[-1] + pd.tseries.frequencies.to_offset("MS")
        feats = []
        for L in lags:
            feats.append(history[-L])
        for w in windows:
            window_vals = history[-w:]
            feats.append(np.mean(window_vals))
            feats.append(np.std(window_vals))
        feats.append(np.sin(2 * np.pi * next_month.month / 12))
        feats.append(np.cos(2 * np.pi * next_month.month / 12))
        x = np.array(feats).reshape(1, -1)
        yhat = float(predict_fn(x)[0])
        preds.append(yhat)
        history.append(yhat)
        months.append(next_month)
    return np.array(preds)


def fit_random_forest(y: pd.Series, horizon: int,
                      lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 132),
                      windows: tuple[int, ...] = (3, 12)) -> dict:
    """Tree-based model — random forest on lagged features."""
    lags = list(lags)
    windows = list(windows)
    ds = _build_features(y, lags, windows)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    rf.fit(ds.X, ds.y)
    fitted = pd.Series(rf.predict(ds.X), index=y.index[-len(ds.y):])

    point = _recursive_forecast(rf.predict, y, horizon, lags, windows)
    forecast = pd.Series(point, index=_future_index(y.index[-1], horizon))

    # Empirical prediction interval from per-tree spread
    per_tree = np.stack([t.predict(ds.X) for t in rf.estimators_])
    sigma = per_tree.std(axis=0).mean()
    resid = (y.iloc[-len(ds.y):] - fitted).dropna()
    sigma_resid = resid.std()
    band_sigma = np.sqrt(sigma ** 2 + sigma_resid ** 2)
    h = np.arange(1, horizon + 1)
    band = 1.96 * band_sigma * np.sqrt(h)

    return {
        "name": "Random Forest",
        "fitted": fitted,
        "forecast": forecast,
        "lower": pd.Series(forecast.values - band, index=forecast.index),
        "upper": pd.Series(forecast.values + band, index=forecast.index),
        "residuals": resid,
        "params": {
            "n_estimators": 300, "lags": lags, "windows": windows,
            "feature_importance": dict(zip(ds.feature_names, rf.feature_importances_)),
        },
    }


def fit_neural_network(y: pd.Series, horizon: int,
                       lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 132),
                       windows: tuple[int, ...] = (3, 12)) -> dict:
    """Neural network — multilayer perceptron on lagged features."""
    lags = list(lags)
    windows = list(windows)
    ds = _build_features(y, lags, windows)
    scaler = StandardScaler().fit(ds.X)
    X_s = scaler.transform(ds.X)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    mlp.fit(X_s, ds.y)
    fitted = pd.Series(mlp.predict(X_s), index=y.index[-len(ds.y):])

    def predict_fn(x):
        return mlp.predict(scaler.transform(x))

    point = _recursive_forecast(predict_fn, y, horizon, lags, windows)
    forecast = pd.Series(point, index=_future_index(y.index[-1], horizon))

    resid = (y.iloc[-len(ds.y):] - fitted).dropna()
    sigma = resid.std()
    h = np.arange(1, horizon + 1)
    band = 1.96 * sigma * np.sqrt(h)

    return {
        "name": "Neural Network (MLP)",
        "fitted": fitted,
        "forecast": forecast,
        "lower": pd.Series(forecast.values - band, index=forecast.index),
        "upper": pd.Series(forecast.values + band, index=forecast.index),
        "residuals": resid,
        "params": {"hidden_layers": (64, 32), "lags": lags, "windows": windows,
                   "n_iter": int(mlp.n_iter_)},
    }


# --------------------------------------------------------------------------- #
#  Master runner                                                               #
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "Simple Exponential Smoothing": fit_simple_exp,
    "Holt":                         fit_holt,
    "Holt-Winters":                 fit_holt_winters,
    "ARIMA (auto)":                 fit_arima,
    "Random Forest":                fit_random_forest,
    "Neural Network (MLP)":         fit_neural_network,
}


def run_all(y: pd.Series, horizon: int, selected: list[str] | None = None) -> dict:
    """Fit every selected model and return a {name: result_dict} mapping."""
    if selected is None:
        selected = list(MODEL_REGISTRY)
    out = {}
    for name in selected:
        try:
            out[name] = MODEL_REGISTRY[name](y, horizon)
        except Exception as e:  # noqa: BLE001
            out[name] = {**_empty_result(name), "error": str(e)}
    return out
