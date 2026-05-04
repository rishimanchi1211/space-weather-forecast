"""
Residual diagnostics and forecast-accuracy metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


# --------------------------------------------------------------------------- #
#  Accuracy metrics                                                            #
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy metrics on the *aligned* portion of two series."""
    df = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1).dropna()
    if len(df) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE %": np.nan, "sMAPE %": np.nan, "n": 0}
    return {
        "MAE":     mae(df["y"], df["yhat"]),
        "RMSE":    rmse(df["y"], df["yhat"]),
        "MAPE %":  mape(df["y"], df["yhat"]),
        "sMAPE %": smape(df["y"], df["yhat"]),
        "n":       int(len(df)),
    }


# --------------------------------------------------------------------------- #
#  Residual diagnostics                                                        #
# --------------------------------------------------------------------------- #
def residual_acf(residuals: pd.Series, n_lags: int = 24) -> pd.DataFrame:
    """ACF of residuals with 95% bounds."""
    r = residuals.dropna()
    vals = acf(r, nlags=n_lags, fft=True)
    n = len(r)
    bound = 1.96 / np.sqrt(n)
    return pd.DataFrame({
        "lag": np.arange(len(vals)),
        "acf": vals,
        "lower": -bound,
        "upper":  bound,
    })


def ljung_box(residuals: pd.Series, lags: int = 12) -> dict:
    """Ljung-Box test for autocorrelation."""
    r = residuals.dropna()
    if len(r) < lags + 5:
        return {"stat": np.nan, "p_value": np.nan, "white_noise": None}
    res = acorr_ljungbox(r, lags=[lags], return_df=True)
    return {
        "stat": float(res["lb_stat"].iloc[0]),
        "p_value": float(res["lb_pvalue"].iloc[0]),
        "white_noise": bool(res["lb_pvalue"].iloc[0] > 0.05),
    }


def normality(residuals: pd.Series) -> dict:
    """Jarque-Bera and basic moments."""
    r = residuals.dropna()
    if len(r) < 8:
        return {"jb_stat": np.nan, "jb_p": np.nan,
                "skew": np.nan, "kurtosis": np.nan, "normal": None}
    jb = stats.jarque_bera(r)
    return {
        "jb_stat": float(jb.statistic),
        "jb_p":    float(jb.pvalue),
        "skew":    float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "normal":  bool(jb.pvalue > 0.05),
    }
