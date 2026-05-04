"""
Data loader for the GFZ Potsdam combined space-weather file.

Source (official): https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt
This file is updated daily by the GFZ German Research Centre for Geosciences.

It contains, for every day since 1932-01-01:
    - 8 Kp sub-indices (3-hourly)         -> not used here
    - 8 ap sub-indices (3-hourly)         -> not used here
    - Ap   : daily planetary index (mean)
    - SN   : international sunspot number
    - F10.7obs : observed 10.7 cm solar flux (sfu)
    - F10.7adj : flux adjusted to 1 AU

We keep the three monitored series: SN, F10.7obs, Ap.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
LOCAL_CACHE = Path(__file__).parent / "data" / "space_weather_daily.csv"


# Column layout of the GFZ flat file (positions confirmed against the header).
# The file is whitespace-separated; we just read the columns we need by index.
GFZ_COLUMNS = [
    "year", "month", "day",
    "days", "days_m", "Bsr", "dB",
    "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
    "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
    "Ap", "SN", "F107obs", "F107adj", "D",
]


def _parse_gfz_text(text: str) -> pd.DataFrame:
    """Parse the GFZ fixed-width-ish file into a clean daily DataFrame."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
        header=None,
        names=GFZ_COLUMNS,
        engine="python",
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("date")[["SN", "F107obs", "Ap"]]
    df.columns = ["SN", "F107", "Ap"]

    # The GFZ file uses sentinel values for missing data
    df = df.replace(-1, np.nan).replace(-1.0, np.nan)
    # F10.7 was not measured before Feb 1947 -> drop those rows for that column
    df.loc[df.index < "1947-02-14", "F107"] = np.nan
    return df


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching data from GFZ Potsdam…")
def load_daily(use_remote: bool = True) -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with columns SN, F107, Ap."""
    if use_remote:
        try:
            resp = requests.get(GFZ_URL, timeout=30)
            resp.raise_for_status()
            df = _parse_gfz_text(resp.text)
            # Keep a local cached copy so the app survives a future outage
            try:
                LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(LOCAL_CACHE)
            except Exception:
                pass
            return df
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not reach GFZ Potsdam ({e}). Using local cache.")

    if LOCAL_CACHE.exists():
        return pd.read_csv(LOCAL_CACHE, index_col=0, parse_dates=True)

    raise RuntimeError(
        "No data available: remote fetch failed and no local cache exists. "
        "Run `python data_loader.py` once with internet access to populate the cache."
    )


@st.cache_data(ttl=24 * 3600)
def load_monthly() -> pd.DataFrame:
    """Monthly means — used by the dashboard for cleaner modelling."""
    daily = load_daily()
    monthly = daily.resample("MS").mean()
    # Drop very early years where F10.7 is NaN to keep all three series aligned
    monthly = monthly.dropna()
    return monthly


SERIES_META = {
    "SN":   {"label": "Sunspot Number",        "unit": "count",
             "desc": "International Sunspot Number — proxy for solar activity. "
                     "Decision-relevant for: long-range planning of satellite missions."},
    "F107": {"label": "F10.7 cm Solar Flux",   "unit": "sfu",
             "desc": "10.7 cm radio flux. Drives thermospheric density and therefore "
                     "satellite drag. Decision-relevant for: orbit maintenance, re-entry "
                     "predictions, collision avoidance."},
    "Ap":   {"label": "Planetary Ap index",    "unit": "nT (×2)",
             "desc": "Daily mean planetary geomagnetic activity. "
                     "Decision-relevant for: power-grid GIC alerts, polar flight routing, "
                     "HF radio blackout warnings, GPS accuracy degradation."},
}


if __name__ == "__main__":
    print("Downloading GFZ Potsdam space-weather file…")
    df = load_daily(use_remote=True)
    print(df.tail())
    print(f"Saved {len(df):,} daily rows to {LOCAL_CACHE}")
