
from typing import Optional

import numpy as np
import pandas as pd


def _fourier(series, period, K, prefix, origin: Optional[pd.Timestamp] = None):
    """Create Fourier series features for a date series.

    Parameters
    ----------
    series : pd.Series
        Series of datetimes.
    period : float
        Period of the Fourier features (e.g. 7 for weekly).
    K : int
        Number of Fourier terms to generate.
    prefix : str
        Prefix for column names.
    origin : pd.Timestamp, optional
        Reference date from which to compute day differences. If ``None``,
        the minimum of ``series`` is used.
    """

    series = series.dt.floor("D")
    if origin is None:
        origin = series.min()
    t = (series - origin).dt.days.astype(float)
    out = {}
    for k in range(1, K + 1):
        out[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(out, index=series.index)


def create_fourier_features(df: pd.DataFrame, date_col: str, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    fcfg = cfg.get("features", {}).get("fourier", {})
    K_w = int(fcfg.get("weekly_K", 3))
    K_y = int(fcfg.get("yearly_K", 10))

    # Align to the global minimum date so that identical dates share features
    origin = out[date_col].dt.floor("D").min()
    fw = _fourier(out[date_col], period=7.0, K=K_w, prefix="fw", origin=origin)
    fy = _fourier(out[date_col], period=365.25, K=K_y, prefix="fy", origin=origin)
    out = pd.concat([out, fw, fy], axis=1)
    return out
