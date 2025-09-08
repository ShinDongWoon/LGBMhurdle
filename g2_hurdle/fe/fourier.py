
import numpy as np
import pandas as pd

def _fourier(series, period, K, prefix):
    t = np.arange(len(series), dtype=float)
    out = {}
    for k in range(1, K+1):
        out[f"{prefix}_sin_{k}"] = np.sin(2*np.pi*k*t/period)
        out[f"{prefix}_cos_{k}"] = np.cos(2*np.pi*k*t/period)
    return pd.DataFrame(out, index=series.index)

def create_fourier_features(df: pd.DataFrame, date_col: str, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    fcfg = cfg.get("features", {}).get("fourier", {})
    K_w = int(fcfg.get("weekly_K", 3))
    K_y = int(fcfg.get("yearly_K", 10))
    # use sequential index
    fw = _fourier(out[date_col], period=7.0, K=K_w, prefix="fw")
    fy = _fourier(out[date_col], period=365.25, K=K_y, prefix="fy")
    out = pd.concat([out, fw, fy], axis=1)
    return out
