
import pandas as pd
import numpy as np

def create_lags_and_rolling_features(df: pd.DataFrame, target_col: str, series_cols, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    lags = cfg.get("features", {}).get("lags", [1,2,7,14,28,365])
    rolls = cfg.get("features", {}).get("rollings", [7,14,28])

    if series_cols:
        g = out.groupby(series_cols, group_keys=False, observed=False)
    else:
        # treat whole df as one group
        g = [(None, out)]

    frames = []
    if isinstance(g, list):
        iterable = g
    else:
        iterable = g

    # Implement with groupby apply to ensure shift(1) before rolling
    def _apply(group):
        s = group[target_col]
        for lag in lags:
            group[f"lag_{lag}"] = s.shift(lag)
        s_shift = s.shift(1)  # leakage guard
        for w in rolls:
            r = s_shift.rolling(window=w)
            group[f"roll_mean_{w}"] = r.mean()
            group[f"roll_std_{w}"] = r.std()
            group[f"roll_min_{w}"] = r.min()
            group[f"roll_max_{w}"] = r.max()
        return group

    if series_cols:
        out = out.groupby(series_cols, group_keys=False, observed=False).apply(_apply)
    else:
        out = _apply(out)

    return out
