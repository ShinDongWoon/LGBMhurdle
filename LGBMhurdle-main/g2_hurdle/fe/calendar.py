
import pandas as pd
import numpy as np


def create_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Generate basic calendar features.

    In addition to integer date parts, cyclical representations are added and the
    original discrete components are cast to ``category`` so downstream callers
    can treat them as such (e.g. for LightGBM's categorical_feature).
    """

    out = df.copy()
    d = out[date_col]

    # raw components
    out["dow"] = d.dt.weekday
    out["week"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year

    # cyclical (sin/cos) encodings
    for col, period in [("dow", 7), ("week", 52), ("month", 12), ("quarter", 4)]:
        val = out[col].astype(float)
        out[f"{col}_sin"] = np.sin(2 * np.pi * val / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * val / period)
        out[col] = out[col].astype("category")

    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_weekend"] = d.dt.weekday.isin([5, 6]).astype(int)
    return out
