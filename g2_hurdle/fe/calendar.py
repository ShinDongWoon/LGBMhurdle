
import pandas as pd
import numpy as np

def create_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    d = out[date_col]
    out["dow"] = d.dt.weekday
    out["week"] = d.dt.isocalendar().week.astype(int)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["year"] = d.dt.year
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_weekend"] = d.dt.weekday.isin([5,6]).astype(int)
    return out
