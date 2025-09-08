
import pandas as pd
import numpy as np

def build_series_id(df: pd.DataFrame, series_cols):
    if not series_cols:
        # Single series – fallback id
        return pd.Series(["__single__"]*len(df), index=df.index)
    parts = []
    for c in series_cols:
        s = df[c].astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
        parts.append(s)
    key = parts[0]
    for s in parts[1:]:
        key = key + "__" + s
    return key

def align_to_submission(sub_df: pd.DataFrame, pred_df: pd.DataFrame, id_col="id"):
    # Keep original columns/rows of submission
    out = sub_df.copy()
    # Identify fillable columns: numeric columns except id
    fill_cols = [c for c in out.columns if c != id_col]
    # Left-join pred on submission id
    merged = out[[id_col]].merge(pred_df, how="left", on=id_col)
    # Fill only NaNs in submission
    for c in fill_cols:
        if c in pred_df.columns:
            mask = out[c].isna()
            out.loc[mask, c] = merged.loc[mask, c]
    return out

def ensure_wide_columns(h=7):
    return [f"D{i}" for i in range(1, h+1)]
