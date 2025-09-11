
import pandas as pd
import numpy as np
import re

def normalize_series_name(s):
    if isinstance(s, pd.Series):
        return s.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    return re.sub(r"\s+", "_", str(s).strip())


def build_series_id(df: pd.DataFrame, series_cols):
    if not series_cols:
        # Single series – fallback id
        return pd.Series(["__single__"]*len(df), index=df.index)
    parts = []
    for c in series_cols:
        parts.append(normalize_series_name(df[c]))
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
    # Fill NaNs or placeholder zeros in submission while preserving dtypes
    for c in fill_cols:
        if c in pred_df.columns:
            pred_col = merged[c]
            if pd.api.types.is_numeric_dtype(out[c]):
                mask = out[c].isna() | (out[c] == 0)
            else:
                mask = out[c].isna()
            out.loc[mask, c] = pred_col.loc[mask].astype(out[c].dtype, copy=False)
    return out

def ensure_wide_columns(h=7):
    return [f"D{i}" for i in range(1, h+1)]
