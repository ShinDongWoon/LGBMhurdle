
import pandas as pd
import numpy as np
from ..fe import run_feature_engineering

def _predict_one_step(df_future_row, clf, reg, threshold):
    obj_cols = df_future_row.select_dtypes(include="object").columns
    for c in obj_cols:
        df_future_row[c] = df_future_row[c].astype("category")
    p = clf.predict_proba(df_future_row)
    q = reg.predict(df_future_row)
    yhat = (p > threshold).astype(float) * np.maximum(0.0, q)
    return float(yhat[0]), float(p[0]), float(q[0])

def recursive_forecast_grouped(context_df: pd.DataFrame, schema: dict, cfg: dict, clf, reg, threshold: float, horizon: int=7):
    """Run recursive forecast per series group (identified by schema['series']).
    context_df: must contain at least the last 28 days per series.
    Returns DataFrame with columns: id, D1..Dh and optionally stacks of p,q for analysis.
    """
    date_col = schema["date"]
    target_col = schema["target"]
    series_cols = schema["series"]
    H = int(cfg.get("cv", {}).get("horizon", horizon))

    # Ensure series id
    from ..utils.keys import build_series_id, ensure_wide_columns
    id_series = build_series_id(context_df, series_cols)
    context_df = context_df.copy()
    context_df["_series_id"] = id_series

    out_rows = []
    for sid, g in context_df.groupby("_series_id"):
        g = g.sort_values(date_col).copy()
        # Keep a mutable context for this series
        ctx = g.copy()
        preds, probs, qtys = [], [], []
        last_date = ctx[date_col].max()
        for h in range(1, H+1):
            future_date = last_date + pd.Timedelta(days=h)
            new_row = ctx.iloc[-1:].copy()
            new_row[date_col] = future_date
            new_row[target_col] = np.nan
            ctx_ext = pd.concat([ctx, new_row], ignore_index=True)
            # Recompute features for the extended context
            fe_ctx = run_feature_engineering(ctx_ext, cfg, schema)
            # Use the last row as the feature row
            X_row = fe_ctx.iloc[[-1]].drop(columns=[date_col, target_col], errors="ignore")
            yhat, p, q = _predict_one_step(X_row, clf, reg, threshold)
            preds.append(yhat); probs.append(p); qtys.append(q)
            # inject prediction as if observed for next step
            new_row[target_col] = yhat
            ctx = pd.concat([ctx.iloc[:-1], new_row], ignore_index=True)

        row = {"id": sid}
        for i, v in enumerate(preds, start=1):
            row[f"D{i}"] = v
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    # Ensure D1..Dh exist
    need_cols = ensure_wide_columns(H)
    for c in need_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[["id", *need_cols]]
