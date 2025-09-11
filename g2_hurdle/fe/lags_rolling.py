
import pandas as pd
import numpy as np

def create_lags_and_rolling_features(df: pd.DataFrame, target_col: str, series_cols, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    lags = cfg.get("features", {}).get("lags", [1, 7, 28, 365])
    rolls = cfg.get("features", {}).get("rollings", [7, 14, 28])

    if series_cols:
        g = out.groupby(series_cols, group_keys=False, observed=False, sort=False)
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
            if lag in (2, 14):
                continue
            group[f"lag_{lag}"] = s.shift(lag)
        s_shift = s.shift(1)  # leakage guard
        for w in rolls:
            r = s_shift.rolling(window=w, min_periods=1)
            if w not in (7, 14):
                group[f"roll_mean_{w}"] = r.mean()
            group[f"roll_std_{w}"] = r.std()
            group[f"roll_min_{w}"] = r.min()
            group[f"roll_max_{w}"] = r.max()
        for c in group.select_dtypes(include="category").columns:
            if 0 not in group[c].cat.categories:
                group[c] = group[c].cat.add_categories([0])
        group.fillna(0, inplace=True)
        return group

    if series_cols:
        out = out.groupby(
            series_cols, group_keys=False, observed=False, sort=False
        ).apply(_apply)
        out = out.sort_index()
    else:
        out = _apply(out)

    return out


def update_lags_and_rollings(ctx_tail: pd.DataFrame, new_y: float, cfg: dict) -> pd.DataFrame:
    """Update lag and rolling statistics for the next step.

    Parameters
    ----------
    ctx_tail : pd.DataFrame
        DataFrame containing only the most recent rows required to compute
        lag/rolling features. The last row should correspond to the latest
        known observation.
    new_y : float
        Newly observed (or predicted) target value to append to the history.
    cfg : dict
        Configuration dictionary containing ``features.lags`` and
        ``features.rollings`` entries.

    Returns
    -------
    pd.DataFrame
        Updated tail dataframe whose last row holds the features for the
        next horizon step.
    """

    ctx = ctx_tail.copy()
    lags = cfg.get("features", {}).get("lags", [1, 7, 28, 365])
    rolls = cfg.get("features", {}).get("rollings", [7, 14, 28])

    # infer target column: numeric column that is not a lag/rolling feature
    num_cols = ctx.select_dtypes(include="number").columns
    target_candidates = [
        c for c in num_cols if not c.startswith("lag_") and not c.startswith("roll_")
    ]
    if not target_candidates:
        raise ValueError("Could not infer target column for lag/rolling update")
    target_col = target_candidates[0]

    # set the newest observation on the last row
    ctx.loc[ctx.index[-1], target_col] = new_y

    # base new row on the last row to keep static columns
    new_row = ctx.tail(1).copy()

    # compute lag features for the next step
    for lag in lags:
        if lag in (2, 14):
            continue
        if lag == 1:
            new_row[f"lag_{lag}"] = new_y
        else:
            prev_col = f"lag_{lag-1}"
            if prev_col in ctx.columns:
                new_row[f"lag_{lag}"] = ctx.iloc[-1][prev_col]
            else:
                new_row[f"lag_{lag}"] = ctx[target_col].iloc[-lag]

    # compute rolling statistics using the updated target history
    y_series = ctx[target_col]
    for w in rolls:
        hist = y_series.tail(w)
        if w not in (7, 14):
            new_row[f"roll_mean_{w}"] = hist.mean()
        new_row[f"roll_std_{w}"] = hist.std()
        new_row[f"roll_min_{w}"] = hist.min()
        new_row[f"roll_max_{w}"] = hist.max()

    # future target is unknown
    new_row[target_col] = np.nan

    # append and keep only the necessary tail
    ctx = pd.concat([ctx, new_row], ignore_index=True)
    max_lag = max(lags) if lags else 1
    max_roll = max(rolls) if rolls else 1
    tail_len = max(max_lag, max_roll) + 1
    ctx = ctx.tail(tail_len).reset_index(drop=True)
    ctx.fillna(0, inplace=True)
    return ctx
