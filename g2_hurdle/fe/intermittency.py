
import pandas as pd
import numpy as np

def create_intermittency_features(df: pd.DataFrame, target_col: str, series_cols):
    out = df.copy()

    def _apply(group):
        y = group[target_col]
        # days since last sale
        # approach: cumulative count reset where sale>0
        mask_pos = y > 0
        # index of last positive
        last = None
        dsls = []
        for i, v in enumerate(mask_pos.values):
            if v:
                last = i
                dsls.append(0)
            else:
                dsls.append((i - last) if last is not None else 0)
        group["days_since_last_sale"] = dsls

        # rolling zero count 7d
        zero_flag = (y==0).astype(int)
        group["rolling_zero_count_7d"] = zero_flag.shift(1).rolling(window=7, min_periods=1).sum()

        # average interdemand interval (rolling mean of dsls)
        dsls_series = group["days_since_last_sale"]
        group["avg_interdemand_interval"] = dsls_series.shift(1).rolling(window=28, min_periods=1).mean()
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


def update_intermittency_features(ctx_tail: pd.DataFrame, new_y: float) -> pd.DataFrame:
    """Update intermittency-related features for the next step."""

    ctx = ctx_tail.copy()

    # infer target column similar to lag/rolling update
    num_cols = ctx.select_dtypes(include="number").columns
    target_candidates = [
        c
        for c in num_cols
        if c
        not in {
            "days_since_last_sale",
            "rolling_zero_count_7d",
            "avg_interdemand_interval",
        }
    ]
    target_col = target_candidates[0] if target_candidates else None

    # historical columns exclude the last placeholder row
    history_y = ctx[target_col].iloc[:-1] if target_col else pd.Series(dtype=float)

    prev_dsls = ctx.iloc[-2]["days_since_last_sale"] if len(ctx) >= 2 else 0
    dsls_next = 0 if new_y > 0 else prev_dsls + 1

    zero_flags = (history_y == 0).astype(int)
    rolling_zero = zero_flags.tail(7).sum()
    avg_idi = ctx["days_since_last_sale"].iloc[:-1].tail(28).mean()

    ctx.loc[ctx.index[-1], "days_since_last_sale"] = dsls_next
    ctx.loc[ctx.index[-1], "rolling_zero_count_7d"] = rolling_zero
    ctx.loc[ctx.index[-1], "avg_interdemand_interval"] = avg_idi
    ctx.fillna(0, inplace=True)
    return ctx
