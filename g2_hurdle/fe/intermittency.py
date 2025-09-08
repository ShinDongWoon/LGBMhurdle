
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
                dsls.append((i - last) if last is not None else np.nan)
        group["days_since_last_sale"] = dsls

        # rolling zero count 7d
        zero_flag = (y==0).astype(int)
        group["rolling_zero_count_7d"] = zero_flag.shift(1).rolling(window=7).sum()

        # average interdemand interval (rolling mean of dsls)
        dsls_series = group["days_since_last_sale"]
        group["avg_interdemand_interval"] = dsls_series.shift(1).rolling(window=28, min_periods=3).mean()
        return group

    if series_cols:
        out = out.groupby(series_cols, group_keys=False).apply(_apply)
    else:
        out = _apply(out)
    return out
