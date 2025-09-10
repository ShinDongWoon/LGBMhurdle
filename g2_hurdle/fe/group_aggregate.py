import pandas as pd


def create_group_aggregate_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add store-level and menu-level aggregate features.

    For each of ``store_id`` and ``menu_id`` present in ``df``, compute the
    mean and standard deviation of the target column across the entire history
    and merge them back onto the dataframe.  Missing values are filled with 0
    so the resulting features are numeric and ready for modelling.
    """
    out = df.copy()
    added_cols = []
    for col in ("store_id", "menu_id"):
        if col in out.columns:
            stats = (
                out.groupby(col, observed=False)[target_col]
                .agg(["mean", "std"])
                .rename(columns={"mean": f"{col}_avg_sales", "std": f"{col}_volatility"})
            )
            out = out.merge(stats, how="left", left_on=col, right_index=True)
            added_cols.extend(stats.columns)
    if added_cols:
        out[added_cols] = out[added_cols].fillna(0)
    return out
