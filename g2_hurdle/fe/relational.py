import pandas as pd
import numpy as np


def demand_rank_in_store(
    df: pd.DataFrame, store_col: str, date_col: str, lag_col: str = "lag_1"
) -> pd.Series:
    """Rank items within each store by previous-day demand.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe containing ``lag_col``.
    store_col : str
        Column denoting the store identifier.
    date_col : str
        Column with the date information.
    lag_col : str, default "lag_1"
        Column representing previous day's demand for the item.

    Returns
    -------
    pd.Series
        Dense rank of ``lag_col`` within each ``store_col`` on ``date_col``.
    """
    if not {store_col, date_col, lag_col}.issubset(df.columns):
        return pd.Series(np.zeros(len(df)), index=df.index)
    ranks = df.groupby([store_col, date_col])[lag_col].rank(
        method="dense", ascending=False
    )
    return ranks.fillna(0)


def demand_ratio_in_store(
    df: pd.DataFrame, store_col: str, date_col: str, lag_col: str = "lag_1"
) -> pd.Series:
    """Compute ratio of an item's demand to total store demand.

    Uses ``lag_col`` to avoid leakage. For each store and date, the feature is
    defined as ``lag_col`` divided by the sum of ``lag_col`` across all menu
    items in the same store.
    """
    if not {store_col, date_col, lag_col}.issubset(df.columns):
        return pd.Series(np.zeros(len(df)), index=df.index)
    store_sum = df.groupby([store_col, date_col])[lag_col].transform("sum")
    ratio = df[lag_col] / store_sum.replace(0, np.nan)
    return ratio.fillna(0)


def price_rank_in_store(
    df: pd.DataFrame, store_col: str, price_col: str = "price"
) -> pd.Series:
    """Rank menu items within a store by their price.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``price_col``.
    store_col : str
        Column denoting the store identifier.
    price_col : str, default "price"
        Column with the static price information for each item.
    """
    if not {store_col, price_col}.issubset(df.columns):
        return pd.Series(np.zeros(len(df)), index=df.index)
    ranks = df.groupby(store_col)[price_col].rank(method="dense", ascending=False)
    return ranks.fillna(0)


def create_relational_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Create relational features based on store-level interactions.

    The following features are added when the necessary columns are present:

    - ``demand_rank_in_store``: rank of an item's previous-day demand within its
      store for that date.
    - ``demand_ratio_in_store``: ratio of an item's previous-day demand to the
      total demand of the store for that date.
    - ``price_rank_in_store``: rank of an item's price within its store.
    """
    out = df.copy()
    date_col = schema.get("date")
    if date_col and "store_id" in out.columns:
        out["demand_rank_in_store"] = demand_rank_in_store(out, "store_id", date_col)
        out["demand_ratio_in_store"] = demand_ratio_in_store(out, "store_id", date_col)
        out["price_rank_in_store"] = price_rank_in_store(out, "store_id", "price")
    else:
        out["demand_rank_in_store"] = 0
        out["demand_ratio_in_store"] = 0
        out["price_rank_in_store"] = 0
    return out
