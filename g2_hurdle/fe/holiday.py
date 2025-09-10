import pandas as pd
import holidayskr


def create_holiday_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add Korean holiday features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a date column.
    date_col : str
        Name of the datetime column in ``df``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``is_holiday`` and ``holiday_name`` columns added.
    """
    out = df.copy()
    d = out[date_col]
    d_str = d.dt.strftime("%Y-%m-%d")
    years = d.dt.year.unique()
    holiday_map = {}
    for year in years:
        for day, name in holidayskr.year_holidays(str(year)):
            holiday_map[day.strftime("%Y-%m-%d")] = name
    out["is_holiday"] = d_str.isin(holiday_map).astype(int)
    out["holiday_name"] = (
        d_str.map(holiday_map).fillna("None").astype("category")
    )
    return out
