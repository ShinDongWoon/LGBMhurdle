import pandas as pd

from .calendar import create_calendar_features
from .fourier import create_fourier_features
from .holiday import create_holiday_features


def prepare_static_future_features(df: pd.DataFrame, schema: dict, cfg: dict, horizon: int) -> pd.DataFrame:
    """Pre-compute calendar and Fourier features for all future dates.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data containing at least the date column.
    schema : dict
        Schema specifying column names. Must include ``date``.
    cfg : dict
        Configuration used for Fourier feature generation.
    horizon : int
        Number of days ahead to generate features for.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by future dates with calendar and Fourier features
        for each required horizon step.
    """
    date_col = schema["date"]
    last_date = df[date_col].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({date_col: future_dates})
    out = create_calendar_features(future_df, date_col)
    if cfg.get("features", {}).get("use_holidays"):
        out = create_holiday_features(out, date_col)
    out = create_fourier_features(out, date_col, cfg)
    out = out.set_index(date_col)
    return out
