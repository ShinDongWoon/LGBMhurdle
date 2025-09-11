
import pandas as pd
from .calendar import create_calendar_features
from .fourier import create_fourier_features
from .holiday import create_holiday_features
from .lags_rolling import create_lags_and_rolling_features
from .intermittency import create_intermittency_features
from .preprocess import prepare_features


def run_feature_engineering(
    df: pd.DataFrame,
    cfg: dict,
    schema: dict,
):
    date_col = schema["date"]
    target_col = schema["target"]
    series_cols = schema["series"]
    out = df.copy()
    out = create_calendar_features(out, date_col)
    if cfg.get("features", {}).get("use_holidays"):
        out = create_holiday_features(out, date_col)
    else:
        out["is_holiday"] = 0
        out["holiday_name"] = pd.Series(["None"] * len(out), dtype="category")
    out = create_fourier_features(out, date_col, cfg)
    # Group aggregate features are computed inside each CV fold using only
    # the training portion of the data to avoid leakage.
    out = create_lags_and_rolling_features(out, target_col, series_cols, cfg)
    if cfg.get("features", {}).get("intermittency", {}).get("enable", True):
        out = create_intermittency_features(out, target_col, series_cols)
    return out, {}
