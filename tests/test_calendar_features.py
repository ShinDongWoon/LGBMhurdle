import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe.calendar import create_calendar_features


def test_calendar_features_week_and_weekend():
    """create_calendar_features should add week and weekend indicators."""

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=7, freq="D", tz="Asia/Seoul")
    })
    result = create_calendar_features(df, "date")

    assert "week" in result.columns
    assert "is_weekend" in result.columns

    # ISO calendar week 1 for all dates in the range
    assert result["week"].tolist() == [1] * 7
    # Weekend indicator: Saturday and Sunday only
    assert result["is_weekend"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert result["date"].dt.tz.zone == "Asia/Seoul"


def test_calendar_features_timezone_boundary():
    """Weekend/week numbers should respect Asia/Seoul near UTC boundaries."""

    utc_dates = pd.to_datetime(["2024-03-31 14:30", "2024-03-31 15:30"], utc=True)
    df = pd.DataFrame({"date": utc_dates.tz_convert("Asia/Seoul")})
    result = create_calendar_features(df, "date")

    assert result["date"].dt.tz.zone == "Asia/Seoul"
    assert result["is_weekend"].tolist() == [1, 0]
    assert result["week"].tolist() == [13, 14]

