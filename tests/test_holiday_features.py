import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe import run_feature_engineering


def test_holiday_features():
    """run_feature_engineering should mark holidays correctly."""

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)
            .tz_convert("Asia/Seoul"),
            "y": [0, 0],
        }
    )
    cfg = {
        "features": {
            "use_holidays": True,
            "lags": [],
            "rollings": [],
            "fourier": {"weekly_K": 0, "yearly_K": 0},
            "intermittency": {"enable": False},
        }
    }
    schema = {"date": "date", "target": "y", "series": []}

    result, _ = run_feature_engineering(df, cfg, schema)

    assert result["date"].dt.tz.zone == "Asia/Seoul"

    assert result["is_holiday"].tolist() == [1, 0]
    assert result["holiday_name"].astype(str).tolist() == ["신정", "None"]
