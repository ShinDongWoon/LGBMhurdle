import pandas as pd
import holidayskr
from g2_hurdle.fe import run_feature_engineering


def test_holiday_features(monkeypatch):
    """run_feature_engineering should mark holidays correctly."""

    original_is_holiday = holidayskr.is_holiday

    def _patched(date):
        if not isinstance(date, str):
            date = date.strftime("%Y-%m-%d")
        return original_is_holiday(date)

    monkeypatch.setattr(holidayskr, "is_holiday", _patched)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
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

    result = run_feature_engineering(df, cfg, schema)

    assert result["is_holiday"].tolist() == [1, 0]
    assert result["holiday_name"].astype(str).tolist() == ["신정", "None"]
