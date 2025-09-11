import pandas as pd
from g2_hurdle.fe import run_feature_engineering

def test_run_feature_engineering_adds_target_encoding():
    df = pd.DataFrame({
        "d": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]),
        "y": [1, 0, 2, 1],
        "store_id": ["s1", "s1", "s2", "s2"],
        "menu_id": ["m1", "m2", "m1", "m3"],
    })
    cfg = {
        "features": {
            "use_holidays": False,
            "lags": [],
            "rollings": [],
            "fourier": {"weekly_K": 0, "yearly_K": 0},
            "intermittency": {"enable": False},
            "target_encoding": {"smoothing": 0},
        }
    }
    schema = {"date": "d", "target": "y", "series": ["store_id", "menu_id"]}
    result, mapping = run_feature_engineering(df, cfg, schema)
    assert "store_id_te_mean" in result.columns
    assert "menu_id_te_mean" in result.columns
    assert "store_id" in mapping
    assert "menu_id" in mapping
