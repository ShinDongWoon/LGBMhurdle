import json

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.pipeline.train import run_train
import g2_hurdle.pipeline.predict as predict_module
from g2_hurdle.pipeline.predict import run_predict
from g2_hurdle.utils.keys import normalize_series_name


def test_pipeline_produces_dow_and_weekend(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    dates = pd.date_range("2024-01-04", periods=5, freq="D", tz="Asia/Seoul")
    train_df = pd.DataFrame(
        {
            "영업일자": dates,
            "영업장명_메뉴명": ["매장A_메뉴1"] * len(dates),
            "매출수량": [0, 1, 0, 2, 0],
        }
    )
    train_df.to_csv(train_path, index=False)

    artifacts_dir = tmp_path / "artifacts"
    cfg_train = {
        "paths": {"train_csv": str(train_path)},
        "io": {"artifacts_dir": str(artifacts_dir)},
        "cv": {
            "horizon": 1,
            "init_train_ratio": 0.8,
            "early_stopping_rounds": 0,
            "min_positive_samples": 0,
            "min_negative_samples": 0,
        },
        "runtime": {"n_jobs": 1, "seed": 0, "use_gpu": False},
        "model": {
            "classifier": {"n_estimators": 1},
            "regressor": {"n_estimators": 1},
        },
        "features": {
            "lags": [1],
            "rollings": [2],
            "use_holidays": False,
            "intermittency": {"enable": False},
        },
        "data": {
            "date_col": "영업일자",
            "target_col": "매출수량",
            "id_col_candidates": ["store_menu_id"],
            "min_positive_ratio": 0.0,
        },
    }

    run_train(cfg_train)

    with open(artifacts_dir / "features.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "dow" in meta["categorical_cols"]
    assert "is_weekend" in meta["feature_cols"]
    assert set(meta["categories"]["dow"]) == set(range(7)) | {"missing"}

    test_dir = tmp_path / "test"
    test_dir.mkdir()
    test_dates = pd.date_range("2024-01-09", periods=7, freq="D", tz="Asia/Seoul")
    test_df = pd.DataFrame(
        {
            "영업일자": test_dates,
            "영업장명_메뉴명": ["매장A_메뉴1"] * len(test_dates),
            "매출수량": [0] * len(test_dates),
        }
    )
    test_df.to_csv(test_dir / "TEST_001.csv", index=False)

    sample_df = pd.DataFrame(
        {"id": [normalize_series_name("매장A_메뉴1")], "D1": [0]}
    )
    sample_path = tmp_path / "sample.csv"
    sample_df.to_csv(sample_path, index=False)
    out_path = tmp_path / "out.csv"

    cfg_pred = {
        "paths": {
            "test_dir": str(test_dir),
            "sample_submission": str(sample_path),
            "out_path": str(out_path),
            "artifacts_dir": str(artifacts_dir),
        },
        "data": {"min_context_days": 1},
        "cv": {"horizon": 1},
    }

    captured = {}
    orig_prepare = predict_module.prepare_features

    def capture_prepare(fe, *args, **kwargs):
        captured["fe"] = fe.copy()
        X, fcols, ccols = orig_prepare(fe, *args, **kwargs)
        captured["X"] = X
        return X, fcols, ccols

    monkeypatch.setattr(predict_module, "prepare_features", capture_prepare)
    run_predict(cfg_pred)

    assert captured, "prepare_features was not called"
    fe_df = captured["fe"]
    X_pred = captured["X"]
    assert fe_df["영업일자"].dt.tz.zone == "Asia/Seoul"
    assert "dow" in X_pred.columns
    assert "is_weekend" in X_pred.columns
