import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.utils.preprocessing import clip_negative_values
import g2_hurdle.pipeline.train as train_module


def test_clip_negative_values_basic():
    df = pd.DataFrame({"a": [-1, 2], "b": [3, -4]})
    res = clip_negative_values(df.copy(), ["a", "b"])
    assert (res[["a", "b"]] >= 0).all().all()


def test_run_train_clips_target(tmp_path, monkeypatch):
    data = {
        "date": pd.date_range("2021-01-01", periods=10, freq="D"),
        "store_id": ["A"] * 10,
        "sales": [-1, 2, -3, 4, 5, 6, 7, 8, 9, 10],
        "feat1": np.arange(10),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "paths": {"train_csv": str(csv_path)},
        "io": {"artifacts_dir": str(tmp_path / "artifacts")},
        "data": {
            "date_col_candidates": ["date"],
            "target_col_candidates": ["sales"],
            "id_col_candidates": ["store_id"],
            "non_negative_cols": ["sales"],
            "min_positive_ratio": 0.0,
        },
        "cv": {
            "horizon": 1,
            "init_train_ratio": 0.8,
            "min_positive_samples": 0,
            "min_negative_samples": 0,
        },
        "runtime": {"seed": 42, "n_jobs": 1, "use_gpu": False},
    }

    class DummyReg:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
            assert (y >= 0).all()
            self.feature_names_ = list(X.columns)

    class DummyClf:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None):
            self.feature_names_ = list(X.columns)

        def predict_proba(self, X):
            return np.zeros(len(X))

    def fake_run_fe(df, cfg, schema):
        return df

    def fake_prepare_features(df, drop_cols, feature_cols=None, categorical_cols=None):
        X = df.drop(columns=drop_cols)
        return X, X.columns.tolist(), []

    def fake_split(df, date_col, H, init_ratio):
        n = len(df)
        tr_mask = df.index < n - 1
        va_mask = ~tr_mask
        tr_end = df.loc[tr_mask, date_col].max()
        va_start = df.loc[va_mask, date_col].min()
        va_end = df.loc[va_mask, date_col].max()
        yield tr_mask, va_mask, (tr_end, va_start, va_end)

    def fake_recursive_forecast_grouped(context, schema, cfg, clf, reg, threshold, horizon, feature_cols, categorical_cols):
        ids = context["id"].unique()
        out = pd.DataFrame({"id": ids})
        for i in range(1, horizon + 1):
            out[f"D{i}"] = 0.0
        return out

    def fake_find_optimal_threshold(y_true, p, q, cfg):
        return 0.5, 0.0

    monkeypatch.setattr(train_module, "run_feature_engineering", fake_run_fe)
    monkeypatch.setattr(train_module, "prepare_features", fake_prepare_features)
    monkeypatch.setattr(train_module, "rolling_forecast_origin_split", fake_split)
    monkeypatch.setattr(train_module, "HurdleRegressor", DummyReg)
    monkeypatch.setattr(train_module, "HurdleClassifier", DummyClf)
    import types, sys
    fake_recursion = types.ModuleType("g2_hurdle.pipeline.recursion")
    fake_recursion.recursive_forecast_grouped = fake_recursive_forecast_grouped
    sys.modules["g2_hurdle.pipeline.recursion"] = fake_recursion
    import g2_hurdle.model.threshold as threshold_module
    monkeypatch.setattr(threshold_module, "find_optimal_threshold", fake_find_optimal_threshold)
    monkeypatch.setattr(train_module, "save_artifacts", lambda artifacts, path: None)

    train_module.run_train(cfg)
