import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe.lags_rolling import (
    create_lags_and_rolling_features,
    update_lags_and_rollings,
)
from g2_hurdle.pipeline.recursion import _build_dynamic_row, _compute_dynamic_features


def test_roll_min_max_creation():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
    cfg = {"features": {"rollings": [3], "lags": []}}
    out = create_lags_and_rolling_features(df, "y", [], cfg)
    assert "roll_min_3" in out.columns
    assert "roll_max_3" in out.columns
    s_shift = df["y"].shift(1)
    expected_min = s_shift.rolling(3, min_periods=1).min().fillna(0).tolist()
    expected_max = s_shift.rolling(3, min_periods=1).max().fillna(0).tolist()
    assert out["roll_min_3"].tolist() == expected_min
    assert out["roll_max_3"].tolist() == expected_max


def test_update_lags_and_rollings_updates_min_max():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5]})
    cfg = {"features": {"rollings": [3], "lags": [1]}}
    ctx = create_lags_and_rolling_features(df, "y", [], cfg)
    new_ctx = update_lags_and_rollings(ctx, 6, cfg)
    last = new_ctx.iloc[-1]
    assert last["roll_min_3"] == 3
    assert last["roll_max_3"] == 6


def test_build_dynamic_row_includes_min_max():
    history = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    row = _build_dynamic_row(history, None, [], [3], "date", pd.Timestamp("2020-01-01"))
    assert row["roll_min_3"].iat[0] == 3.0
    assert row["roll_max_3"].iat[0] == 5.0


def test_compute_dynamic_features_with_min_max():
    hist_matrix = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    dsls_hist_matrix = np.zeros((1, 1), dtype=np.float32)
    lag_steps = np.array([], dtype=np.int64)
    lag_idx = np.array([], dtype=np.int64)
    roll_steps = np.array([3], dtype=np.int64)
    roll_mean_idx = np.array([-1], dtype=np.int64)
    roll_std_idx = np.array([-1], dtype=np.int64)
    roll_min_idx = np.array([0], dtype=np.int64)
    roll_max_idx = np.array([1], dtype=np.int64)
    dsls_idx = -1
    rzero_idx = -1
    avg_idi_idx = -1
    out = np.zeros((1, 2), dtype=np.float32)
    _compute_dynamic_features(
        hist_matrix,
        dsls_hist_matrix,
        lag_steps,
        lag_idx,
        roll_steps,
        roll_mean_idx,
        roll_std_idx,
        roll_min_idx,
        roll_max_idx,
        dsls_idx,
        rzero_idx,
        avg_idi_idx,
        out,
    )
    assert out[0, 0] == 3.0
    assert out[0, 1] == 5.0
