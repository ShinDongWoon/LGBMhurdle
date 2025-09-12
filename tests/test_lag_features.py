import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe.lags_rolling import (
    create_lags_and_rolling_features,
    update_lags_and_rollings,
)


def test_lag_2_and_14_creation_and_update():
    df = pd.DataFrame({"y": list(range(1, 21))})
    cfg = {"features": {"lags": [1, 2, 14], "rollings": []}}

    ctx = create_lags_and_rolling_features(df, "y", [], cfg)
    assert "lag_2" in ctx.columns
    assert "lag_14" in ctx.columns

    expected_lag2 = df["y"].shift(2).fillna(0).tolist()
    expected_lag14 = df["y"].shift(14).fillna(0).tolist()
    assert ctx["lag_2"].tolist() == expected_lag2
    assert ctx["lag_14"].tolist() == expected_lag14

    updated = update_lags_and_rollings(ctx, 21, cfg)
    last = updated.iloc[-1]
    assert last["lag_1"] == 21
    assert last["lag_2"] == ctx.iloc[-1]["lag_1"]
    assert last["lag_14"] == ctx["y"].iloc[-14]
