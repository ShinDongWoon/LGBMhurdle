import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe.static import prepare_static_future_features


def test_prepare_static_future_features_timezone():
    df = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Seoul")}
    )
    schema = {"date": "ds"}
    cfg = {"features": {}}
    out = prepare_static_future_features(df, schema, cfg, horizon=3)
    assert out.index.tz.zone == "Asia/Seoul"
    assert len(out) == 3
