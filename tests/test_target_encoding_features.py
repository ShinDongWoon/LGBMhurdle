import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.fe.embeddings import create_target_encoding_features


def test_target_encoding_global_stats_incremental_and_persisted():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "cat": ["a", "a", "a"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    cfg = {"features": {"target_encoding": {"smoothing": 0}}}

    out, mapping = create_target_encoding_features(
        df, ["cat"], "y", "d", cfg, None
    )

    assert np.allclose(out["cat_te_mean"].tolist(), [0.0, 1.0, 1.5])
    assert np.allclose(out["cat_te_std"].tolist(), [0.0, 0.0, 0.5])

    assert np.isclose(mapping["__global__"]["mean"], 2.0)
    assert np.isclose(mapping["__global__"]["std"], np.sqrt(2 / 3))

    df_inf = pd.DataFrame(
        {
            "d": pd.to_datetime(["2020-01-04"]),
            "cat": ["b"],
            "y": [1000.0],
        }
    )
    out_inf, _ = create_target_encoding_features(
        df_inf, ["cat"], "y", "d", cfg, mapping
    )

    # Uses stored mapping for unseen category 'b'
    assert np.allclose(out_inf["cat_te_mean"].tolist(), [2.0])
    assert np.allclose(out_inf["cat_te_std"].tolist(), [np.sqrt(2 / 3)])
