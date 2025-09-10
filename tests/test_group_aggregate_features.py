import numpy as np
import pandas as pd

from g2_hurdle.fe.group_aggregate import create_group_aggregate_features


def test_group_aggregate_features():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "store_id": ["A", "A", "B", "B"],
            "menu_id": ["X", "Y", "X", "Y"],
            "sales": [1, 2, 3, 4],
        }
    )
    out = create_group_aggregate_features(df, "sales")
    row_a = out[out["store_id"] == "A"].iloc[0]
    assert np.isclose(row_a["store_id_avg_sales"], 1.5)
    assert np.isclose(row_a["store_id_volatility"], np.std([1, 2], ddof=1))
    row_x = out[out["menu_id"] == "X"].iloc[0]
    assert np.isclose(row_x["menu_id_avg_sales"], 2.0)
    assert np.isclose(row_x["menu_id_volatility"], np.std([1, 3], ddof=1))
