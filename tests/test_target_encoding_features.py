import numpy as np
import pandas as pd

from g2_hurdle.fe.embeddings import create_target_encoding_features


def test_target_encoding_features_with_mapping_and_unseen():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "store_id": [1, 1, 2, 2],
            "menu_id": [10, 20, 10, 30],
            "y": [1, 2, 3, 4],
        }
    )
    cfg = {"features": {"target_encoding": {"smoothing": 0}}}

    out, mapping = create_target_encoding_features(
        df, ["store_id", "menu_id"], "y", "d", cfg, None
    )

    # Check that expected columns exist
    for col in [
        "store_id_te_mean",
        "store_id_te_std",
        "menu_id_te_mean",
        "menu_id_te_std",
    ]:
        assert col in out.columns

    # Verify encoded values on the training frame
    assert np.allclose(out["store_id_te_mean"], [0.0, 1.0, 1.5, 3.0])
    assert np.allclose(out["store_id_te_std"], [0.0, 0.0, 0.5, 0.0])
    assert np.allclose(out["menu_id_te_mean"], [0.0, 1.0, 1.0, 2.0])
    assert np.allclose(out["menu_id_te_std"], [0.0, 0.0, 0.0, np.sqrt(2 / 3)])

    # Mapping contains per-category and default statistics
    assert np.isclose(mapping["store_id"]["1"]["mean"], 1.5)
    assert np.isclose(mapping["store_id"]["2"]["std"], 0.5)
    assert np.isclose(mapping["menu_id"]["10"]["mean"], 2.0)
    assert np.isclose(mapping["menu_id"]["30"]["std"], 0.0)
    assert np.isclose(
        mapping["store_id"]["__default__"]["mean"], mapping["__global__"]["mean"]
    )
    assert np.isclose(
        mapping["menu_id"]["__default__"]["std"], mapping["__global__"]["std"]
    )

    # Reapply mapping to new data containing seen and unseen identifiers
    df2 = pd.DataFrame(
        {
            "d": pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-07"]),
            "store_id": [1, 3, 4],
            "menu_id": [20, 10, 40],
            "y": [0, 0, 0],
        }
    )
    out2, _ = create_target_encoding_features(
        df2, ["store_id", "menu_id"], "y", "d", cfg, mapping
    )

    s_default = mapping["store_id"]["__default__"]
    m_default = mapping["menu_id"]["__default__"]

    # Row 0: both identifiers seen during fitting
    assert np.isclose(out2.loc[0, "store_id_te_mean"], mapping["store_id"]["1"]["mean"])
    assert np.isclose(out2.loc[0, "menu_id_te_mean"], mapping["menu_id"]["20"]["mean"])

    # Row 1: unseen store_id but seen menu_id
    assert np.isclose(out2.loc[1, "store_id_te_mean"], s_default["mean"])
    assert np.isclose(out2.loc[1, "store_id_te_std"], s_default["std"])
    assert np.isclose(out2.loc[1, "menu_id_te_mean"], mapping["menu_id"]["10"]["mean"])
    assert np.isclose(out2.loc[1, "menu_id_te_std"], mapping["menu_id"]["10"]["std"])

    # Row 2: unseen identifiers receive default statistics
    assert np.isclose(out2.loc[2, "store_id_te_mean"], s_default["mean"])
    assert np.isclose(out2.loc[2, "store_id_te_std"], s_default["std"])
    assert np.isclose(out2.loc[2, "menu_id_te_mean"], m_default["mean"])
    assert np.isclose(out2.loc[2, "menu_id_te_std"], m_default["std"])

