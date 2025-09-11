import copy
import pandas as pd
from g2_hurdle.fe import run_feature_engineering, compute_dtw_clusters


def test_dtw_cluster_feature():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5).tolist() * 2,
            "store_menu_id": ["A"] * 5 + ["B"] * 5,
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    cfg = {
        "features": {
            "dtw": {"enable": True, "n_clusters": 2, "use_gpu": False},
            "lags": [],
            "rollings": [],
            "fourier": {"weekly_K": 0, "yearly_K": 0},
            "intermittency": {"enable": False},
            "use_holidays": False,
        }
    }
    schema = {"date": "date", "target": "y", "series": ["store_menu_id"]}

    out, extras = run_feature_engineering(df, cfg, schema)

    assert "demand_cluster" in out.columns
    assert pd.api.types.is_categorical_dtype(out["demand_cluster"])
    assert set(extras["dtw_clusters"].keys()) == set(df["store_menu_id"].unique())


def test_dtw_cluster_train_only():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5).tolist() * 3,
            "store_menu_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "y": [0, 1, 0, 1, 0] * 3,
        }
    )
    cfg = {
        "features": {
            "dtw": {"enable": True, "n_clusters": 2, "use_gpu": False},
            "lags": [],
            "rollings": [],
            "fourier": {"weekly_K": 0, "yearly_K": 0},
            "intermittency": {"enable": False},
            "use_holidays": False,
        }
    }
    schema = {"date": "date", "target": "y", "series": ["store_menu_id"]}

    cfg_off = copy.deepcopy(cfg)
    cfg_off["features"]["dtw"]["enable"] = False
    fe_base, _ = run_feature_engineering(df, cfg_off, schema)

    df_tr = df[df["store_menu_id"].isin(["A", "B"])]
    clusters = compute_dtw_clusters(df_tr, schema, n_clusters=2, use_gpu=False)
    fe_fold = fe_base.copy()
    fe_fold["demand_cluster"] = fe_fold["store_menu_id"].map(clusters)

    assert set(clusters.keys()) == {"A", "B"}
    assert fe_fold.loc[fe_fold["store_menu_id"] == "C", "demand_cluster"].isna().all()
