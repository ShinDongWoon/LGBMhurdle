
import os, glob
import re
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.timer import Timer
from ..utils.io import load_artifacts, load_data
from ..utils.keys import (
    align_to_submission,
    ensure_wide_columns,
    normalize_series_name,
)
from ..fe import run_feature_engineering, prepare_features
from .recursion import recursive_forecast_grouped

logger = get_logger("Predict")

def run_predict(cfg: dict):
    paths = cfg.get("paths", {})
    test_dir = paths["test_dir"]
    sample_path = paths["sample_submission"]
    out_path = paths["out_path"]
    artifacts_dir = paths.get("artifacts_dir", cfg.get("io", {}).get("artifacts_dir", "./artifacts"))

    with Timer("Load artifacts"):
        art = load_artifacts(artifacts_dir)
        clf = art.get("classifier.pkl")
        reg = art.get("regressor.pkl")
        thresh = float(art.get("threshold.json", {}).get("threshold", 0.5))
        schema = art.get("schema.json", None) or {}
        if schema:
            schema["series"] = ["store_menu_id"]
        features_meta = art.get("features.json", {})
        feature_cols = features_meta.get("feature_cols", [])
        categorical_cols = features_meta.get("categorical_cols", [])
        categories_map = features_meta.get("categories", {})
        dtw_clusters = art.get("dtw_clusters.json", {})
        te_map = art.get("target_encoding.pkl", {})
        base_cats = [
            "week",
            "holiday_name",
            "store_id",
            "menu_id",
            "store_menu_id",
            "demand_cluster",
        ]
        categorical_cols = sorted(set(categorical_cols).union(base_cats))
        train_cfg = art.get("config.json", {})
        if "features" in train_cfg:
            cfg["features"] = train_cfg["features"]

    H = int(cfg.get("cv", {}).get("horizon", 7))

    # Collect predictions across TEST_* files
    pred_all = {}
    test_files = sorted(glob.glob(os.path.join(test_dir, "TEST_*.csv")))
    if not test_files:
        raise FileNotFoundError("No TEST_*.csv files found")
    for f in test_files:
        test_name = os.path.splitext(os.path.basename(f))[0]
        df, _schema = load_data(
            f,
            {"data": {"date_col_candidates": [schema.get("date")], "target_col_candidates": [schema.get("target")], "id_col_candidates": schema.get("series", [])}} if schema else cfg,
        )
        _schema["series"] = ["store_menu_id"]
        # ensure id
        df["id"] = normalize_series_name(df["store_menu_id"])
        if dtw_clusters:
            df["demand_cluster"] = (
                df["store_menu_id"].map(dtw_clusters).astype("category")
            )
        # context length check
        min_ctx = int(cfg.get("data", {}).get("min_context_days", 28))
        # For each id, ensure at least 28 rows
        bad = [sid for sid, g in df.groupby("id") if len(g) < min_ctx]
        if bad:
            raise ValueError(f"{os.path.basename(f)}: some series have < {min_ctx} days: {bad[:5]} ...")

        # Optionally compute features to ensure column alignment
        schema_use = schema or _schema
        schema_use["series"] = ["store_menu_id"]
        df = df.sort_values(["store_menu_id", schema_use["date"]]).reset_index(drop=True)
        fe, _ = run_feature_engineering(df, cfg, schema_use, mapping=te_map)
        drop_cols = [
            schema_use["date"],
            schema_use["target"],
            "id",
            *[
                c
                for c in schema_use["series"]
                if c not in ("store_id", "menu_id", "store_menu_id")
            ],
        ]
        X_test, _, _ = prepare_features(
            fe, drop_cols, feature_cols, categorical_cols, categories_map
        )
        if "holiday_name" in X_test.columns:
            assert pd.api.types.is_categorical_dtype(
                X_test["holiday_name"]
            ), "holiday_name should be categorical after prepare_features"

        preds_df = recursive_forecast_grouped(
            df,
            schema_use,
            cfg,
            clf,
            reg,
            threshold=thresh,
            horizon=H,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
        )
        pred_all[test_name] = preds_df

    preds = pd.concat(pred_all.values(), ignore_index=True)
    # Load sample submission and align
    sub = pd.read_csv(sample_path, encoding="utf-8-sig", dtype=str)
    id_col = "id" if "id" in sub.columns else None
    if id_col is None:
        row_key_col = sub.columns[0]
        menu_cols = [c for c in sub.columns if c != row_key_col]
        out = sub.copy()

        # Normalize menu column names to match prediction ids
        menu_map = {c: normalize_series_name(c) for c in menu_cols}

        for idx, row in out.iterrows():
            row_key = row[row_key_col]
            if "+" not in row_key:
                continue
            test_part, day_part = row_key.split("+", 1)
            day_match = re.search(r"\d+", day_part)
            if not day_match:
                continue
            day_col = f"D{int(day_match.group())}"
            preds_df = pred_all.get(test_part)
            if preds_df is None or day_col not in preds_df.columns:
                continue
            for orig_col in menu_cols:
                pid = menu_map[orig_col]
                val = preds_df.loc[preds_df["id"] == pid, day_col]
                if not val.empty:
                    out.at[idx, orig_col] = val.iloc[0]

        assert list(out.columns) == list(sub.columns), "Output columns differ from sample submission"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved submission to {out_path}")
        return

    needed = ensure_wide_columns(H)
    # ensure preds has D1..Dh
    for c in needed:
        if c not in preds.columns:
            preds[c] = np.nan
    out = align_to_submission(sub, preds[["id", *needed]], id_col="id")
    assert list(out.columns) == list(sub.columns), "Output columns differ from sample submission"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved submission to {out_path}")
