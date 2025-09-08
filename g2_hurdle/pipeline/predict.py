
import os, glob
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.timer import Timer
from ..utils.io import load_artifacts, load_data
from ..utils.keys import build_series_id, align_to_submission, ensure_wide_columns
from ..fe import run_feature_engineering
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

    H = int(cfg.get("cv", {}).get("horizon", 7))

    # Collect predictions across TEST_* files
    pred_all = []
    test_files = sorted(glob.glob(os.path.join(test_dir, "TEST_*.csv")))
    if not test_files:
        raise FileNotFoundError("No TEST_*.csv files found")
    for f in test_files:
        df, _schema = load_data(f, {"data": {"date_col_candidates": [schema.get("date")], "target_col_candidates": [schema.get("target")], "id_col_candidates": schema.get("series", [])}} if schema else cfg)
        # ensure id
        df["id"] = build_series_id(df, (schema or _schema)["series"])
        # context length check
        min_ctx = int(cfg.get("data", {}).get("min_context_days", 28))
        # For each id, ensure at least 28 rows
        bad = [sid for sid, g in df.groupby("id") if len(g) < min_ctx]
        if bad:
            raise ValueError(f"{os.path.basename(f)}: some series have < {min_ctx} days: {bad[:5]} ...")

        preds_df = recursive_forecast_grouped(df, schema or _schema, cfg, clf, reg, threshold=thresh, horizon=H)
        pred_all.append(preds_df)

    preds = pd.concat(pred_all, ignore_index=True)
    # Load sample submission and align
    sub = pd.read_csv(sample_path)
    id_col = "id" if "id" in sub.columns else None
    if id_col is None:
        # try to build id from schema series columns if present in sample
        if schema and all(c in sub.columns for c in schema.get("series", [])):
            sub = sub.copy()
            sub["id"] = build_series_id(sub, schema.get("series"))
            id_col = "id"
        else:
            # fallback: merge on index (unsafe but last resort)
            logger.warning("No id column found in sample_submission; falling back to row-wise fill")
            needed = ensure_wide_columns(H)
            out = sub.copy()
            for i, c in enumerate(needed):
                if c in out.columns and i < len(preds):
                    mask = out[c].isna()
                    out.loc[mask, c] = preds[c].values[:mask.sum()]
            out.to_csv(out_path, index=False)
            logger.info(f"Saved submission to {out_path} (fallback merge)")
            return

    needed = ensure_wide_columns(H)
    # ensure preds has D1..Dh
    for c in needed:
        if c not in preds.columns:
            preds[c] = np.nan
    out = align_to_submission(sub, preds[["id", *needed]], id_col="id")
    out.to_csv(out_path, index=False)
    logger.info(f"Saved submission to {out_path}")
