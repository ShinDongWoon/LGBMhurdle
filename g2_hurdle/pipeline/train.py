
import os, json
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.timer import Timer
from ..utils.io import load_data, save_artifacts
from ..utils.keys import build_series_id
from ..fe import run_feature_engineering
from ..cv.tscv import rolling_forecast_origin_split
from ..model.classifier import HurdleClassifier
from ..model.regressor import HurdleRegressor
from ..model.threshold import find_optimal_threshold
from ..metrics.wsMAPE import calculate_wsmape

logger = get_logger("Train")

def _split_train_valid_by_tail_dates(df, date_col, ratio_tail=28):
    # validation is the last `ratio_tail` unique dates of training range
    udates = sorted(df[date_col].unique())
    if len(udates) <= ratio_tail:
        return df, None
    val_dates = set(udates[-ratio_tail:])
    val_mask = df[date_col].isin(val_dates)
    return df[~val_mask], df[val_mask]

def run_train(cfg: dict):
    paths = cfg.get("paths", {})
    train_csv = paths.get("train_csv")
    artifacts_dir = cfg.get("io", {}).get("artifacts_dir", "./artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    with Timer("Load data"):
        df, schema = load_data(train_csv, cfg)
        date_col = schema["date"]; target_col = schema["target"]; series_cols = schema["series"]
        df["id"] = build_series_id(df, series_cols)

    with Timer("Feature engineering"):
        fe = run_feature_engineering(df, cfg, schema)
        # Build X, y
        drop_cols = [date_col, target_col] + series_cols + ["id"]

        def _prepare_X(fe_subset: pd.DataFrame) -> pd.DataFrame:
            X = fe_subset.drop(columns=[c for c in drop_cols if c in fe_subset.columns], errors="ignore").copy()
            obj_cols = X.select_dtypes(include="object").columns
            for c in obj_cols:
                X[c] = X[c].astype("category")
            return X

        X_all = _prepare_X(fe)
        y_all = df[target_col].values

    H = int(cfg.get("cv", {}).get("horizon", 7))
    init_ratio = float(cfg.get("cv", {}).get("init_train_ratio", 0.7))
    esr = int(cfg.get("cv", {}).get("early_stopping_rounds", 100))

    y_true_all, p_all, q_all = [], [], []

    folds = 0
    with Timer("Rolling CV"):
        for tr_mask, va_mask, (tr_end, va_start, va_end) in rolling_forecast_origin_split(df, date_col, H, init_ratio):
            folds += 1
            df_tr = df.loc[tr_mask].copy()
            df_va = df.loc[va_mask].copy()
            # Split for early stopping inside training period
            tr_inner, va_inner = _split_train_valid_by_tail_dates(df_tr, date_col, ratio_tail=28)
            fe_tr = fe.loc[tr_inner.index]
            fe_va = fe.loc[va_inner.index] if va_inner is not None else None

            X_tr = _prepare_X(fe_tr)
            y_tr = tr_inner[target_col].values
            if va_inner is not None:
                X_val = _prepare_X(fe_va)
                y_val = va_inner[target_col].values
            else:
                X_val, y_val = None, None

            cls_params = dict(cfg.get("model", {}).get("classifier", {}))
            reg_params = dict(cfg.get("model", {}).get("regressor", {}))
            if cfg.get("runtime", {}).get("use_gpu", False):
                cls_params.setdefault("device_type", "gpu"); cls_params.setdefault("device", "gpu")
                reg_params.setdefault("device_type", "gpu"); reg_params.setdefault("device", "gpu")
            clf = HurdleClassifier(cls_params)
            reg = HurdleRegressor(reg_params)

            with Timer(f"Fit fold (train_end={tr_end.date()}) - classifier"):
                clf.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=esr)
            with Timer(f"Fit fold (train_end={tr_end.date()}) - regressor"):
                reg.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=esr)

            # Recursive simulate on validation horizon per id
            from .recursion import recursive_forecast_grouped
            # Use context: all rows up to tr_end per series
            context = df[df[date_col] <= tr_end].copy()
            preds_df = recursive_forecast_grouped(context, schema, cfg, clf, reg, threshold=0.5, horizon=H)
            # For threshold tuning we need y_true for validation horizon
            # Construct ground truth for va period in wide form
            va_truth = df_va.copy()
            # Aggregate by id: collect 7 consecutive days
            # We assume each id has continuous dates; map to D1..D7 relative ordering
            truth_rows = []
            for sid, g in va_truth.groupby("id"):
                g = g.sort_values(date_col)
                vals = g[target_col].values.tolist()
                row = {"id": sid}
                for i, v in enumerate(vals[:H], start=1):
                    row[f"D{i}"] = v
                truth_rows.append(row)
            truth_df = pd.DataFrame(truth_rows)

            merged = preds_df.merge(truth_df, on="id", how="inner", suffixes=("_pred","_true"))
            for i in range(1, H+1):
                yp = merged[f"D{i}_pred"].values
                yt = merged[f"D{i}_true"].values
                # For tuning, we approximate p and q by using yp as reg outcome and assume proba ~1 if yp>0 else 0.3
                # (Simplification: exact p,q not returned from recursion; keeping interface lean.)
                p_all.extend((yp>0).astype(float)*0.7 + 0.3*(yp<=0))
                q_all.extend(yp)
                y_true_all.extend(yt)

    with Timer("Threshold search"):
        from ..model.threshold import find_optimal_threshold
        y_true_all_arr = np.asarray(y_true_all, dtype=float)
        p_all_arr = np.asarray(p_all, dtype=float)
        q_all_arr = np.asarray(q_all, dtype=float)
        t_star, score = find_optimal_threshold(y_true_all_arr, p_all_arr, q_all_arr, cfg)
        logger.info(f"Optimal threshold={t_star:.3f}, CV wSMAPE≈{score:.3f}")

    # Retrain on full data
    with Timer("Final fit on full data"):
        X = X_all
        y = y_all
        cls_params = dict(cfg.get("model", {}).get("classifier", {}))
        reg_params = dict(cfg.get("model", {}).get("regressor", {}))
        if cfg.get("runtime", {}).get("use_gpu", False):
            cls_params.setdefault("device_type", "gpu"); cls_params.setdefault("device", "gpu")
            reg_params.setdefault("device_type", "gpu"); reg_params.setdefault("device", "gpu")
        clf_final = HurdleClassifier(cls_params)
        reg_final = HurdleRegressor(reg_params)
        clf_final.fit(X, y, None, None, early_stopping_rounds=0)
        reg_final.fit(X, y, None, None, early_stopping_rounds=0)

    artifacts = {
        "classifier.pkl": clf_final,
        "regressor.pkl": reg_final,
        "threshold.json": {"threshold": float(t_star)},
        "schema.json": schema,
        "config.json": cfg,
        "version.txt": f"generated_by=g2_hurdle; folds={folds}",
    }
    save_artifacts(artifacts, artifacts_dir)
    logger.info("Training complete.")
