
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..utils.logging import get_logger
from ..utils.timer import Timer
from ..utils.io import load_data, save_artifacts
from ..utils.keys import build_series_id
from ..utils.preprocessing import ensure_min_positive_ratio
from ..fe import run_feature_engineering, prepare_features
from ..cv.tscv import rolling_forecast_origin_split
from ..model.classifier import HurdleClassifier
from ..model.regressor import HurdleRegressor
from ..model.threshold import find_optimal_threshold

logger = get_logger("Train")


class ZeroPredictor:
    """Simple model that always predicts zeros."""

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros(len(X))

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
        date_col = schema["date"]
        target_col = schema["target"]
        series_cols = schema["series"]
        df = df.sort_values([*series_cols, date_col]).reset_index(drop=True)
        df["id"] = build_series_id(df, series_cols)

    min_pos_ratio = float(cfg.get("data", {}).get("min_positive_ratio", 0.0))
    seed = int(cfg.get("runtime", {}).get("seed", 42))

    with Timer("Feature engineering"):
        fe, embed_map = run_feature_engineering(df, cfg, schema)
        drop_cols = [
            date_col,
            target_col,
            "id",
            *[c for c in series_cols if c not in ("store_id", "menu_id")],
        ]
        X_all, feature_cols, categorical_cols = prepare_features(fe, drop_cols)
        # ensure calendar components are treated as categorical features
        base_cats = [c for c in ["dow", "week", "month", "quarter", "holiday_name"] if c in X_all.columns]
        categorical_cols = sorted(set(categorical_cols).union(base_cats))
        if "holiday_name" in X_all.columns:
            assert pd.api.types.is_categorical_dtype(
                X_all["holiday_name"]
            ), "holiday_name should be categorical after prepare_features"
        y_all = df[target_col].values

    H = int(cfg.get("cv", {}).get("horizon", 7))
    init_ratio = float(cfg.get("cv", {}).get("init_train_ratio", 0.7))
    esr = int(cfg.get("cv", {}).get("early_stopping_rounds", 100))
    n_jobs = int(cfg.get("runtime", {}).get("n_jobs", 1))
    if cfg.get("runtime", {}).get("use_gpu", False) and n_jobs != 1:
        logger.warning("use_gpu=True detected; forcing n_jobs=1 for sequential CV.")
        n_jobs = 1

    y_true_all, p_all, q_all = [], [], []
    preds_all = []

    # Precompute splits so they can be processed in parallel
    split_args = list(rolling_forecast_origin_split(df, date_col, H, init_ratio))

    def _run_fold(split):
        tr_mask, va_mask, (tr_end, va_start, va_end) = split
        df_tr = df.loc[tr_mask].copy()
        df_va = df.loc[va_mask].copy()
        tr_inner, va_inner = _split_train_valid_by_tail_dates(df_tr, date_col, ratio_tail=28)

        X_tr = X_all.loc[tr_inner.index, feature_cols]
        y_tr = tr_inner[target_col].values
        if va_inner is not None:
            X_val = X_all.loc[va_inner.index, feature_cols]
            y_val = va_inner[target_col].values
        else:
            X_val, y_val = None, None

        min_pos_samples = int(cfg.get("cv", {}).get("min_positive_samples", 0))
        min_neg_samples = int(cfg.get("cv", {}).get("min_negative_samples", 0))
        y_tr_bin = (y_tr > 0)
        unique = np.unique(y_tr_bin)
        pos_count = int(y_tr_bin.sum())
        neg_count = int(len(y_tr_bin) - pos_count)
        skip_fold = False
        if len(unique) < 2:
            skip_fold = True
            logger.warning(
                f"Fold (train_end={tr_end.date()}): training data has a single class; skipping training.")
        elif pos_count < min_pos_samples or neg_count < min_neg_samples:
            skip_fold = True
            logger.warning(
                f"Fold (train_end={tr_end.date()}): only {pos_count} positive or {neg_count} negative samples; skipping training.")

        preds_df = None
        if not skip_fold:
            drop_cols_tr = [c for c in X_tr.columns if X_tr[c].nunique(dropna=True) <= 1]
            if drop_cols_tr:
                X_tr = X_tr.drop(columns=drop_cols_tr)
                if X_val is not None:
                    X_val = X_val.drop(columns=drop_cols_tr, errors="ignore")
            feature_cols_tr = [c for c in feature_cols if c not in drop_cols_tr]
            categorical_cols_tr = [c for c in categorical_cols if c not in drop_cols_tr]
            if not feature_cols_tr:
                skip_fold = True
                logger.warning(
                    f"Fold (train_end={tr_end.date()}): no usable features after constant-column removal; skipping.")
            else:
                X_tr = X_tr[feature_cols_tr]
                if X_val is not None:
                    X_val = X_val[feature_cols_tr]

                if min_pos_ratio > 0:
                    X_tr, y_tr = ensure_min_positive_ratio(X_tr, y_tr, min_pos_ratio, seed=seed)

                cat_tr = [c for c in categorical_cols_tr if c in X_tr.columns]
                cls_params = dict(cfg.get("model", {}).get("classifier", {}))
                reg_params = dict(cfg.get("model", {}).get("regressor", {}))
                if cfg.get("runtime", {}).get("use_gpu", False):
                    # Ensure GPU-specific defaults are present; adjust max_bin or num_leaves if OOM occurs
                    cls_params.setdefault("device_type", "gpu")
                    reg_params.setdefault("device_type", "gpu")
                    cls_params.setdefault("gpu_use_dp", False)
                    reg_params.setdefault("gpu_use_dp", False)
                    cls_params.setdefault("max_bin", 255)
                    reg_params.setdefault("max_bin", 255)
                    cls_params.setdefault("feature_fraction", 0.8)
                    reg_params.setdefault("feature_fraction", 0.8)
                reg = HurdleRegressor(reg_params, categorical_feature=cat_tr)

                # Fit regressor first
                with Timer(f"Fit fold (train_end={tr_end.date()}) - regressor"):
                    reg.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=esr)
                    reg.feature_names_ = list(
                        getattr(getattr(reg, "model", reg), "feature_name_", getattr(reg, "feature_names_", []))
                    )

                # Refit classifier on regressor-selected features
                reg_feats = list(getattr(reg, "feature_names_", []))
                X_tr = X_tr[reg_feats]
                if X_val is not None:
                    X_val = X_val[reg_feats]
                cat_tr = [c for c in cat_tr if c in reg_feats]
                clf = HurdleClassifier(cls_params, categorical_feature=cat_tr)
                with Timer(f"Fit fold (train_end={tr_end.date()}) - classifier"):
                    clf.fit(X_tr, y_tr, X_val, y_val, early_stopping_rounds=esr)
                    clf.feature_names_ = list(
                        getattr(getattr(clf, "model", clf), "feature_name_", getattr(clf, "feature_names_", []))
                    )

                # Recursive simulate on validation horizon per id using regressor feature list
                from .recursion import recursive_forecast_grouped
                # Use context: all rows up to tr_end per series
                context = df[df[date_col] <= tr_end].copy()
                preds_df = recursive_forecast_grouped(
                    context,
                    schema,
                    cfg,
                    clf,
                    reg,
                    threshold=0.5,
                    horizon=H,
                    feature_cols=reg_feats,
                    categorical_cols=cat_tr,
                )
        if skip_fold:
            ids = df_va["id"].unique()
            preds_df = pd.DataFrame({"id": ids})
            for i in range(1, H + 1):
                preds_df[f"D{i}"] = 0.0
        # For threshold tuning we need y_true for validation horizon
        # Construct ground truth for va period in wide form
        va_truth = df_va.copy()
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
        y_true_fold, p_fold, q_fold = [], [], []
        for i in range(1, H + 1):
            yp = merged[f"D{i}_pred"].values
            yt = merged[f"D{i}_true"].values
            if skip_fold:
                p_fold.extend(np.zeros_like(yt))
                q_fold.extend(np.zeros_like(yt))
            else:
                # For tuning, we approximate p and q by using yp as reg outcome and assume proba ~1 if yp>0 else 0.3
                # (Simplification: exact p,q not returned from recursion; keeping interface lean.)
                p_fold.extend((yp > 0).astype(float) * 0.7 + 0.3 * (yp <= 0))
                q_fold.extend(yp)
            y_true_fold.extend(yt)

        return preds_df, y_true_fold, p_fold, q_fold

    with Timer("Rolling CV"):
        if n_jobs == 1:
            fold_outputs = [_run_fold(s) for s in split_args]
        else:
            fold_outputs = Parallel(n_jobs=n_jobs)(delayed(_run_fold)(s) for s in split_args)

    folds = len(fold_outputs)
    for preds_df, y_true_fold, p_fold, q_fold in fold_outputs:
        preds_all.append(preds_df)
        y_true_all.extend(y_true_fold)
        p_all.extend(p_fold)
        q_all.extend(q_fold)

    with Timer("Threshold search"):
        from ..model.threshold import find_optimal_threshold
        y_true_all_arr = np.asarray(y_true_all, dtype=float)
        p_all_arr = np.asarray(p_all, dtype=float)
        q_all_arr = np.asarray(q_all, dtype=float)
        t_star, score = find_optimal_threshold(y_true_all_arr, p_all_arr, q_all_arr, cfg)
        logger.info(f"Optimal threshold={t_star:.3f}, CV wSMAPE≈{score:.3f}")

    # Retrain on full data
    # Repeat variance check on full dataset
    drop_cols_full = [c for c in X_all.columns if X_all[c].nunique(dropna=True) <= 1]
    if drop_cols_full:
        X_all = X_all.drop(columns=drop_cols_full)
    feature_cols = X_all.columns.tolist()
    categorical_cols = X_all.select_dtypes(include="category").columns.tolist()
    if X_all.shape[1] == 0:
        logger.warning("Full data has no usable features after constant-column removal; using ZeroPredictor.")
        clf_final = ZeroPredictor()
        reg_final = ZeroPredictor()
    else:
        with Timer("Final fit on full data"):
            X = X_all.loc[:, feature_cols]
            y = y_all
            cls_params = dict(cfg.get("model", {}).get("classifier", {}))
            reg_params = dict(cfg.get("model", {}).get("regressor", {}))
            if cfg.get("runtime", {}).get("use_gpu", False):
                # Ensure GPU defaults are applied; lower max_bin or num_leaves if memory is tight
                cls_params.setdefault("device_type", "gpu")
                reg_params.setdefault("device_type", "gpu")
                cls_params.setdefault("gpu_use_dp", False)
                reg_params.setdefault("gpu_use_dp", False)
                cls_params.setdefault("max_bin", 255)
                reg_params.setdefault("max_bin", 255)
                cls_params.setdefault("feature_fraction", 0.8)
                reg_params.setdefault("feature_fraction", 0.8)
            min_pos_samples = int(cfg.get("cv", {}).get("min_positive_samples", 0))
            min_neg_samples = int(cfg.get("cv", {}).get("min_negative_samples", 0))
            y_bin_full = (y > 0)
            unique_full = np.unique(y_bin_full)
            pos_count_full = int(y_bin_full.sum())
            neg_count_full = int(len(y_bin_full) - pos_count_full)
            if len(unique_full) < 2:
                logger.warning("Full data has a single class; using ZeroPredictor.")
                clf_final = ZeroPredictor()
                reg_final = ZeroPredictor()
            elif pos_count_full < min_pos_samples or neg_count_full < min_neg_samples:
                logger.warning(
                    f"Full data has only {pos_count_full} positive or {neg_count_full} negative samples; using ZeroPredictor.")
                clf_final = ZeroPredictor()
                reg_final = ZeroPredictor()
            else:
                if min_pos_ratio > 0:
                    X, y = ensure_min_positive_ratio(X, y, min_pos_ratio, seed=seed)

                # Fit regressor first on all features
                reg_final = HurdleRegressor(reg_params, categorical_feature=categorical_cols)
                with Timer("Final fit - regressor"):
                    reg_final.fit(X, y, None, None, early_stopping_rounds=0)
                    reg_final.feature_names_ = list(
                        getattr(
                            getattr(reg_final, "model", reg_final),
                            "feature_name_",
                            getattr(reg_final, "feature_names_", feature_cols),
                        )
                    )

                # Refit classifier using regressor-selected features
                reg_feats = list(getattr(reg_final, "feature_names_", []))
                X = X[reg_feats]
                cat_final = [c for c in categorical_cols if c in reg_feats]
                clf_final = HurdleClassifier(cls_params, categorical_feature=cat_final)
                with Timer("Final fit - classifier"):
                    clf_final.fit(X, y, None, None, early_stopping_rounds=0)
                    clf_final.feature_names_ = list(
                        getattr(
                            getattr(clf_final, "model", clf_final),
                            "feature_name_",
                            getattr(clf_final, "feature_names_", reg_feats),
                        )
                    )

                # ensure categorical_cols reflects regressor-selected subset
                categorical_cols = cat_final
                feature_cols = reg_feats

    artifacts = {
        "classifier.pkl": clf_final,
        "regressor.pkl": reg_final,
        "threshold.json": {"threshold": float(t_star)},
        "schema.json": schema,
        "config.json": cfg,
        "version.txt": f"generated_by=g2_hurdle; folds={folds}",
        "features.json": {"feature_cols": feature_cols, "categorical_cols": categorical_cols},
        "embeddings.json": embed_map,
    }
    save_artifacts(artifacts, artifacts_dir)
    logger.info("Training complete.")
