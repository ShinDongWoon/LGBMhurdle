import numpy as np
import pandas as pd


def prepare_features(fe_df: pd.DataFrame, drop_cols, feature_cols=None, categorical_cols=None, categories_map=None):
    X = fe_df.drop(columns=[c for c in drop_cols if c in fe_df.columns], errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in ["store_id", "menu_id"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    # Handle missing values separately for categorical and non-categorical columns
    cat_cols = X.select_dtypes(include="category").columns
    obj_cols = X.select_dtypes(include="object").columns
    non_cat_cols = [c for c in X.columns if c not in cat_cols.union(obj_cols)]
    # Fill non-categorical columns with 0
    X[non_cat_cols] = X[non_cat_cols].fillna(0)
    # For categorical columns, add a "missing" category and fill NaNs with it
    for c in cat_cols:
        if "missing" not in X[c].cat.categories:
            X[c] = X[c].cat.add_categories(["missing"])
        X[c] = X[c].fillna("missing")
    # Fill object columns with "missing" then convert to category
    for c in obj_cols:
        X[c] = X[c].fillna("missing").astype("category")
    # Ensure no object columns remain
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        X[c] = X[c].astype("category")
    if feature_cols is None:
        bad = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
        X = X.drop(columns=bad)
        feature_cols = X.columns.tolist()
        categorical_cols = X.select_dtypes(include="category").columns.tolist()
    else:
        missing = [c for c in feature_cols if c not in X.columns]
        for c in missing:
            X[c] = 0
        extra = [c for c in X.columns if c not in feature_cols]
        if extra:
            X = X.drop(columns=extra)
        X = X[feature_cols]
        for c in categorical_cols or []:
            if c in X.columns:
                X[c] = X[c].astype("category")

    if categories_map:
        for c, cats in categories_map.items():
            if c in X.columns:
                X[c] = (
                    X[c]
                    .astype("category")
                    .cat.set_categories(cats, inplace=False)
                    .fillna("missing")
                )

    return X, feature_cols, categorical_cols
