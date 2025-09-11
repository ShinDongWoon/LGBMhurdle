import pandas as pd
from g2_hurdle.fe.preprocess import prepare_features


def test_prepare_features_missing_category_map():
    df = pd.DataFrame(
        {
            "store_id": pd.Series(["A", None, "B"], dtype="category"),
            "obj_col": ["x", None, "y"],
            "num_col": [1.0, 2.0, None],
        }
    )

    X, feature_cols, categorical_cols = prepare_features(
        df,
        drop_cols=[],
        categories_map={"store_id": ["A", "B"]},
    )

    assert pd.api.types.is_categorical_dtype(X["store_id"])
    assert "missing" in X["store_id"].cat.categories
    assert X.loc[1, "store_id"] == "missing"
    assert X.select_dtypes(include="object").empty
