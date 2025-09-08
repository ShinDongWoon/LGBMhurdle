
from typing import Optional, Sequence
import numpy as np

from ..utils.logging import get_logger

try:
    import lightgbm
    from lightgbm import LGBMRegressor
except Exception as e:
    lightgbm = None
    LGBMRegressor = None


logger = get_logger("Regressor")


class ZeroPredictor:
    """Simple model that always predicts zeros."""

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class HurdleRegressor:
    def __init__(self, model_params: dict, categorical_feature: Optional[Sequence[str]] = None):
        if LGBMRegressor is None:
            raise ImportError("lightgbm not installed. Please install lightgbm.")
        self.model = LGBMRegressor(**model_params)
        self.categorical_feature = categorical_feature or "auto"

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        """Fit the underlying regressor on positive targets only.

        After filtering to positive samples, columns that become constant
        (``<=1`` unique value) are removed from both training and validation
        data to avoid feeding non-informative features to the model.
        """
        mask_tr = (y_train > 0)
        pos_count = int(mask_tr.sum())
        min_leaf = int(self.model.get_params().get("min_child_samples", 20))
        if pos_count < min_leaf:
            if pos_count == 0:
                logger.warning("No positive samples; using ZeroPredictor for regressor.")
                self.model = ZeroPredictor()
                if hasattr(X_train, "columns"):
                    self.feature_names_ = list(X_train.columns)
                return self
            logger.warning(
                f"min_child_samples={min_leaf} exceeds positive samples={pos_count}; reducing to {pos_count}."
            )
            self.model.set_params(min_child_samples=pos_count)
        X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
        if hasattr(X_tr, "nunique"):
            tr_counts = X_tr.nunique()
            drop_cols = tr_counts[tr_counts <= 1].index
            if len(drop_cols) > 0:
                X_tr = X_tr.drop(columns=drop_cols)
        if X_val is not None and y_val is not None:
            mask_va = (y_val > 0)
            X_va, y_va = X_val[mask_va], y_val[mask_va]
            if hasattr(X_va, "nunique"):
                va_counts = X_va.nunique()
                drop_va_cols = va_counts[va_counts <= 1].index
                if len(drop_va_cols) > 0:
                    X_va = X_va.drop(columns=drop_va_cols)
            if hasattr(X_tr, "columns") and hasattr(X_va, "columns"):
                common = X_tr.columns.intersection(X_va.columns, sort=False)
                X_tr = X_tr[common]
                X_va = X_va[common]
            fit_params = {
                "categorical_feature": self.categorical_feature,
                "eval_set": [(X_va, y_va)],
            }
            if hasattr(X_tr, "columns"):
                fit_params["feature_name"] = list(X_tr.columns)
            if early_stopping_rounds > 0:
                callbacks = fit_params.get("callbacks", [])
                callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
                fit_params["callbacks"] = callbacks
        else:
            fit_params = {
                "categorical_feature": self.categorical_feature,
            }
            if hasattr(X_tr, "columns"):
                fit_params["feature_name"] = list(X_tr.columns)
        self.model.fit(X_tr, y_tr, **fit_params)
        if hasattr(X_tr, "columns"):
            self.feature_names_ = list(X_tr.columns)

    def predict(self, X):
        return self.model.predict(X)
