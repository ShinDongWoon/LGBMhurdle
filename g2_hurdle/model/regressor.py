
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
        mask_tr = (y_train > 0)
        pos_count = int(mask_tr.sum())
        min_leaf = int(self.model.get_params().get("min_data_in_leaf", 20))
        if pos_count < min_leaf:
            if pos_count == 0:
                logger.warning("No positive samples; using ZeroPredictor for regressor.")
                self.model = ZeroPredictor()
                return self
            logger.warning(
                f"min_data_in_leaf={min_leaf} exceeds positive samples={pos_count}; reducing to {pos_count}."
            )
            self.model.set_params(min_data_in_leaf=pos_count)
        X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
        fit_params = {
            "categorical_feature": self.categorical_feature,
        }
        if hasattr(X_train, "columns"):
            fit_params["feature_name"] = list(X_train.columns)
        if X_val is not None and y_val is not None:
            mask_va = (y_val > 0)
            X_va, y_va = X_val[mask_va], y_val[mask_va]
            fit_params["eval_set"] = [(X_va, y_va)]
            if early_stopping_rounds > 0:
                callbacks = fit_params.get("callbacks", [])
                callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
                fit_params["callbacks"] = callbacks
        self.model.fit(X_tr, y_tr, **fit_params)

    def predict(self, X):
        return self.model.predict(X)
