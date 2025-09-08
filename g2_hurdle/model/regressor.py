
from typing import Optional, Sequence
import numpy as np

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None

class HurdleRegressor:
    def __init__(self, model_params: dict, categorical_feature: Optional[Sequence[str]]=None):
        if LGBMRegressor is None:
            raise ImportError("lightgbm not installed. Please install lightgbm.")
        self.model = LGBMRegressor(**model_params)
        self.categorical_feature = categorical_feature or "auto"

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        mask_tr = (y_train > 0)
        X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
        fit_params = {}
        if X_val is not None and y_val is not None:
            mask_va = (y_val > 0)
            X_va, y_va = X_val[mask_va], y_val[mask_va]
            fit_params["eval_set"] = [(X_va, y_va)]
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        self.model.fit(X_tr, y_tr, **fit_params)

    def predict(self, X):
        return self.model.predict(X)
