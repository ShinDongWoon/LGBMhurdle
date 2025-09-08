
from typing import Optional, Sequence
import numpy as np
import pandas as pd

try:
    import lightgbm
    from lightgbm import LGBMRegressor
except Exception as e:
    lightgbm = None
    LGBMRegressor = None

class HurdleRegressor:
    def __init__(self, model_params: dict, categorical_feature: Optional[Sequence[str]]=None):
        if LGBMRegressor is None:
            raise ImportError("lightgbm not installed. Please install lightgbm.")
        self.model = LGBMRegressor(**model_params)
        self.categorical_feature = categorical_feature or "auto"

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        mask_tr = (y_train > 0)
        if isinstance(X_train, pd.DataFrame):
            X_tr = X_train.loc[mask_tr]
        else:
            X_tr = X_train[mask_tr]
        y_tr = y_train[mask_tr]

        fit_params = {}
        if X_val is not None and y_val is not None:
            mask_va = (y_val > 0)
            if isinstance(X_val, pd.DataFrame):
                X_va = X_val.loc[mask_va]
            else:
                X_va = X_val[mask_va]
            y_va = y_val[mask_va]
            fit_params["eval_set"] = [(X_va, y_va)]
            if early_stopping_rounds > 0:
                callbacks = fit_params.get("callbacks", [])
                callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
                fit_params["callbacks"] = callbacks
        self.model.fit(X_tr, y_tr, **fit_params)

    def predict(self, X):
        return self.model.predict(X)
