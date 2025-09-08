
from typing import Optional, Sequence
import numpy as np

try:
    import lightgbm
    from lightgbm import LGBMClassifier
except Exception as e:
    lightgbm = None
    LGBMClassifier = None

class HurdleClassifier:
    def __init__(self, model_params: dict, categorical_feature: Optional[Sequence[str]]=None):
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed. Please install lightgbm.")
        self.model = LGBMClassifier(**model_params)
        self.categorical_feature = categorical_feature or "auto"

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100):
        y_train_bin = (y_train > 0).astype(int)
        fit_params = {
            "categorical_feature": self.categorical_feature,
        }
        if hasattr(X_train, "columns"):
            fit_params["feature_name"] = list(X_train.columns)
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, (y_val > 0).astype(int))]
            if early_stopping_rounds > 0:
                callbacks = fit_params.get("callbacks", [])
                callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
                fit_params["callbacks"] = callbacks
        self.model.fit(X_train, y_train_bin, **fit_params)

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        return p[:,1] if p.ndim == 2 else p
