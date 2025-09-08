
from typing import Optional, Sequence
import numpy as np
import warnings

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
        unique_classes = np.unique(y_train_bin)
        if len(unique_classes) < 2:
            raise ValueError("Training data contains only one class.")

        fit_params = {
            "categorical_feature": self.categorical_feature,
        }
        if hasattr(X_train, "columns"):
            fit_params["feature_name"] = list(X_train.columns)

        # Remove constant columns
        if hasattr(X_train, "nunique"):
            nunique = X_train.nunique()
            informative_cols = nunique[nunique > 1].index
            X_train = X_train[informative_cols]
            if X_val is not None:
                X_val = X_val[informative_cols]
            if "feature_name" in fit_params:
                fit_params["feature_name"] = [
                    f for f in fit_params["feature_name"] if f in informative_cols
                ]
            n_features = len(informative_cols)
        else:
            nunique = np.apply_along_axis(lambda col: np.unique(col).size, 0, X_train)
            informative_mask = nunique > 1
            X_train = X_train[:, informative_mask]
            if X_val is not None:
                X_val = X_val[:, informative_mask]
            if "feature_name" in fit_params:
                fit_params["feature_name"] = [
                    f for f, keep in zip(fit_params["feature_name"], informative_mask) if keep
                ]
            n_features = informative_mask.sum()

        if n_features == 0:
            raise ValueError("No informative features after constant-column removal.")

        if X_val is not None and y_val is not None:
            y_val_bin = (y_val > 0).astype(int)
            if np.unique(y_val_bin).size < 2:
                warnings.warn("Validation data contains only one class. eval_set will be ignored.")
            else:
                fit_params["eval_set"] = [(X_val, y_val_bin)]
                if early_stopping_rounds > 0:
                    callbacks = fit_params.get("callbacks", [])
                    callbacks.append(lightgbm.early_stopping(early_stopping_rounds))
                    fit_params["callbacks"] = callbacks

        self.model.fit(X_train, y_train_bin, **fit_params)

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        return p[:,1] if p.ndim == 2 else p
