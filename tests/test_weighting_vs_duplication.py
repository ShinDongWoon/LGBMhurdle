import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.utils.preprocessing import ensure_min_positive_ratio
from g2_hurdle.model.classifier import HurdleClassifier
from g2_hurdle.model.regressor import HurdleRegressor


def _make_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = np.zeros(100)
    y[:5] = 1.0
    return X, y


def test_weighting_matches_duplication_classifier_regressor():
    X, y = _make_data()
    X_dup, y_dup, w_dup = ensure_min_positive_ratio(X, y, 0.2, seed=0, use_weights=False)
    X_w, y_w, w_w = ensure_min_positive_ratio(X, y, 0.2, seed=0, use_weights=True)
    # Shapes differ because of duplication
    assert len(X_dup) > len(X_w)
    # Train classifier with both strategies
    clf_params = {"n_estimators": 10, "random_state": 0}
    clf_dup = HurdleClassifier(clf_params)
    clf_dup.fit(X_dup, y_dup)
    clf_w = HurdleClassifier(clf_params)
    clf_w.fit(X_w, y_w, sample_weight=w_w)
    preds_dup = clf_dup.predict_proba(X_w)
    preds_w = clf_w.predict_proba(X_w)
    assert np.mean(np.abs(preds_dup - preds_w)) < 0.03
    # Train regressor with both strategies
    reg_params = {"n_estimators": 10, "random_state": 0, "min_child_samples": 1}
    reg_dup = HurdleRegressor(reg_params)
    reg_dup.fit(X_dup, y_dup)
    reg_w = HurdleRegressor(reg_params)
    reg_w.fit(X_w, y_w, sample_weight=w_w)
    preds_dup_reg = reg_dup.predict(X_w)
    preds_w_reg = reg_w.predict(X_w)
    assert np.allclose(preds_dup_reg, preds_w_reg)
