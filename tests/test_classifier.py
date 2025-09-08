import numpy as np
import warnings
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from g2_hurdle.model.classifier import HurdleClassifier


@pytest.fixture
def simple_model():
    return HurdleClassifier(model_params={'n_estimators': 10})


def test_fit_raises_on_single_class_training(simple_model):
    X_train = np.random.rand(10, 2)
    y_train = np.ones(10)
    with pytest.raises(ValueError, match="Training data contains only one class"):
        simple_model.fit(X_train, y_train)


def test_fit_warns_and_ignores_eval_set_with_single_class_validation(simple_model):
    X_train = np.random.rand(10, 2)
    y_train = np.array([0, 1] * 5)
    X_val = np.random.rand(5, 2)
    y_val = np.zeros(5)
    with warnings.catch_warnings(record=True) as w:
        simple_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=1)
        assert any("Validation data contains only one class" in str(warn.message) for warn in w)
    assert getattr(simple_model.model, "evals_result_", {}) == {}


def test_fit_raises_on_empty_training_data(simple_model):
    X_train = np.empty((0, 2))
    y_train = np.empty((0,))
    with pytest.raises(ValueError):
        simple_model.fit(X_train, y_train)
