"""
This file tests the ML model in Staging.
If all tests pass, the model can be promoted to Production automatically.
"""

from utils import load_mlflow
import numpy as np
import pandas as pd
import pytest

# Stage of the model to test
stage = "Staging"


# Test 1: Load the model
def test_load_model():
    model = load_mlflow(stage=stage)
    assert model, "Model could not be loaded from MLflow"


# Test 2: Test model input
@pytest.mark.dependency(depends=["test_load_model"])
def test_model_input():
    model = load_mlflow(stage=stage)
    # Create dummy input with 2 features
    X = np.array([1, 2]).reshape(-1, 2)
    X = pd.DataFrame(X, columns=["x1", "x2"])
    pred = model.predict(X)  # type: ignore
    assert pred is not None, "Model did not produce any prediction"


# Test 3: Test model output shape
@pytest.mark.dependency(depends=["test_model_input"])
def test_model_output():
    model = load_mlflow(stage=stage)
    X = np.array([1, 2]).reshape(-1, 2)
    X = pd.DataFrame(X, columns=["x1", "x2"])
    pred = model.predict(X)  # type: ignore
    assert pred.shape == (1, 1), f"Prediction shape mismatch: {pred.shape}"


# Test 4: Test model coefficients
@pytest.mark.dependency(depends=["test_load_model"])
def test_model_coeff():
    model = load_mlflow(stage=stage)
    assert model.coef_.shape == (1, 2), f"Coefficient shape mismatch: {model.coef_.shape}"  # type: ignore
