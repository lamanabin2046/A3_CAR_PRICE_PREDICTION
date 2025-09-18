"""
This file tests the MLflow model in Staging.
If all tests pass, the model can be promoted to Production automatically.
"""

import pytest
import numpy as np
import pandas as pd
import mlflow.pyfunc



# MLflow server and model info
MLFLOW_URI = "http://192.41.170.142:5001"  # replace with your MLflow server IP
MODEL_NAME = "st125985-a3-model"
STAGE = "Staging"

# ------------------------------
# Helper function to load model
# ------------------------------
def load_mlflow_model(stage=STAGE):
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

# ------------------------------
# Test 1: Model loads correctly
# ------------------------------
def test_load_model():
    model = load_mlflow_model()
    assert model is not None, "Failed to load model from MLflow"

# ------------------------------
# Test 2: Model accepts input
# ------------------------------
@pytest.mark.dependency(depends=["test_load_model"])
def test_model_input():
    model = load_mlflow_model()
    # Dummy input matching your feature vector length (35 features)
    X_dummy = np.zeros((1, 35))
    X_dummy = pd.DataFrame(X_dummy, columns=[f"feature_{i}" for i in range(35)])
    pred = model.predict(X_dummy)
    assert pred is not None, "Model did not produce a prediction"

# ------------------------------
# Test 3: Output shape
# ------------------------------
@pytest.mark.dependency(depends=["test_model_input"])
def test_model_output_shape():
    model = load_mlflow_model()
    X_dummy = np.zeros((3, 35))  # 3 samples
    X_dummy = pd.DataFrame(X_dummy, columns=[f"feature_{i}" for i in range(35)])
    pred = model.predict(X_dummy)
    assert len(pred) == 3, f"Expected 3 predictions, got {len(pred)}"
