import pytest
import numpy as np
from model import get_X, load_model  # MLflow model

# Example input features: brand, year, max_power, mileage, fuel
feature_vals = [82.4, 2017, 19.42, "Diesel", "Maruti"]

def test_model_input_shape():
    """
    Test that the feature vector returned by get_X has the expected shape and type.
    Since brand is one-hot encoded, the total features should be 35.
    """
    X = get_X(*feature_vals)
    
    # Check the shape (1 row, 35 features)
    assert X.shape == (1, 35), f"Expected shape (1, 35), got {X.shape}"
    
    # Check the dtype is numeric
    assert X.dtype == np.float64 or X.dtype == np.float32, f"Expected numeric dtype, got {X.dtype}"

def test_model_output_shape():
    """
    Test that the raw model output has the expected shape: (1,)
    """
    X = get_X(*feature_vals)
    model = load_model()
    
    y_pred = model.predict(X)  # raw numeric prediction
    assert isinstance(y_pred, np.ndarray), f"Expected np.ndarray, got {type(y_pred)}"
    assert y_pred.shape == (1,), f"Expected shape (1,), got {y_pred.shape}"
