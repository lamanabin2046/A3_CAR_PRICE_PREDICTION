import pytest
import numpy as np
from model import get_y, get_X, predict_selling_price

# Labels for prediction
labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

# -------- Test 1: Input shape check --------
def test_model_input_shape():
    """
    The model should accept a 2D numpy array as input.
    """
    X = np.zeros((1, 35))  # Shape must match model's expected input
    try:
        y_pred = get_y(X)
    except Exception as e:
        pytest.fail(f"Model failed on valid input: {e}")

# -------- Test 2: Output shape check --------
def test_model_output_shape():
    """
    The model output should have the same number of rows as input.
    """
    X = np.zeros((5, 35))
    y_pred = get_y(X)

    assert len(y_pred) == 5, f"Expected 5 predictions, got {len(y_pred)}"

# -------- Test 3: predict_selling_price callback --------
def test_predict_selling_price():
    feature_vals = ['Maruti', 2017, 82.4, 0, 'Diesel']
    output = predict_selling_price(*feature_vals)
    assert output[0] in labels, f"Unexpected class label: {output[0]}"
