import pytest
import numpy as np
from code.model import get_y, get_X, predict_selling_price  # note the 'code.' prefix

# Labels for prediction
labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

def test_model_input_shape():
    X = np.zeros((1, 35))
    try:
        y_pred = get_y(X)
    except Exception as e:
        pytest.fail(f"Model failed on valid input: {e}")

def test_model_output_shape():
    X = np.zeros((5, 35))
    y_pred = get_y(X)
    assert len(y_pred) == 5, f"Expected 5 predictions, got {len(y_pred)}"

def test_predict_selling_price():
    feature_vals = ['Maruti', 2017, 82.4, 0, 'Diesel']
    output = predict_selling_price(*feature_vals)
    assert output[0] in labels
