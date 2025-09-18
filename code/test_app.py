# tests/test_model.py

import numpy as np
from code.model import get_X, get_y, predict_selling_price

# Example input
feature_vals = ['Maruti', 2017, 82.4, 0, 'Diesel']
labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

def test_input_shape():
    """
    Test 1: The model takes the expected input
    """
    X, features = get_X(*feature_vals)
    assert X.shape == (1, 35), "Input feature vector shape is incorrect"
    assert X.dtype == np.float64, "Input feature type should be float64"

def test_output_shape():
    """
    Test 2: The output of the model has the expected shape
    """
    X, _ = get_X(*feature_vals)
    y = get_y(X)
    assert y.shape == (1,), "Output shape is incorrect"

    # Also test the Dash callback
    output = predict_selling_price(*feature_vals, 1)
    assert output[0] in labels, "Predicted label is invalid"
