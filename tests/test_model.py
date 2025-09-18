import numpy as np
from code.model import get_X, get_y, predict_selling_price

feature_vals = ['Maruti', 2017, 82.4, 0, 'Diesel']
labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

def test_input_output_shape():
    X, features = get_X(*feature_vals)
    assert X.shape == (1, 35)
    assert X.dtype == np.float64
    
    y = get_y(X)
    assert y.shape == (1,)

def test_predict_selling_price():
    output = predict_selling_price(*feature_vals, 1)
    assert output[0] in labels
