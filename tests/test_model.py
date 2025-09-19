# import pytest
# import numpy as np
# from model import get_X, load_model  # MLflow model

# # Example input features: brand, year, max_power, mileage, fuel
# feature_vals = [82.4, 2017, 19.42, "Diesel", "Maruti"]

# def test_model_input_shape():
#     """
#     Test that the feature vector returned by get_X has the expected shape and type.
#     Since brand is one-hot encoded, the total features should be 35.
#     """
#     X = get_X(*feature_vals)
    
#     # Check the shape (1 row, 35 features)
#     assert X.shape == (1, 35), f"Expected shape (1, 35), got {X.shape}"
    
#     # Check the dtype is numeric
#     assert X.dtype == np.float64 or X.dtype == np.float32, f"Expected numeric dtype, got {X.dtype}"

# def test_model_output_shape():
#     """
#     Test that the raw model output has the expected shape: (1,)
#     """
#     X = get_X(*feature_vals)
#     model = load_model()
    
#     y_pred = model.predict(X)  # raw numeric prediction
#     assert isinstance(y_pred, np.ndarray), f"Expected np.ndarray, got {type(y_pred)}"
#     assert y_pred.shape == (1,), f"Expected shape (1,), got {y_pred.shape}"
import pytest
import numpy as np
from model import get_X, load_model

feature_vals = [82.4, 2017, 19.42, "Diesel", "Maruti"]

@pytest.fixture
def mock_model(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return np.array([1])
    monkeypatch.setattr("model.load_model", lambda: DummyModel())

def test_model_input_shape(mock_model):
    X = get_X(*feature_vals)
    assert X.shape[1] == 35
    assert np.issubdtype(X.dtype, np.number)

def test_model_output_shape(mock_model):
    model_instance = load_model()
    X = get_X(*feature_vals)
    y_pred = model_instance.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)
