# code/model.py

import mlflow.pyfunc
import numpy as np
import os

# --------------------------
# MLflow Setup
# --------------------------
# Use MLflow tracking URI from environment variable if defined (recommended for Docker)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

# Load model directly from the registry (Production stage)
MODEL_NAME = "st125985-a3-model"
model_uri = f"models:/{MODEL_NAME}/Production"
model = mlflow.pyfunc.load_model(model_uri)

# --------------------------
# Feature preprocessing
# --------------------------
def get_X(brand, year, km_driven, owner, fuel):
    """
    Convert raw input into ML-ready feature vector.
    Replace this logic with actual preprocessing: scaling, encoding, one-hot, etc.
    """
    X = np.zeros((1, 35))  # Replace 35 with actual feature vector length

    # Example: populate numeric features
    # X[0, 0] = year
    # X[0, 1] = km_driven
    # TODO: Encode brand, fuel, owner, etc. as one-hot or integer encoding

    return X, ["feature1", "feature2", "..."]  # optional feature names

# --------------------------
# Model prediction
# --------------------------
def get_y(X):
    """
    Use the loaded MLflow model to predict numeric output.
    """
    y = model.predict(X)
    return y

# --------------------------
# Dash callback helper
# --------------------------
def predict_selling_price(brand, year, km_driven, owner, fuel, n_clicks=None):
    """
    Convert numeric prediction into class label for Dash callback.
    """
    X, _ = get_X(brand, year, km_driven, owner, fuel)
    y = get_y(X)
    labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

    # Convert numeric prediction to class index safely
    return [labels[int(y[0]) % len(labels)]]
