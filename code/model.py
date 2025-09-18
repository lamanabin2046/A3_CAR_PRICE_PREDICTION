# code/model.py

import mlflow
import mlflow.pyfunc
import numpy as np

# --------------------------
# MLflow Setup
# --------------------------
# Set your MLflow tracking URI (replace with your server address if remote)
mlflow.set_tracking_uri("http://localhost:5001")  # or your MLflow server URL

# Load model from MLflow Model Registry
model_name = "st125985-a3-model"  # your registered model name
model_version = 1               # version to use

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# --------------------------
# Feature preprocessing
# --------------------------
def get_X(brand, year, km_driven, owner, fuel):
    """
    Convert raw input into ML-ready feature vector.
    Example: one-hot encode categorical variables and scale numeric ones.
    """
    X = np.zeros((1, 35))  # replace 35 with your actual feature vector length

    # TODO: Replace this with your actual preprocessing logic
    # Example:
    # X[0, 0] = year
    # X[0, 1] = km_driven
    # Encode brand, fuel, owner, etc. as one-hot

    return X, ["feature1", "feature2", "..."]  # Return feature names optionally


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
# Dash callback
# --------------------------
def predict_selling_price(brand, year, km_driven, owner, fuel, n_clicks):
    """
    Convert numeric prediction into class label for Dash callback.
    """
    X, _ = get_X(brand, year, km_driven, owner, fuel)
    y = get_y(X)
    labels = ["Cheap", "Average", "Expensive", "Very Expensive"]

    # Convert numeric prediction to class index
    return [labels[int(y[0]) % len(labels)]]
