import os
import logging
import numpy as np
import mlflow.pyfunc

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(level=logging.INFO)

# --------------------------
# MLflow Setup
# --------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
RUN_ID = os.getenv("RUN_ID", None)

# --------------------------
# Model loading
# --------------------------
model = None
try:
    if RUN_ID:  # load from a specific run
        model_uri = f"runs:/{RUN_ID}/model"
    else:       # load from registered model (Production stage)
        model_uri = f"models:/{MODEL_NAME}/Production"

    logging.info(f"Attempting to load model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info("✅ Model loaded successfully")

except Exception as e:
    logging.exception("❌ Failed to load MLflow model")
    model = None

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

    return X, [f"feature_{i}" for i in range(35)]  # optional feature names

# --------------------------
# Model prediction
# --------------------------
def get_y(X):
    if model is None:
        raise RuntimeError("Model is not loaded. Check logs for details.")
    y = model.predict(X)
    return y

# --------------------------
# Dash callback helper
# --------------------------
def predict_selling_price(brand, year, km_driven, owner, fuel, n_clicks=None):
    """
    Convert numeric prediction into class label for Dash callback.
    """
    try:
        X, _ = get_X(brand, year, km_driven, owner, fuel)
        y = get_y(X)
        labels = ["Cheap", "Average", "Expensive", "Very Expensive"]
        return [labels[int(y[0]) % len(labels)]]
    except Exception as e:
        logging.exception("Prediction failed")
        return ["Error"]
