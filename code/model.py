
# import pickle
# import numpy as np
# import logging
# import mlflow.pyfunc
# import mlflow
# import os
# import time

# logging.basicConfig(level=logging.INFO)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # --------------------------
# # Paths for scaler and encoders
# # --------------------------
# SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
# BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
# FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

# # Load scaler and encoders
# scaler = pickle.load(open(SCALAR_PATH, "rb"))
# brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
# fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))

# brand_classes = brand_encoder.categories_[0].tolist()
# fuel_classes = fuel_encoder.classes_.tolist()

# # --------------------------
# # Prepare feature vector
# # --------------------------
# def get_X(max_power, year, mileage, fuel, brand):
#     fuel_val = fuel_encoder.transform([fuel])[0]
#     numeric_features = np.array([[max_power, year, mileage]])
#     numeric_scaled = scaler.transform(numeric_features)
#     brand_encoded = brand_encoder.transform([[brand]])
#     X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
#     return X

# # --------------------------
# # Load MLflow model (once)
# # --------------------------
# def load_model():
#     MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
#     mlflow.set_tracking_uri(MLFLOW_URI)

#     username = os.getenv("MLFLOW_TRACKING_USERNAME")
#     password = os.getenv("MLFLOW_TRACKING_PASSWORD")
#     if username and password:
#         os.environ["MLFLOW_TRACKING_USERNAME"] = username
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        
#     if username and password:
#         logging.info(f"Using MLflow credentials from environment: {username}/*****")
#     else:
#         logging.warning("No MLflow credentials found")

#     run_id = os.getenv("RUN_ID")
#     model_name = os.getenv("MODEL_NAME", "st125985-a3-model")
#     model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production"

#     for attempt in range(5):
#         try:
#             logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt+1}/5)")
#             model = mlflow.pyfunc.load_model(model_uri)
#             logging.info("✅ MLflow model loaded successfully")
#             return model
#         except Exception as e:
#             logging.warning(f"Attempt {attempt+1} failed: {type(e).__name__}: {e}")
#             time.sleep(3)

#     raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")

# # Load model at import
# try:
#     mlflow_model = load_model()
# except Exception as e:
#     logging.error(f"MLflow model could not be loaded: {e}")
#     mlflow_model = None

# # --------------------------
# # Predict
# # --------------------------
# def predict_selling_price(max_power, year, mileage, fuel, brand):
#     if mlflow_model is None:
#         raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

#     X = get_X(max_power, year, mileage, fuel, brand)
#     raw_pred = mlflow_model.predict(X)[0]
#     class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
#     label = class_map[int(raw_pred)]
#     return raw_pred, label






import os
import time
import pickle
import numpy as np
import logging
import mlflow
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# Paths for scaler and encoders
# --------------------------
SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

# Load scaler and encoders
scaler = pickle.load(open(SCALAR_PATH, "rb"))
brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))

brand_classes = brand_encoder.categories_[0].tolist()
fuel_classes = fuel_encoder.classes_.tolist()

# --------------------------
# Prepare feature vector
# --------------------------
def get_X(max_power, year, mileage, fuel, brand):
    fuel_val = fuel_encoder.transform([fuel])[0]
    numeric_features = np.array([[max_power, year, mileage]])
    numeric_scaled = scaler.transform(numeric_features)
    brand_encoded = brand_encoder.transform([[brand]])
    X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
    return X

# --------------------------
# Load MLflow model
# --------------------------
def load_model():
    """
    Load MLflow model using credentials from environment.
    Retries up to 5 times in case of failure.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Set credentials in environment before setting tracking URI
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logging.info(f"Using MLflow credentials: {username}/*****")
    else:
        logging.warning("No MLflow credentials found in environment")

    mlflow.set_tracking_uri(mlflow_uri)

    run_id = os.getenv("RUN_ID")
    model_name = os.getenv("MODEL_NAME", "st125985-a3-model")
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production"

    for attempt in range(5):
        try:
            logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/5)")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ MLflow model loaded successfully")
            return model
        except mlflow.exceptions.MlflowException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")

# Load MLflow model at import
try:
    mlflow_model = load_model()
except Exception as e:
    logging.error(f"MLflow model could not be loaded: {e}")
    mlflow_model = None  # Allow app to start but prediction will fail

# --------------------------
# Predict function
# --------------------------
def predict_selling_price(max_power, year, mileage, fuel, brand):
    if mlflow_model is None:
        raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

    X = get_X(max_power, year, mileage, fuel, brand)
    raw_pred = mlflow_model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return raw_pred, label
