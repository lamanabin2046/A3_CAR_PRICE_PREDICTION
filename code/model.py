# # model.py
# import pickle
# import numpy as np
# import logging
# import mlflow.pyfunc
# import os
# import time

# logging.basicConfig(level=logging.INFO)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
# BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
# FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

# scaler = pickle.load(open(SCALAR_PATH, "rb"))
# brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
# fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))

# brand_classes = brand_encoder.categories_[0].tolist()
# fuel_classes = fuel_encoder.classes_.tolist()

# MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
# MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME")
# MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")
# MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
# RUN_ID = os.getenv("RUN_ID", None)

# # ---------------------------------------
# # Lazy-load model function
# # ---------------------------------------
# _model = None
# def load_model():
#     global _model
#     if _model is not None:
#         return _model

#     mlflow.set_tracking_uri(MLFLOW_URI)
#     if MLFLOW_USERNAME and MLFLOW_PASSWORD:
#         os.environ["MLFLOW_USERNAME"] = MLFLOW_USERNAME
#         os.environ["MLFLOW_PASSWORD"] = MLFLOW_PASSWORD

#     model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

#     for attempt in range(10):
#         try:
#             logging.info(f"Loading model from {model_uri} (attempt {attempt+1}/10)")
#             _model = mlflow.pyfunc.load_model(model_uri)
#             logging.info("✅ Model loaded successfully")
#             return _model
#         except Exception as e:
#             logging.warning(f"Failed to load model: {e}")
#             time.sleep(3)
#     raise RuntimeError("Failed to load MLflow model after 10 attempts.")

# # ---------------------------------------
# # Prepare feature vector
# # ---------------------------------------
# def get_X(max_power, year, mileage, fuel, brand):
#     fuel_val = fuel_encoder.transform([fuel])[0]
#     numeric_features = np.array([[max_power, year, mileage]])
#     numeric_scaled = scaler.transform(numeric_features)
#     brand_encoded = brand_encoder.transform([[brand]])
#     X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
#     return X

# # ---------------------------------------
# # Prediction
# # ---------------------------------------
# def predict_selling_price(max_power, year, mileage, fuel, brand):
#     try:
#         model = load_model()  # lazy-load here
#         X = get_X(max_power, year, mileage, fuel, brand)
#         raw_pred = model.predict(X)[0]
#         class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
#         label = class_map[int(raw_pred)]
#         logging.info(f"Raw prediction: {raw_pred}, Mapped class: {label}")
#         return raw_pred, label
#     except Exception as e:
#         logging.exception("Prediction failed")
#         return "Error", "Error"
import pickle
import numpy as np
import logging
import mlflow.pyfunc
import os
import time

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ml.brain.cs.ait.ac.th/")
    mlflow.set_tracking_uri(MLFLOW_URI)
    MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
    RUN_ID = os.getenv("RUN_ID", None)
    model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

    for attempt in range(10):
        try:
            logging.info(f"Loading model from {model_uri} (attempt {attempt + 1}/10)")
            _model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ Model loaded successfully")
            return _model
        except Exception as e:
            logging.warning(f"Failed to load model: {e}")
            time.sleep(3)
    raise RuntimeError("Failed to load MLflow model after 10 attempts.")

# --------------------------
# Predict
# --------------------------
def predict_selling_price(max_power, year, mileage, fuel, brand):
    model = load_model()
    X = get_X(max_power, year, mileage, fuel, brand)
    raw_pred = model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return raw_pred, label
