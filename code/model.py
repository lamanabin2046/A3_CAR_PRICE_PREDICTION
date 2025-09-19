# # model.py

# import pickle
# import numpy as np
# import logging
# import mlflow.pyfunc
# import os
# import time

# # --------------------------
# # Logging setup
# # --------------------------
# logging.basicConfig(level=logging.INFO)

# # --------------------------
# # Load scaler and encoders
# # --------------------------
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
# BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
# FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

# scaler = pickle.load(open(SCALAR_PATH, "rb"))
# brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))  # OneHotEncoder
# fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))    # LabelEncoder

# brand_classes = brand_encoder.categories_[0].tolist()
# fuel_classes = fuel_encoder.classes_.tolist()

# # --------------------------
# # MLflow setup
# # --------------------------
# MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ml.brain.cs.ait.ac.th/")
# mlflow.set_tracking_uri(MLFLOW_URI)
# MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
# RUN_ID = os.getenv("RUN_ID", None)

# model = None
# model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

# # Retry loading MLflow model
# for attempt in range(10):
#     try:
#         logging.info(f"Loading model from {model_uri} (attempt {attempt + 1}/10)")
#         model = mlflow.pyfunc.load_model(model_uri)
#         logging.info("✅ Model loaded successfully")
#         break
#     except Exception as e:
#         logging.warning(f"Failed to load model: {e}")
#         time.sleep(3)
# else:
#     raise RuntimeError("Failed to load MLflow model after 10 attempts.")

# # --------------------------
# # Prepare feature vector
# # --------------------------
# def get_X(max_power, year, mileage, fuel, brand):
#     # Fuel label
#     fuel_val = fuel_encoder.transform([fuel])[0]
    
#     # Numeric features
#     numeric_features = np.array([[max_power, year, mileage]])  # exclude fuel
#     numeric_scaled = scaler.transform(numeric_features)  # scale numeric features
    
#     # Combine scaled numeric + fuel + brand one-hot
#     brand_encoded = brand_encoder.transform([[brand]])  # one-hot
#     X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
#     return X


# # --------------------------
# # Model prediction
# # --------------------------
# def predict_selling_price(max_power, year, mileage, fuel, brand):
#     """
#     Predict price class using raw inputs.
#     Returns:
#         raw_pred: numeric prediction from model (0-3)
#         label: string class ("Cheap", "Average", "Expensive", "Very Expensive")
#     """
#     try:
#         X = get_X(max_power, year, mileage, fuel, brand)
#         raw_pred = model.predict(X)[0]  # 0,1,2,3

#         # Map numeric class to descriptive label
#         class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
#         label = class_map[int(raw_pred)]

#         logging.info(f"Raw prediction: {raw_pred}, Mapped class: {label}")
#         return raw_pred, label
#     except Exception as e:
#         logging.exception("Prediction failed")
#         return "Error", "Error"

# model.py

import pickle
import numpy as np
import logging
import mlflow.pyfunc
import os
import time

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(level=logging.INFO)

# --------------------------
# Load scaler and encoders
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

scaler = pickle.load(open(SCALAR_PATH, "rb"))
brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))  # OneHotEncoder
fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))    # LabelEncoder

brand_classes = brand_encoder.categories_[0].tolist()
fuel_classes = fuel_encoder.classes_.tolist()

# --------------------------
# MLflow setup
# --------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USERNAME = os.getenv("MLFLOW_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_PASSWORD")
MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
RUN_ID = os.getenv("RUN_ID", None)

mlflow.set_tracking_uri(MLFLOW_URI)

# Optional: set environment variables for MLflow auth if your server requires username/password
if MLFLOW_USERNAME and MLFLOW_PASSWORD:
    os.environ["MLFLOW_USERNAME"] = MLFLOW_USERNAME
    os.environ["MLFLOW_PASSWORD"] = MLFLOW_PASSWORD

model = None
model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

# Retry loading MLflow model
for attempt in range(10):
    try:
        logging.info(f"Loading model from {model_uri} (attempt {attempt + 1}/10)")
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("✅ Model loaded successfully")
        break
    except Exception as e:
        logging.warning(f"Failed to load model: {e}")
        time.sleep(3)
else:
    raise RuntimeError("Failed to load MLflow model after 10 attempts.")

# --------------------------
# Prepare feature vector
# --------------------------
def get_X(max_power, year, mileage, fuel, brand):
    # Fuel label
    fuel_val = fuel_encoder.transform([fuel])[0]
    
    # Numeric features
    numeric_features = np.array([[max_power, year, mileage]])  # exclude fuel
    numeric_scaled = scaler.transform(numeric_features)  # scale numeric features
    
    # Combine scaled numeric + fuel + brand one-hot
    brand_encoded = brand_encoder.transform([[brand]])  # one-hot
    X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
    return X

# --------------------------
# Model prediction
# --------------------------
def predict_selling_price(max_power, year, mileage, fuel, brand):
    """
    Predict price class using raw inputs.
    Returns:
        raw_pred: numeric prediction from model (0-3)
        label: string class ("Cheap", "Average", "Expensive", "Very Expensive")
    """
    try:
        X = get_X(max_power, year, mileage, fuel, brand)
        raw_pred = model.predict(X)[0]  # 0,1,2,3

        # Map numeric class to descriptive label
        class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
        label = class_map[int(raw_pred)]

        logging.info(f"Raw prediction: {raw_pred}, Mapped class: {label}")
        return raw_pred, label
    except Exception as e:
        logging.exception("Prediction failed")
        return "Error", "Error"
