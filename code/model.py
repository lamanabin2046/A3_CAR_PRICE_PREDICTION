# # import pickle
# # import numpy as np
# # import logging
# # import mlflow.pyfunc
# # import mlflow
# # import os
# # import time

# # logging.basicConfig(level=logging.INFO)

# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
# # BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
# # FUEL_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_fuel_encoder.model")

# # # Load scaler and encoders
# # scaler = pickle.load(open(SCALAR_PATH, "rb"))
# # brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
# # fuel_encoder = pickle.load(open(FUEL_ENCODER_PATH, "rb"))

# # brand_classes = brand_encoder.categories_[0].tolist()
# # fuel_classes = fuel_encoder.classes_.tolist()

# # # --------------------------
# # # Prepare feature vector
# # # --------------------------
# # def get_X(max_power, year, mileage, fuel, brand):
# #     fuel_val = fuel_encoder.transform([fuel])[0]
# #     numeric_features = np.array([[max_power, year, mileage]])
# #     numeric_scaled = scaler.transform(numeric_features)
# #     brand_encoded = brand_encoder.transform([[brand]])
# #     X = np.hstack([numeric_scaled, [[fuel_val]], brand_encoded])
# #     return X

# # # --------------------------
# # # Load MLflow model
# # # --------------------------
# # def load_model():
# #     MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
# #     mlflow.set_tracking_uri(MLFLOW_URI)

# #     # Optional: support username/password from env
# #     username = os.getenv("MLFLOW_USERNAME")
# #     password = os.getenv("MLFLOW_PASSWORD")
# #     if username and password:
# #         os.environ["MLFLOW_USERNAME"] = username
# #         os.environ["MLFLOW_PASSWORD"] = password

# #     MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
# #     RUN_ID = os.getenv("RUN_ID", None)
# #     model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

# #     for attempt in range(10):
# #         try:
# #             logging.info(f"Loading model from {model_uri} (attempt {attempt + 1}/10)")
# #             _model = mlflow.pyfunc.load_model(model_uri)
# #             logging.info("✅ Model loaded successfully")
# #             return _model
# #         except Exception as e:
# #             logging.warning(f"Failed to load model: {e}")
# #             time.sleep(3)
# #     raise RuntimeError("Failed to load MLflow model after 10 attempts.")

# # # --------------------------
# # # Predict
# # # --------------------------
# # def predict_selling_price(max_power, year, mileage, fuel, brand):
# #     model = load_model()
# #     X = get_X(max_power, year, mileage, fuel, brand)
# #     raw_pred = model.predict(X)[0]
# #     class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
# #     label = class_map[int(raw_pred)]
# #     return raw_pred, label
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

#     username = os.getenv("MLFLOW_USERNAME")
#     password = os.getenv("MLFLOW_PASSWORD")
#     if username and password:
#         os.environ["MLFLOW_TRACKING_USERNAME"] = username
#         os.environ["MLFLOW_TRACKING_PASSWORD"] = password

#     MODEL_NAME = os.getenv("MODEL_NAME", "st125985-a3-model")
#     RUN_ID = os.getenv("RUN_ID", None)

#     if RUN_ID:
#         model_uri = f"runs:/{RUN_ID}/model"
#     else:
#         model_uri = f"models:/{MODEL_NAME}/Production"

#     for attempt in range(10):
#         try:
#             logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/10)")
#             model = mlflow.pyfunc.load_model(model_uri)
#             logging.info("✅ Model loaded successfully")
#             return model
#         except Exception as e:
#             logging.warning(f"Attempt {attempt + 1} failed: {e}")
#             time.sleep(3)

#     raise RuntimeError(f"Failed to load MLflow model after 10 attempts. Tried URI: {model_uri}")

# # Load the model once at import
# try:
#     mlflow_model = load_model()
# except Exception as e:
#     logging.error(f"MLflow model could not be loaded: {e}")
#     mlflow_model = None  # allow app to start, but prediction will fail

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

import pickle
import numpy as np
import logging
import mlflow.pyfunc
import os
import time

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
# Load MLflow model (once)
# --------------------------
def load_mlflow_model():
    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
    USERNAME = os.getenv("MLFLOW_USERNAME") or os.getenv("MLFLOW_TRACKING_USERNAME")
    PASSWORD = os.getenv("MLFLOW_PASSWORD") or os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not MLFLOW_URI or not USERNAME or not PASSWORD:
        logging.error(
            f"MLflow env variables missing: "
            f"MLFLOW_TRACKING_URI={MLFLOW_URI}, "
            f"USERNAME={'set' if USERNAME else 'missing'}, "
            f"PASSWORD={'set' if PASSWORD else 'missing'}"
        )
        return None

    mlflow.set_tracking_uri(MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = PASSWORD

    MODEL_NAME = os.getenv("MODEL_NAME")
    RUN_ID = os.getenv("RUN_ID")
    model_uri = f"runs:/{RUN_ID}/model" if RUN_ID else f"models:/{MODEL_NAME}/Production"

    for attempt in range(10):
        try:
            logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/10)")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ MLflow model loaded successfully")
            return model
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    logging.error(f"Failed to load MLflow model after 10 attempts. Tried URI: {model_uri}")
    return None

# Load once at import
mlflow_model = load_mlflow_model()

# --------------------------
# Predict
# --------------------------
def predict_selling_price(max_power, year, mileage, fuel, brand):
    if mlflow_model is None:
        raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

    X = get_X(max_power, year, mileage, fuel, brand)
    raw_pred = mlflow_model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return raw_pred, label
