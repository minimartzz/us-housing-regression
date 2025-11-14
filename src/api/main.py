"""
FastAPI server to host trained ML model for public use over HTTP
"""
from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3
import os

from src.inference_pipine.inference import predict
from src.batch.run_monthly import run_monthly_predictions

# ---------- Define Variables ----------
S3_BUCKET = os.getenv("S3_BUCKET", "us-housing-regression")
REGION = os.getenv("AWS_REGION", "ap-southeast-1")

s3 = boto3.client("s3", region_name=REGION)


# ---------- Helper Functions ----------
def load_from_s3(key, local_path):
  """
  Download from S3 if not already cached locally.
  Ensures that app always has the latest model and data
  """
  local_path = Path(local_path)
  if not local_path.exists():
    os.makedirs(local_path.parent, exist_ok=True)
    print(f"📥 Downloading {key} from S3…")
    s3.download_file(S3_BUCKET, key, str(local_path))
  return str(local_path)


# ---------- Paths ----------
MODEL_PATH = Path(load_from_s3("models/xgb_model.pkl", "models/xgb_model.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("processed/feature_engineered_train.csv", "data/clean/feature_engineered_train.csv"))

if TRAIN_FE_PATH.exists():
  _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
  TRAIN_FEATURE_COLS = [c for c in _train_cols.columns if c != "price"]
else:
  TRAIN_FEATURE_COLS = None


# ---------- App ----------
# Instantiate app
app = FastAPI(title="Housing Regression API")

# "API test" route
@app.get("/")
def root():
  return {"message": "Housing Regression API is running"}

# Health check API - Checks if the model was downloaded. If yes, return feature details
@app.get("/health")
def health():
  status = Dict[str, Any] = {"model_path": str(MODEL_PATH)}
  if not MODEL_PATH.exists():
    status['status'] = 'unhealthy'
    status['error'] = 'Model not found'
  else:
    status['status'] = 'healthy'
    if TRAIN_FEATURE_COLS:
      status['n_features_expected'] = len(TRAIN_FEATURE_COLS)
  return status

# Prediction endpoint - Receives batch of data and returns results
@app.post("/predict")
def predict_batch(data: List[dict]):
  if not MODEL_PATH.exists():
    return {"error": f"Model not found at {str(MODEL_PATH)}"}
  
  df = pd.DataFrame(data)
  if df.empty():
    return {"error": "No data provided"}
  
  predictions = predict(df, model_path=MODEL_PATH)
  
  response = {"predictions": predictions["predicted_price"].astype(float).tolist()}
  if "actual_price" in predictions.columns:
    response['actual'] = predictions['actual_price'].astype(float).tolist()

  return response

  
# ---------- Batch Predictions ----------
@app.post("/run_batch")
def run_batch():
  preds = run_monthly_predictions()
  return {
    "status": "success",
    "rows_predicted": int(len(preds)),
    "output_dir": "data/predictions/"
  }

# Preview results of latest batch
@app.get("/latest_predictions")
def latest_predictions(n_limit: int=5):
  pred_dir = Path("data/predictions")
  files = sorted(pred_dir.glob("preds_*.csv"))
  if not files:
    return {"error": "No predictions found"}

  latest_file = files[-1]
  df = pd.read_csv(latest_file)
  return {
    "file": latest_file.name,
    "rows": int(len(df)),
    "preview": df.head(limit).to_dict(orient="records")
  }