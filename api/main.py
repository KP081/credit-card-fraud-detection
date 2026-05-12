###########################################################
#   |   |   |   |   Import Libraries
###########################################################

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pathlib
from dotenv import load_dotenv

from typing import List, Optional, Dict, Any

# For Save ML Models
import joblib

# Fast API
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Pydantic
from pydantic import BaseModel, Field, validator

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


###########################################################
#   |   |   |   |   Confifuration
###########################################################

# load .env file
load_dotenv()

# Paths
BASE_DIR     = pathlib.Path(__file__).parent.parent
PIPELINE_DIR = str(BASE_DIR / "pipeline")
SERVER_START_TIME = datetime.now()


###########################################################
#   |   |   |   |   Pipeline Class
###########################################################

class FraudDetectionPipeline:

  def __init__(self, model, scaler, threshold=0.5, feature_names=None):
    self.model = model
    self.scaler = scaler
    self.threshold = threshold
    self.feature_names = feature_names
    self.version = "1.0.0"

  def predict(self, transaction, return_details=True):

    # convert data into DataFrame
    if isinstance(transaction, dict):
      transaction = pd.DataFrame([transaction])

    elif isinstance(transaction, pd.Series):
      transaction = transaction.to_frame().T

    # ensure Feature Order
    if self.feature_names is not None:
      transaction = transaction[self.feature_names]

    # Scale
    transaction_scaled = self.scaler.transform(transaction)

    # Predict probability
    fraud_probability = self.model.predict_proba(transaction_scaled)[0][1]

    # apply threshold
    is_fraud = fraud_probability >= self.threshold

    if not return_details:
      return int(is_fraud)

    # assign risk level
    if fraud_probability >= 0.90:
      risk_level = "CRITICAL"
      recommendation = "Block immediately"
    elif fraud_probability >= 0.70:
      risk_level = "HIGH"
      recommendation = "Manual Review - call customer"
    elif fraud_probability >= 0.50:
      risk_level = "MEDIUM"
      recommendation = "OTP Verifications Required"
    elif fraud_probability >= 0.30:
      risk_level = "LOW"
      recommendation = "Monitor - Allow With Logging"
    else:
      risk_level = "SAFE"
      recommendation = "Allow Transaction"

    # return results
    return {
        "is_fraud": bool(is_fraud),
        "confidence": round(fraud_probability, 4),
        "confidence_pct": f"{fraud_probability * 100:.2f}%",
        "risk_level": risk_level,
        "recommendation": recommendation,
        "threshold_used": self.threshold
    }

  def predict_batch(self, transactions_df):

    if self.feature_names is not None:
      transactions_df = transactions_df[self.feature_names]

    scaled = self.scaler.transform(transactions_df)

    probabilites = self.model.predict_proba(scaled)[:, 1]

    predictions = (probabilites >= self.threshold).astype(int)

    # Result DataFrame
    result_df = transactions_df.copy()

    result_df['fraud_probability'] = probabilites.round(4)
    result_df['predicted_fraud'] = predictions
    result_df['risk_level'] = pd.cut(probabilites,
                                     bins=[0, 0.30, 0.50, 0.70, 0.90, 1],
                                     labels=['SAFE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])

    return result_df

  def save(self, directory):

    os.makedirs(directory, exist_ok=True)

    joblib.dump(self.model, os.path.join(directory, 'model.joblib'))
    joblib.dump(self.scaler, os.path.join(directory, 'scaler.joblib'))

    config = {
        'threshold': float(self.threshold),
        'version': self.version,
        'feature_names': self.feature_names,
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(os.path.join(directory, 'config.json'), 'w') as f:
      json.dump(config, f, indent=4)

    print(f"Pipeline saved at {directory}")

  # Load Saved Pipeline
  @classmethod
  def load(cls, directory):

    model = joblib.load(os.path.join(directory, 'model.joblib'))
    scaler = joblib.load(os.path.join(directory, 'scaler.joblib'))

    with open(os.path.join(directory, 'config.json'), 'r') as f:
      config = json.load(f)

    pipeline = cls(
        model = model,
        scaler = scaler,
        threshold = config['threshold'],
        feature_names = config['feature_names']
    )

    return pipeline

  def __repr__(self) -> str:
    return (f"FraudDetectionPipeline (version = {self.version}, "
            f"threshold = {self.threshold}, "
            f"model = {type(self.model).__name__})")


###########################################################
#   |   |   |   |   Load Pipeline at startup
###########################################################

print("\nLoading pipeline...")
pipeline = FraudDetectionPipeline.load(PIPELINE_DIR)
print(f"Pipeline loaded! Threshold: {pipeline.threshold}")


###########################################################
#   |   |   |   |   Pydantic Models
###########################################################

class TransactionInput(BaseModel):

  Time : float = Field(..., gt=0, le=172792, description="Time in seconds since first transaction")

  # PCA Features (V1 to V28)
  V1  : float = Field(..., description="PCA Feature 1")
  V2  : float = Field(..., description="PCA Feature 2")
  V3  : float = Field(..., description="PCA Feature 3")
  V4  : float = Field(..., description="PCA Feature 4")
  V5  : float = Field(..., description="PCA Feature 5")
  V6  : float = Field(..., description="PCA Feature 6")
  V7  : float = Field(..., description="PCA Feature 7")
  V8  : float = Field(..., description="PCA Feature 8")
  V9  : float = Field(..., description="PCA Feature 9")
  V10 : float = Field(..., description="PCA Feature 10")
  V11 : float = Field(..., description="PCA Feature 11")
  V12 : float = Field(..., description="PCA Feature 12")
  V13 : float = Field(..., description="PCA Feature 13")
  V14 : float = Field(..., description="PCA Feature 14")
  V15 : float = Field(..., description="PCA Feature 15")
  V16 : float = Field(..., description="PCA Feature 16")
  V17 : float = Field(..., description="PCA Feature 17")
  V18 : float = Field(..., description="PCA Feature 18")
  V19 : float = Field(..., description="PCA Feature 19")
  V20 : float = Field(..., description="PCA Feature 20")
  V21 : float = Field(..., description="PCA Feature 21")
  V22 : float = Field(..., description="PCA Feature 22")
  V23 : float = Field(..., description="PCA Feature 23")
  V24 : float = Field(..., description="PCA Feature 24")
  V25 : float = Field(..., description="PCA Feature 25")
  V26 : float = Field(..., description="PCA Feature 26")
  V27 : float = Field(..., description="PCA Feature 27")
  V28 : float = Field(..., description="PCA Feature 28")

  # Actual features
  Amount    : float = Field(..., gt=0, description="Transaction amount in USD")
  Time_Hour : float = Field(..., ge=0, le=48, description="Time in hours since first transaction")

class BatchInput(BaseModel):
  transactions : List[TransactionInput] = Field(
      ...,
      description="List of transactions",
      min_items=1,
      max_items=100
  )

class PredictionOutput(BaseModel):

  is_fraud        : bool  = Field(..., description="True = Fraud detected")
  confidence      : float = Field(..., description="Fraud probability (0.0 to 1.0)")
  confidence_pct  : str   = Field(..., description="Fraud probability as percentage")
  risk_level      : str   = Field(..., description="SAFE / LOW / MEDIUM / HIGH / CRITICAL")
  recommendation  : str   = Field(..., description="Suggested action")
  threshold_used  : float = Field(..., description="Threshold applied")
  processing_time : str   = Field(..., description="API response time")

class HealthResponse(BaseModel):

  status      : str = Field(..., description="API status")
  model       : str = Field(..., description="Model name")
  version     : str = Field(..., description="Pipeline version")
  timestamp   : str = Field(..., description="Current server time")
  uptime_info : str = Field(..., description="Server info")

class BatchOutput(BaseModel):
  total_transactions : int                  = Field(..., description="Total processed")
  fraud_count        : int                  = Field(..., description="Frauds detected")
  fraud_rate         : str                  = Field(..., description="Fraud percentage")
  predictions        : List[Dict[str, Any]] = Field(..., description="Per-transaction results")
  processing_time    : str                  = Field(..., description="Total processing time")


###########################################################
#   |   |   |   |   FastAPI App
###########################################################

# create FastAPI app
app = FastAPI(
    title = "Credit Card Fraud Detection API",
    version = "1.0.0",
    docs_url = "/docs",
    redoc_url = "/redoc"
 )

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# Endpoint 1 : Root ( / )
@app.get("/", tags=["Info"])
def root():
  return {
      "message" : "Welcome to Fraud Detection API!",
      "docs"    : "/docs",
  }


# Endpoint 2 : Health Check ( /health )
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():

  uptime = datetime.now() - SERVER_START_TIME

  return HealthResponse(
      status = "🟢 Healthy",
      model = "XGBoost (Optuna Tuned)",
      version = pipeline.version,
      timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      uptime_info = f"Running for {str(uptime).split('.')[0]}"
  )


# Endpoint 3 : Single Prediction ( /predict )
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_fraud(transaction: TransactionInput):

  start_time = time.time()

  try:
    # convert into python dict for pydantic model
    transaction_dict = transaction.dict()

    # pipeline prediction
    result = pipeline.predict(transaction_dict)

    # calculate processing time
    proc_time_ms = (time.time() - start_time) * 1000

    return PredictionOutput(
        is_fraud = result['is_fraud'],
        confidence = result['confidence'],
        confidence_pct = result['confidence_pct'],
        risk_level = result['risk_level'],
        recommendation = result['recommendation'],
        threshold_used = result['threshold_used'],
        processing_time = f"{proc_time_ms:.1f}ms"
    )

  except Exception as e:
    raise HTTPException(
        status_code = 500,
        detail = f"Error : {str(e)}"
    )
  

# Endpoint 4 : Batch Prediction ( /batch )
@app.post("/batch", response_model=BatchOutput, tags=["Prediction"])
def predict_batch(batch: BatchInput):

  start_time = time.time()

  try:
    # convert all transactions into list of dicts
    transactions_list = [t.dict() for t in batch.transactions]

    # create DataFrame
    transactions_df = pd.DataFrame(transactions_list)

    # Batch prediction
    results_df = pipeline.predict_batch(transactions_df)

    # Summary statistics
    fraud_count = int(results_df['predicted_fraud'].sum())
    total       = len(results_df)
    fraud_rate  = fraud_count / total * 100

    # Per-transaction results
    predictions = []
    for i, row in results_df.iterrows():
        predictions.append({
            "transaction_index" : int(i),
            "is_fraud" : bool(row['predicted_fraud']),
            "confidence" : round(float(row['fraud_probability']), 4)
        })

    proc_time_ms = (time.time() - start_time) * 1000

    return BatchOutput(
        total_transactions = total,
        fraud_count = fraud_count,
        fraud_rate = f"{fraud_rate:.2f}%",
        predictions = predictions,
        processing_time = f"{proc_time_ms:.1f}ms"
    )

  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error : {str(e)}")
  
  
# Endpoint 5 : Model Info ( /model-info )
@app.get("/model-info", tags=["Info"])
def model_info():

  return {
    "model_type" : type(pipeline.model).__name__,
    "pipeline_version" : pipeline.version,
    "features_count" : pipeline.model.n_features_in_,
    "feature_names" : pipeline.feature_names,
    "threshold" : pipeline.threshold,
    "model_params" : {
        "n_estimators" : int(pipeline.model.n_estimators),
        "max_depth" : int(pipeline.model.max_depth),
        "learning_rate" : float(pipeline.model.learning_rate),
    }
  }