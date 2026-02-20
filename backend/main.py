from fastapi import FastAPI, Request, Security, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
import pickle
import numpy as np
import pandas as pd
import os
import logging

load_dotenv()
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),   # logs to app.log file
        logging.StreamHandler()           # logs to console
    ],
)
logger = logging.getLogger("fraud_api")
app = FastAPI(title="Insurance Fraud Detection API")
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
# ---------- API Key Validator ----------
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt with key: {api_key}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "status": "error",
                "type": "Unauthorized",
                "message": "Invalid or missing API Key!"
            }
        )
    return api_key
# ---------------------------------------

# ---------- Exception Handlers ----------

# 1. Handle validation errors (wrong input format)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "type": "Validation Error",
            "message": "Invalid input! Please check your fields.",
            "details": exc.errors()
        }
    )

# 2. Handle 404 errors (wrong URL)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    logger.warning(f"404 Not Found: {request.url}")
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "type": "Not Found",
            "message": f"URL {request.url} does not exist!"
        }
    )

# 3. Handle unexpected server errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "type": "Server Error",
            "message": "Something went wrong! Please try again."
        }
    )

# ----------------------------------------

# Load all model files (from parent directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(BASE_DIR, 'models/fraud_detection_model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb'))
label_encoders = pickle.load(open(os.path.join(BASE_DIR, 'models/label_encoders.pkl'), 'rb'))
feature_names = pickle.load(open(os.path.join(BASE_DIR, 'models/feature_names.pkl'), 'rb'))

print("✅ All models loaded successfully!")

# Define input schema
class ClaimInput(BaseModel):
    Deductible: int
    Days_Policy_Accident: str
    PastNumberOfClaims: str
    Days_Policy_Claim: str
    AgeOfPolicyHolder: str
    PolicyType: str
    Fault: str
    VehiclePrice: str
    DriverRating: int
    Make: str
    AccidentArea: str

# Home route
@app.get("/")
def home():
    logger.info("Health check called")
    return {"message": "Insurance Fraud Detection API is running! 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: ClaimInput, api_key: str = Security(verify_api_key)):
    # Convert input to dictionary
    logger.info(f"Prediction request: {data.dict()}")
    input_data = data.dict()

    # Encode categorical features
    for col, value in input_data.items():
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform([str(value)])[0]

    # Create dataframe with correct column order
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    fraud_probability = round(float(probability[1]) * 100, 2)
    legit_probability = round(float(probability[0]) * 100, 2)

    # Risk level
    if fraud_probability > 70:
        risk_level = "HIGH"
        message = "Manual review required!"
    elif fraud_probability > 40:
        risk_level = "MEDIUM"
        message = "Additional verification needed"
    else:
        risk_level = "LOW"
        message = "Can be auto-approved"

    result = {
    "prediction": "FRAUDULENT" if prediction == 1 else "LEGITIMATE",
    "fraud_probability": fraud_probability,
    "legitimate_probability": legit_probability,
    "risk_level": risk_level,
    "message": message
    }

    logger.info(f"Prediction result: {result}")
    return result

