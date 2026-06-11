from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from src.utils import load_model
from src.preprocess import preprocess_data
from enum import Enum

app = FastAPI(title="Customer Churn Prediction API", description="Predict whether a telecom customer is likely to churn.", version="1.0.0")

# Load model once at startup
model = load_model("models/churn_model.pkl")
feature_columns = load_model("models/feature_columns.pkl")

class GenderEnum(str,Enum):
    male = "Male"
    female = "Female"
class CustomerData(BaseModel):
    Gender : GenderEnum
    Senior_Citizen : str
    Partner : str
    Dependents : str
    Tenure_Months : int = Field(...,ge=0,le=72)
    Phone_Service : str
    Paperless_Billing : str
    Monthly_Charges : int = Field(...,ge=0,le=2000)
    Total_Charges : int = Field(...,ge=0)
    CLTV : int = Field(...,ge=0)
    Multiple_Lines : str
    Internet_Service : str
    Online_Security : str
    Online_Backup : str
    Device_Protection : str
    Tech_Support : str
    Streaming_TV : str
    Streaming_Movies : str
    Contract : str
    Payment_Method : str

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    risk_level: str

@app.post("/predict",response_model=PredictionResponse)

def predict(customer: CustomerData):

    data = pd.DataFrame([{
        "Gender": customer.Gender,
        "Senior Citizen": customer.Senior_Citizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "Tenure Months": customer.Tenure_Months,
        "Phone Service": customer.Phone_Service,
        "Paperless Billing": customer.Paperless_Billing,
        "Monthly Charges": customer.Monthly_Charges,
        "Total Charges": customer.Total_Charges,
        "CLTV": customer.CLTV,
        "Multiple Lines": customer.Multiple_Lines,
        "Internet Service": customer.Internet_Service,
        "Online Security": customer.Online_Security,
        "Online Backup": customer.Online_Backup,
        "Device Protection": customer.Device_Protection,
        "Tech Support": customer.Tech_Support,
        "Streaming TV": customer.Streaming_TV,
        "Streaming Movies": customer.Streaming_Movies,
        "Contract": customer.Contract,
        "Payment Method": customer.Payment_Method,
    }])

    preprocessed_data = preprocess_data(data)

    processed_data = preprocessed_data.reindex(
        columns=feature_columns,
        fill_value=0
    )
    
    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    if probability>=0.8: 
        risk = "High Risk"
    elif probability >= 0.5:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    

    print("Final shape:", processed_data.shape)
    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level" : risk
    }