from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from src.utils import load_model
from src.preprocess import preprocess_data
app = FastAPI()

# Load model once at startup
model = load_model("models/churn_model.pkl")
feature_columns = load_model("models/feature_columns.pkl")
print(type(feature_columns))
print(feature_columns)
class CustomerData(BaseModel):
    Gender : str
    Senior_Citizen : str
    Partner : str
    Dependents : str
    Tenure_Months : int = Field(...,ge=0)
    Phone_Service : str
    Paperless_Billing : str
    Monthly_Charges : int = Field(...,ge=0)
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


@app.post("/predict")
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


    print("Before preprocessing")
    print(data.head())

    preprocessed_data = preprocess_data(data)

    print("After preprocessing")
    print(preprocessed_data.head())

    processed_data = preprocessed_data.reindex(
        columns=feature_columns,
        fill_value=0
    )
    
    print(processed_data.T)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]
    print("Final shape:", processed_data.shape)
    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }