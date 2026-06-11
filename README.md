# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn.

## Project Highlights

* Built a customer churn prediction model using Logistic Regression.
* Performed EDA and feature engineering.
* Tuned hyperparameters using GridSearchCV.
* Developed a FastAPI prediction service.
* Added request and response validation with Pydantic.
* Containerized the application using Docker.

## Features

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Logistic Regression Model
* Hyperparameter Tuning using GridSearchCV
* FastAPI Prediction API
* Input Validation with Pydantic
* Model Persistence using Joblib

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn

## Project Structure

```text
customer-churn-prediction/
│
├── api/
├── src/
├── models/
├── data/
├── requirements.txt
└── README.md
```

## Run the API

```bash
uvicorn api.app:app --reload
```

## API Documentation

Open:

http://127.0.0.1:8000/docs


## Docker Setup

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the Docker container:

```bash
docker run -p 8000:8000 churn-api
```

The API will be available at:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

## Verify the Deployment

Use the following sample request in Swagger UI:

```json
{
  "Gender": "Male",
  "Senior_Citizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Tenure_Months": 2,
  "Phone_Service": "Yes",
  "Paperless_Billing": "Yes",
  "Monthly_Charges": 95,
  "Total_Charges": 190,
  "CLTV": 500,
  "Multiple_Lines": "Yes",
  "Internet_Service": "Fiber optic",
  "Online_Security": "No",
  "Online_Backup": "No",
  "Device_Protection": "No",
  "Tech_Support": "No",
  "Streaming_TV": "Yes",
  "Streaming_Movies": "Yes",
  "Contract": "Month-to-month",
  "Payment_Method": "Electronic check"
}
```

Expected result:

```json
{
  "prediction": 1,
  "churn_probability": 0.925,
  "risk_level": "High Risk"
}
```


## Model Output

The API returns:

* prediction (0 = No Churn, 1 = Churn)
* churn_probability
* risk_level
