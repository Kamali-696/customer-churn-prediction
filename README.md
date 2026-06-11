# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on customer demographics, services, billing information, and account details.

## Features

* Exploratory Data Analysis (EDA)
* Data Preprocessing and Feature Engineering
* Logistic Regression Model
* Hyperparameter Tuning using GridSearchCV
* FastAPI REST API
* Input Validation with Pydantic
* Docker Containerization
* Cloud Deployment on Render

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Docker
* Render

## Live Demo

API Documentation:

```text
https://YOUR-RENDER-URL.onrender.com/docs
```

Health Check:

```text
https://YOUR-RENDER-URL.onrender.com/health
```

## Model Output

The API returns:

* prediction (0 = No Churn, 1 = Churn)
* churn_probability
* risk_level

## Run Locally

```bash
git clone <repository-url>
cd customer-churn-prediction

pip install -r requirements.txt

uvicorn api.app:app --reload
```

## Docker

Build Image:

```bash
docker build -t churn-api .
```

Run Container:

```bash
docker run -p 8000:8000 churn-api
```
