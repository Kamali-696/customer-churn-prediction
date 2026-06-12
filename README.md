# 📊 Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, based on their demographics, services, billing information, and account details. The project includes full data preprocessing, model training with hyperparameter tuning, a FastAPI, and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| **FastAPI Backend** | https://customer-churn-prediction-duh0.onrender.com |
| **API Docs (Swagger UI)** | https://customer-churn-prediction-duh0.onrender.com/docs |
| **Health Check** | https://customer-churn-prediction-duh0.onrender.com/health |
| **Streamlit Dashboard** | https://customer-churn-prediction-06.streamlit.app/ |

> **Note:** The backend is hosted on Render's free tier. The first request may take **20–30 seconds** due to a cold start.

---

## 📌 Project Overview

Customer churn is one of the most costly problems for subscription businesses. This project uses the **IBM Telco Customer Churn dataset** (from Kaggle) to build and deploy a binary classification model that labels each customer as likely to churn or not.

The end-to-end pipeline covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Training and comparing multiple ML models
- Hyperparameter tuning with GridSearchCV + Stratified K-Fold cross-validation
- Saving the best model and serving it via a FastAPI
- An interactive Streamlit frontend for real-time predictions
- Docker containerization and cloud deployment on Render

---

## ✨ Features

- **EDA Notebook** — visualizations and insights on churn drivers
- **Modular preprocessing pipeline** — encoding, scaling, and one-hot encoding
- **Multiple baseline models** — Logistic Regression, Decision Tree, Random Forest, SVM
- **Hyperparameter tuning** — GridSearchCV optimised on ROC-AUC with 5-fold stratified cross-validation
- **FastAPI** — validated with Pydantic, supports `/predict`, `/health`, and `/docs`
- **Risk classification** — responses include `Low Risk`, `Medium Risk`, or `High Risk`
- **Streamlit Dashboard** — user-friendly form UI with live API integration
- **Docker support** — fully containerised and ready to deploy anywhere
- **Cloud deployment** — live on Render and Streamlit Cloud

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── api/
│   └── app.py                  # FastAPI application (endpoints, Pydantic models)
│
├── data/
│   ├── raw_data/               # Original dataset (not tracked in git)
│   └── preprocessed_data/      # Cleaned and encoded dataset
│
├── models/
│   ├── churn_model.pkl         # Trained Logistic Regression model (pipeline)
│   └── feature_columns.pkl     # Feature column names used during training
│
├── notebooks/
│   ├── churn_eda.ipynb         # Exploratory Data Analysis
│   └── model_interpretation.py # feature importance scripts
│
├── src/
│   ├── preprocess.py           # Data cleaning, encoding, feature engineering
│   ├── train.py                # Model training functions (LR, DT, RF, SVM, GridSearchCV)
│   ├── evaluate.py             # Evaluation metrics (accuracy, ROC-AUC, classification report)
│   └── utils.py                # Save/load model utilities
│
├── streamlit_app.py            # Streamlit frontend dashboard
├── main.py                     # Training entry point — trains and saves the final model
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🧠 Model Details

### Dataset
- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 rows, 33 input features
- **Target:** `Churn Value` (0 = No Churn, 1 = Churn)

### Preprocessing
| Step | Detail |
|---|---|
| Dropped columns | `City`, `Zip Code`, `CustomerID`, `Lat Long`, `Churn Reason`, etc. |
| Binary encoding | `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Paperless Billing` |
| Numeric fix | `Total Charges` coerced to numeric; missing values filled with median |
| One-hot encoding | `Multiple Lines`, `Internet Service`, `Contract`, `Payment Method`, and add-on services |

### Models Evaluated
| Model | Notes |
|---|---|
| Logistic Regression | ✅ Final model |
| Decision Tree | Baseline |
| Random Forest | Baseline |
| Support Vector Classifier (SVC) | Explored with hyperparameter tuning |

### Final Model — Tuned Logistic Regression
- **Scaling:** `RobustScaler` (robust to outliers)
- **Solver:** liblinear (default), `class_weight='balanced'` to handle class imbalance
- **Tuning:** `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10, 100}`
- **Cross-validation:** 5-Fold Stratified K-Fold, optimised on **ROC-AUC**

### 📈 Model Performance (Test Set)

| Metric | Score |
|---|---|
| **Accuracy** | 74.4% |
| **ROC-AUC** | **0.848** |
| **Precision — No Churn (0)** | 0.90 |
| **Recall — No Churn (0)** | 0.73 |
| **F1-Score — No Churn (0)** | 0.81 |
| **Precision — Churn (1)** | 0.51 |
| **Recall — Churn (1)** | 0.78 |
| **F1-Score — Churn (1)** | 0.62 |

**Confusion Matrix** (test set — 1,409 samples):

```
                  Predicted: No Churn   Predicted: Churn
Actual: No Churn        755                  280
Actual: Churn            81                  293
```

> **Key insight:** The model is tuned to catch churners — a high recall of **0.78** on the churn class means 78% of real churners are correctly identified. The strong **ROC-AUC of 0.848** reflects good discriminative ability despite class imbalance (only ~26% of customers actually churn).

### API Response

```json
{
  "prediction": 1,
  "churn_probability": 0.8732,
  "risk_level": "High Risk"
}
```

| Field | Description |
|---|---|
| `prediction` | `0` = No Churn, `1` = Will Churn |
| `churn_probability` | Probability of churning (0.0 – 1.0) |
| `risk_level` | `Low Risk` (< 50%), `Medium Risk` (50–79%), `High Risk` (≥ 80%) |

---

## ⚙️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.13 |
| ML / Data | scikit-learn, pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containerisation | Docker |
| Deployment | Render (backend), Streamlit Cloud (frontend) |

---

## 🏃 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess the data

Place the raw dataset at `data/raw_data/Telco_customer_churn.csv`, then run:

```bash
python src/preprocess.py
```

This saves the processed file to `data/preprocessed_data/churn_processed.csv`.

### 4. Train the model

```bash
python main.py
```

This trains the tuned Logistic Regression model and saves:
- `models/churn_model.pkl`
- `models/feature_columns.pkl`

### 5. Start the FastAPI backend

```bash
uvicorn api.app:app --reload
```

API will be available at `http://127.0.0.1:8000`.  
Interactive docs at `http://127.0.0.1:8000/docs`.

### 6. Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

> The Streamlit app connects to the **live Render API** by default. To point it at your local API, update `API_URL` in `streamlit_app.py` to `http://127.0.0.1:8000/predict`.

---

## 🐳 Docker

### Build the image

```bash
docker build -t churn-api .
```

### Run the container

```bash
docker run -p 8000:8000 churn-api
```

The API will be available at `http://localhost:8000`.

---

## 🔌 API Reference

### `POST /predict`

Accepts a JSON body with customer details and returns a churn prediction.

**Example Request:**

```json
{
  "Gender": "Male",
  "Senior_Citizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure_Months": 12,
  "Phone_Service": "Yes",
  "Paperless_Billing": "Yes",
  "Monthly_Charges": 65,
  "Total_Charges": 780,
  "CLTV": 3500,
  "Multiple_Lines": "No",
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

**Example Response:**

```json
{
  "prediction": 1,
  "churn_probability": 0.8732,
  "risk_level": "High Risk"
}
```

### `GET /health`

Returns `{ "status": "healthy" }` — useful for uptime monitoring.

---

## 📚 What I Learned

This project was built for learning purposes and covers the following concepts end-to-end:

- **Data Analysis** — using pandas and seaborn to explore and understand churn patterns
- **Feature Engineering** — binary mapping, one-hot encoding, handling missing/inconsistent data
- **Model Selection** — training and comparing LR, DT, RF, and SVM with the same evaluation framework
- **Hyperparameter Tuning** — using `GridSearchCV` with `StratifiedKFold` to find the best regularisation parameter
- **ML Pipelines** — wrapping `RobustScaler` + model inside a `sklearn.Pipeline` to prevent data leakage
- **REST APIs** — building a clean, validated API with FastAPI and Pydantic
- **Containerisation** — packaging the app with Docker for reproducible deployment
- **Cloud Deployment** — deploying a Python API to Render for free-tier hosting

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
