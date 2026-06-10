import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path)
    return df

def drop_columns(df):
    cols_to_drop = [
        "City",
        "Zip Code",
        "CustomerID",
        "Churn Reason",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Churn Label",
        "Churn Score",
        "Country",
        "State",
        "Count"
    ]
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols,inplace = True)
    return df


def encode_features(df):

    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    binary_columns = ["Senior Citizen","Partner","Dependents","Phone Service","Paperless Billing"]
    for bc in binary_columns:
        df[bc] = df[bc].map({
            "Yes":1,
            "No":0
        })

    df['Total Charges'] = pd.to_numeric(df['Total Charges'],errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(df['Total Charges'].median())

    remaining_columns = [ 
        'Multiple Lines', 
        'Internet Service',
        'Online Security', 
        'Online Backup', 
        'Device Protection', 
        'Tech Support',
        'Streaming TV', 
        'Streaming Movies', 
        'Contract', 
        'Payment Method'
    ]
    
    df = pd.get_dummies(data=df,columns=remaining_columns)

    return df

def preprocess_data(df):

    print("Before drop:", df.shape)

    df = drop_columns(df)
    print("After drop:", df.shape)
    print(df.columns)
    df = encode_features(df)
    print("After encoding:", df.shape)
    return df


if __name__ == "__main__":
    df = load_data(r"D:\ML_LEARNING\PROJECTS\customer-churn-prediction\data\raw_data\Telco_customer_churn.csv")

    df = preprocess_data(df)

    output_path = r"D:\ML_LEARNING\PROJECTS\customer-churn-prediction\data\preprocessed_data\churn_processed.csv"

    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")


