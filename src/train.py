import pandas as pd

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def split_and_load_data(path):
    df = pd.read_csv(path)
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    return train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

def logistic_regression(X_train,y_train):
    pipeline_lr = Pipeline([
        ("scaler",RobustScaler()),
        ("model",LogisticRegression(class_weight='balanced',max_iter=1000))
    ])
    pipeline_lr.fit(X_train,y_train)
    print("Logistic regression model trained!!")
    return pipeline_lr

def decision_tree(X_train,y_train):
    tree_pipeline = Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ])
    tree_pipeline.fit(X_train,y_train)
    print("Decision Tree model trained!!")
    return tree_pipeline

def random_forest(X_train,y_train):
    forest_pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ])
    forest_pipeline.fit(X_train,y_train)
    print("Random forest model trained!!")
    return forest_pipeline

def svm(X_train,y_train):
    pipeline_svm = Pipeline([
        ("scaler",RobustScaler()),
        ("model",SVC(
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
    ])
    pipeline_svm.fit(X_train,y_train)
    print("SVC model trained!!")
    return pipeline_svm
