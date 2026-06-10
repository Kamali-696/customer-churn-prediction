import pandas as pd

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def split_and_load_data(path):
    """
        1. Loads the dataset by passing the dataset path as a parameter
        2. Splits X and y
        3. Returns the train-test split of the dataset
    """
    df = pd.read_csv(path)
    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    return train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

def logistic_regression(X_train,y_train):
    """
    1. Train a Logistic Regression model using a RobustScaler pipeline by passing the training data as input.
    2. returns the model
    """
    pipeline_lr = Pipeline([
        ("scaler",RobustScaler()),
        ("model",LogisticRegression(class_weight='balanced',max_iter=1000))
    ])
    pipeline_lr.fit(X_train,y_train)
    print("Logistic regression model trained!!")
    return pipeline_lr

def decision_tree(X_train,y_train):
    """
    1. Train a Decision tree model by passing the training data as input.
    2. returns the model
    """
    tree_pipeline = Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ])
    tree_pipeline.fit(X_train,y_train)
    print("Decision Tree model trained!!")
    return tree_pipeline

def random_forest(X_train,y_train):
    """
    1. Train a Random Forest pipeline by passing the training data as input.
    2. returns the model
    """
    forest_pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ])
    forest_pipeline.fit(X_train,y_train)
    print("Random forest model trained!!")
    return forest_pipeline

def svm(X_train,y_train):
    """
    1. Train a SVC model using a RobustScaler pipeline by passing the training data as input.
    2. returns the model
    """
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


#Best models: SVC and Logistic Regression

# Performing Hyperparameter tuning and Stratified K-Fold cross validation for exploring better results
def logistic_gridSearchCV(X_train,y_train):

    """
    this function performs hyperparameter tuning for the logistic regression model and performs stratified k fold cross validation
    
    """
    pipeline = Pipeline([
        ('scaler',RobustScaler()),
        ('model',LogisticRegression(class_weight='balanced',max_iter=1000))
    ])

    param_grid = {
        "model__C": [0.01,0.1,1,10,100]
    }

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_logistic = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs= -1
    )

    grid_logistic.fit(X_train,y_train)

    return grid_logistic


def svm_gridSearchCV(X_train,y_train):
    """
    this function performs hyperparameter tuning for the SVC model and performs stratified k fold cross validation
    
    """
    pipeline = Pipeline([
        ('scaler',RobustScaler()),
        ('model',SVC(class_weight='balanced',random_state=42))
    ])

    param_grid = {
        "model__C": [0.01,0.1,1,10,100],
        "model__kernel": ['rbf','linear']
    }

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_svm = GridSearchCV(
        pipeline,
        param_grid,
        cv=skf,
        scoring="roc_auc",
        n_jobs= -1
    )

    grid_svm.fit(X_train,y_train)

    return grid_svm
