from src.train import split_and_load_data,logistic_regression,decision_tree,random_forest,svm,logistic_gridSearchCV,svm_gridSearchCV
from src.evaluate import evaluate_model
import pandas as pd
import joblib

X_train,X_test,y_train,y_test = split_and_load_data(r"D:\ML_LEARNING\PROJECTS\customer-churn-prediction\data\preprocessed_data\churn_processed.csv")

# Baseline models:

# print("\n===== Logistic Regression =====")
# lr = logistic_regression(X_train,y_train)
# evaluate_model(lr,X_test,y_test)

# print()
# print()

# print("\n===== Decision tree =====")
# dt = decision_tree(X_train,y_train)
# evaluate_model(dt,X_test,y_test)

# print()
# print()

# print("\n===== Random forest =====")
# rf = random_forest(X_train,y_train)
# evaluate_model(rf,X_test,y_test)

# print()
# print()

# print("\n===== Support Vector Classifier =====")
# sv = svm(X_train,y_train)
# evaluate_model(sv,X_test,y_test)

print("====== LOGISTIC REGRESSION - TUNED ======")
lr_tuned = logistic_gridSearchCV(X_train,y_train)
print(lr_tuned.best_params_)
print(lr_tuned.best_score_)

print()
print()

best_lr = lr_tuned.best_estimator_

evaluate_model(
    best_lr,
    X_test,
    y_test
)
# print("===== SUPPORT VECTOR CLASSIFIER - TUNED =====")
# svc_tuned = svm_gridSearchCV(X_train,y_train)
# print(svc_tuned.best_params_)
# print(svc_tuned.best_score_)

joblib.dump(
    best_lr,
    "models/churn_model.pkl"
)

print("Model saved successfully!")
