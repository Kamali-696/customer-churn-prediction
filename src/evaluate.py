from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score

def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print("ACCURACY: ",accuracy_score(y_test,y_pred))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test,y_pred))
    print("CLASSIFICATION REPORT: ")
    print(classification_report(y_test,y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
