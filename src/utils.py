import joblib

def save_model(model, path):
    joblib.dump(model,path)
    print("Model saved successfully!")

def load_model(path):
    return joblib.load(path)