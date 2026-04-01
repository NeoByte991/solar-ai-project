import joblib
import pandas as pd

def load_model():
    return joblib.load("model/model.pkl")

def predict(temp, hour):
    model = load_model()
    X = pd.DataFrame([[temp, hour]], columns=["T2M", "hour"])
    return model.predict(X)[0]