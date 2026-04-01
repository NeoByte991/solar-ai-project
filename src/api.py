from fastapi import FastAPI
from src.predict import predict

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Solar AI API Running"}

@app.get("/predict")
def get_prediction(temp: float, hour: int):
    value = predict(temp, hour)
    return {"predicted_irradiance": float(value)}