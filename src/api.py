from fastapi import FastAPI
from src.predict import predict_kaggle
from src.weather import get_weather_by_city
from src.irradiance import estimate_irradiation
from datetime import datetime

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Solar AI API Running"}


@app.get("/predict_power")
def predict_power(
    irradiation: float,
    temp: float,
    module_temp: float,
    hour: int
):
    value = predict_kaggle(irradiation, temp, module_temp, hour)
    return {"predicted_power": value}


@app.get("/predict_city")
def predict_city(city: str):
    try:
        hour = datetime.now().hour

        temp, clouds = get_weather_by_city(city)

        irradiation = estimate_irradiation(hour, clouds)

        module_temp = temp + 5

        value = predict_kaggle(irradiation, temp, module_temp, hour)

        return {
            "city": city,
            "temperature": temp,
            "clouds": clouds,
            "irradiation": irradiation,
            "predicted_power": value
        }

    except Exception as e:
        return {"error": str(e)}