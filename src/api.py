from fastapi import FastAPI
from src.forecast import get_open_meteo_forecast
from src.predict import predict_forecast_frame, predict_kaggle
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
    hour: int,
    sunrise_hour: int = 6,
    sunset_hour: int = 18,
):
    value = predict_kaggle(irradiation, temp, module_temp, hour, sunrise_hour, sunset_hour)
    return {"predicted_power": value}


@app.get("/predict_city")
def predict_city(city: str):
    try:
        hour = datetime.now().hour

        temp, clouds = get_weather_by_city(city)

        irradiation = estimate_irradiation(hour, clouds)

        module_temp = temp + 5

        value = predict_kaggle(irradiation, temp, module_temp, hour, 6, 18)

        return {
            "city": city,
            "temperature": temp,
            "clouds": clouds,
            "irradiation": irradiation,
            "predicted_power": value
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def forecast(lat: float, lon: float, capacity: float = 3500, days: int = 2):
    try:
        frame = get_open_meteo_forecast(lat, lon, days=days)
        prediction = predict_forecast_frame(frame, max_capacity=capacity)
        prediction = prediction.copy()
        prediction["timestamp"] = prediction["timestamp"].astype(str)
        return prediction[
            [
                "timestamp",
                "predicted_power",
                "IRRADIATION",
                "AMBIENT_TEMPERATURE",
                "HUMIDITY",
                "CLOUD_COVER",
                "WIND_SPEED",
            ]
        ].to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
