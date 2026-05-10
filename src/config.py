import os


APP_NAME = os.getenv("APP_NAME", "Helios Solar Forecast")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Shimoga")
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", APP_NAME)

SOLAR_MODEL_PATH = os.getenv("SOLAR_MODEL_PATH", "model/solar_forecast_model.pkl")
LEGACY_SOLAR_MODEL_PATH = os.getenv("LEGACY_SOLAR_MODEL_PATH", "model/kaggle_model.pkl")

FORECAST_API_URL = os.getenv("FORECAST_API_URL", "https://api.open-meteo.com/v1/forecast")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))
