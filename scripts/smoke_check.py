from datetime import datetime, timedelta

import pandas as pd

from src.config import SOLAR_MODEL_PATH
from src.predict import get_forecast_model_metrics, predict_forecast_frame


def build_sample_forecast():
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []

    for offset in range(24):
        timestamp = start + timedelta(hours=offset)
        hour = timestamp.hour
        daylight = 6 <= hour <= 18
        radiation = max(0, 850 * (1 - abs(hour - 12) / 6)) if daylight else 0

        rows.append(
            {
                "timestamp": timestamp,
                "AMBIENT_TEMPERATURE": 28,
                "HUMIDITY": 62,
                "CLOUD_COVER": 35,
                "WIND_SPEED": 8,
                "SHORTWAVE_RADIATION": radiation,
                "IRRADIATION": radiation / 1000,
                "MODULE_TEMPERATURE": 28 + (radiation / 800 * 20),
                "hour": hour,
                "day_of_year": timestamp.timetuple().tm_yday,
                "month": timestamp.month,
            }
        )

    return pd.DataFrame(rows)


def main():
    forecast = build_sample_forecast()
    prediction = predict_forecast_frame(forecast)
    metrics = get_forecast_model_metrics()

    if prediction.empty:
        raise RuntimeError("Smoke check failed: no prediction rows returned")

    if "predicted_power" not in prediction.columns:
        raise RuntimeError("Smoke check failed: predicted_power column missing")

    if prediction["predicted_power"].isna().any():
        raise RuntimeError("Smoke check failed: prediction contains null values")

    print("Smoke check passed")
    print(f"Model path: {SOLAR_MODEL_PATH}")
    print(f"Rows predicted: {len(prediction)}")
    print(f"Peak predicted power: {prediction['predicted_power'].max():.2f} W")
    if metrics:
        print(f"Training rows: {metrics.get('rows', 'unknown')}")


if __name__ == "__main__":
    main()
