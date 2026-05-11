from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.config import FORECAST_API_URL, REQUEST_TIMEOUT_SECONDS
from src.features import add_solar_features


FORECAST_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
]


def get_open_meteo_forecast(lat, lon, days=2):
    """Fetch hourly weather and solar forecast features from Open-Meteo."""
    requested_days = min(max(days + 1, 2), 16)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(FORECAST_COLUMNS),
        "forecast_days": requested_days,
        "timezone": "auto",
    }

    response = requests.get(
        FORECAST_API_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise ValueError("Forecast API did not return hourly data")

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "AMBIENT_TEMPERATURE": hourly["temperature_2m"],
            "HUMIDITY": hourly["relative_humidity_2m"],
            "CLOUD_COVER": hourly["cloud_cover"],
            "WIND_SPEED": hourly["wind_speed_10m"],
            "SHORTWAVE_RADIATION": hourly["shortwave_radiation"],
        }
    )

    # Make timestamp naive to match historical data
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # The training dataset uses irradiation as energy over a sampling period.
    # Open-Meteo provides W/m2 for the hour, so kWh/m2 is a closer model input.
    df["IRRADIATION"] = df["SHORTWAVE_RADIATION"].clip(lower=0) / 1000
    df["MODULE_TEMPERATURE"] = (
        df["AMBIENT_TEMPERATURE"] + (df["SHORTWAVE_RADIATION"] / 800 * 20)
    ).round(2)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df = add_solar_features(df)

    utc_offset = int(payload.get("utc_offset_seconds", 0))
    now = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=utc_offset)
    horizon_hours = days * 24
    return df[df["timestamp"] >= now].head(horizon_hours).reset_index(drop=True)
