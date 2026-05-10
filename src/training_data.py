import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_solar_features

TARGET_COLUMN = "AC_POWER"


def _read_generation(path):
    generation = pd.read_csv(path)
    generation["DATE_TIME"] = pd.to_datetime(
        generation["DATE_TIME"], format="%d-%m-%Y %H:%M"
    )
    return generation


def _read_weather(path):
    weather = pd.read_csv(path)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])
    return weather


def _estimate_cloud_cover(df):
    daylight = df[df["IRRADIATION"] > 0].copy()
    if daylight.empty:
        return pd.Series(100, index=df.index)

    clear_sky = daylight.groupby("hour")["IRRADIATION"].quantile(0.95)
    expected = df["hour"].map(clear_sky).fillna(df["IRRADIATION"].max())
    expected = expected.replace(0, np.nan).fillna(0.001)
    cloud_cover = 100 * (1 - (df["IRRADIATION"] / expected))
    return cloud_cover.clip(lower=0, upper=100)


def build_training_frame(
    generation_path="data/Plant_1_Generation_Data.csv",
    weather_path="data/Plant_1_Weather_Sensor_Data.csv",
    wind_speed=2.0,
):
    """Create a model-ready historical solar dataset from local plant data."""
    generation = _read_generation(generation_path)
    weather = _read_weather(weather_path)

    df = pd.merge(generation, weather, on=["DATE_TIME", "PLANT_ID"])
    df = df.sort_values("DATE_TIME").dropna()
    df["hour"] = df["DATE_TIME"].dt.hour
    df["day_of_year"] = df["DATE_TIME"].dt.dayofyear
    df["month"] = df["DATE_TIME"].dt.month

    df["CLOUD_COVER"] = _estimate_cloud_cover(df)
    df["HUMIDITY"] = (
        70
        - ((df["AMBIENT_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"].mean()) * 2)
        + (df["CLOUD_COVER"] * 0.15)
    ).clip(lower=20, upper=100)
    df["WIND_SPEED"] = wind_speed
    df = add_solar_features(df)

    return df[["DATE_TIME", TARGET_COLUMN, *FEATURE_COLUMNS]].dropna()
