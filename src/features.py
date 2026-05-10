import math

import numpy as np


BASE_FEATURE_COLUMNS = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HUMIDITY",
    "CLOUD_COVER",
    "WIND_SPEED",
    "hour",
    "day_of_year",
    "month",
]

ENGINEERED_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "daylight_flag",
    "solar_position",
    "clear_sky_index",
    "cloud_loss_proxy",
    "effective_irradiation",
    "module_temp_delta",
    "irradiation_rolling_3",
]

FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *ENGINEERED_FEATURE_COLUMNS]


def add_solar_features(df):
    """Add deterministic solar forecasting features used by training and inference."""
    result = df.copy()

    hour_decimal = result["hour"].astype(float)
    if "timestamp" in result.columns:
        hour_decimal = (
            result["timestamp"].dt.hour
            + (result["timestamp"].dt.minute / 60)
            + (result["timestamp"].dt.second / 3600)
        )
    elif "DATE_TIME" in result.columns:
        hour_decimal = (
            result["DATE_TIME"].dt.hour
            + (result["DATE_TIME"].dt.minute / 60)
            + (result["DATE_TIME"].dt.second / 3600)
        )

    result["hour_sin"] = np.sin(2 * math.pi * hour_decimal / 24)
    result["hour_cos"] = np.cos(2 * math.pi * hour_decimal / 24)
    result["daylight_flag"] = ((hour_decimal >= 6) & (hour_decimal <= 18)).astype(int)

    solar_position = np.sin(((hour_decimal - 6) / 12) * math.pi)
    result["solar_position"] = np.clip(solar_position, 0, None)

    result["effective_irradiation"] = result["IRRADIATION"] * (
        1 - (result["CLOUD_COVER"] / 100).clip(lower=0, upper=1)
    )
    result["clear_sky_index"] = (
        result["IRRADIATION"] / result["solar_position"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    result["clear_sky_index"] = result["clear_sky_index"].fillna(0).clip(lower=0, upper=2)
    result["cloud_loss_proxy"] = result["solar_position"] * (
        result["CLOUD_COVER"] / 100
    ).clip(lower=0, upper=1)
    result["module_temp_delta"] = (
        result["MODULE_TEMPERATURE"] - result["AMBIENT_TEMPERATURE"]
    )
    result["irradiation_rolling_3"] = (
        result["IRRADIATION"].rolling(window=3, min_periods=1).mean()
    )

    return result
