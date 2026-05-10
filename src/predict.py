import joblib
import pandas as pd

from src.config import LEGACY_SOLAR_MODEL_PATH, SOLAR_MODEL_PATH
from src.features import FEATURE_COLUMNS, add_solar_features


_legacy_model = None


def _load_legacy_model():
    global _legacy_model
    if _legacy_model is None:
        _legacy_model = joblib.load(LEGACY_SOLAR_MODEL_PATH)
    return _legacy_model


def load_forecast_model():
    try:
        bundle = joblib.load(SOLAR_MODEL_PATH)
        return bundle["model"], bundle.get("metrics", {})
    except Exception:
        return _load_legacy_model(), {
            "features": [
                "IRRADIATION",
                "AMBIENT_TEMPERATURE",
                "MODULE_TEMPERATURE",
                "hour",
            ]
        }


def predict_kaggle(irradiation, temp, module_temp, hour, sunrise_hour, sunset_hour):
    model = _load_legacy_model()

    if hour < sunrise_hour or hour > sunset_hour:
        return 0

    X = pd.DataFrame(
        [[irradiation, temp, module_temp, hour]],
        columns=["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour"],
    )

    return model.predict(X)[0]


def get_feature_importance():
    model = _load_legacy_model()
    features = ["Irradiance", "Temperature", "Module Temp", "Hour"]
    importance = model.feature_importances_
    return features, importance


def predict_forecast_frame(forecast_df, max_capacity=None):
    forecast_model, metrics = load_forecast_model()
    features = metrics.get("features", FEATURE_COLUMNS)

    X = add_solar_features(forecast_df)
    for column in features:
        if column not in X.columns:
            X[column] = 0

    predictions = forecast_model.predict(X[features])
    result = forecast_df.copy()
    result["predicted_power"] = predictions.clip(min=0)

    if max_capacity:
        result["predicted_power"] = result["predicted_power"].clip(upper=max_capacity)
        result["capacity_factor"] = result["predicted_power"] / max_capacity

    night_mask = result["IRRADIATION"].fillna(0) <= 0.01
    result.loc[night_mask, "predicted_power"] = 0
    return result


def get_forecast_model_metrics():
    _, metrics = load_forecast_model()
    return metrics


def get_forecast_feature_importance():
    forecast_model, metrics = load_forecast_model()
    features = metrics.get("features", FEATURE_COLUMNS)
    if not hasattr(forecast_model, "feature_importances_"):
        return [], []
    return features, forecast_model.feature_importances_
