import os
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.era5 import get_wind_speed
from src.training_data import FEATURE_COLUMNS, TARGET_COLUMN, build_training_frame


def train_forecast_model(output_path="model/solar_forecast_model.pkl"):
    try:
        wind_speed = get_wind_speed()
    except Exception:
        wind_speed = 2.0

    df = build_training_frame(wind_speed=wind_speed)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).clip(min=0)
    test_frame = df.iloc[-len(X_test):].copy()
    test_frame["prediction"] = predictions

    daylight = test_frame[test_frame["IRRADIATION"] > 0.02]
    peak = test_frame[test_frame["solar_position"] >= 0.75]

    metrics = {
        "model_type": "RandomForestRegressor",
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "daylight_mae": float(
            mean_absolute_error(daylight[TARGET_COLUMN], daylight["prediction"])
        ) if not daylight.empty else None,
        "peak_mae": float(
            mean_absolute_error(peak[TARGET_COLUMN], peak["prediction"])
        ) if not peak.empty else None,
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_start": df["DATE_TIME"].min().isoformat(),
        "training_end": df["DATE_TIME"].max().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, output_path)
    return metrics


if __name__ == "__main__":
    result = train_forecast_model()
    print("Forecast model trained")
    print(result)
