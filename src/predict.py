import joblib
import pandas as pd

model = joblib.load("model/kaggle_model.pkl")


def predict_kaggle(irradiation, temp, module_temp, hour, sunrise_hour, sunset_hour):

    # 🌙 Real night check (dynamic)
    if hour < sunrise_hour or hour > sunset_hour:
        return 0

    X = pd.DataFrame(
        [[irradiation, temp, module_temp, hour]],
        columns=["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour"]
    )

    return model.predict(X)[0]


def get_feature_importance():
    features = ["Irradiance", "Temperature", "Module Temp", "Hour"]
    importance = model.feature_importances_
    return features, importance