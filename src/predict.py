import joblib
import pandas as pd   # ✅ ADD THIS

model = joblib.load("model/kaggle_model.pkl")

def predict_kaggle(irradiation, temp, module_temp, hour):

    # 🌙 Night check
    if hour < 6 or hour > 18:
        return 0

    X = pd.DataFrame(
        [[irradiation, temp, module_temp, hour]],
        columns=["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour"]
    )

    return model.predict(X)[0]