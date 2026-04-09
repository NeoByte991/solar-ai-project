import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
gen = pd.read_csv("data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")

# Convert time
gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"])
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

# Merge datasets
df = pd.merge(gen, weather, on=["DATE_TIME", "PLANT_ID"])

# Feature engineering
df["hour"] = df["DATE_TIME"].dt.hour

# Remove missing values
df = df.dropna()

# Select features
X = df[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "hour"]]
y = df["AC_POWER"]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "model/kaggle_model.pkl")

print("Kaggle model trained successfully!")