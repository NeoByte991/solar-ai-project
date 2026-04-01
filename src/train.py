from src.data_fetch import fetch_data
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

lat, lon = 12.97, 77.59

print("Fetching data...")
df = fetch_data(lat, lon)

print("Training model...")

X = df[["T2M", "hour"]]
y = df["IRRADIANCE"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully!")