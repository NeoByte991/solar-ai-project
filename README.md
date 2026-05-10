# Helios - AI Solar Power Forecasting System

Helios predicts future solar power generation from weather and solar forecast inputs.
It combines local historical plant data, hourly Open-Meteo forecasts, and a trained
machine learning model inside a practical Streamlit dashboard.

## What It Does

- Uses historical plant generation and weather sensor data
- Fetches future hourly forecast inputs
- Predicts solar output for the next hour, next 6 hours, and next 24 hours
- Shows power generation trends across the selected forecast window
- Compares historical actual output with model predictions
- Shows weather impact from irradiation, cloud cover, temperature, humidity, and wind
- Exports hourly forecast predictions as CSV

## Data Sources

- `data/Plant_1_Generation_Data.csv`
- `data/Plant_1_Weather_Sensor_Data.csv`
- `data/era5.nc`
- Open-Meteo Forecast API
- OpenStreetMap Nominatim geocoding

The local Kaggle plant dataset provides generation, irradiance, temperature, and
timestamp fields. Humidity, cloud cover, and wind features are included in the
forecast pipeline. Historical cloud/humidity values are estimated where the local
dataset does not provide them directly.

## Prediction Flow

Forecast weather and solar data
-> ML feature preparation
-> Random Forest prediction model
-> Future solar power forecast
-> Dashboard visualization

## Model

Current model: `RandomForestRegressor`

Model inputs:

- Solar irradiation
- Ambient temperature
- Module temperature
- Humidity
- Cloud cover
- Wind speed
- Hour
- Day of year
- Month
- Engineered solar features such as daylight flag, solar position,
  clear-sky index, cloud-loss proxy, effective irradiation, and rolling
  irradiation

Model output:

- Predicted AC solar power in watts

Current validation metrics are stored inside `model/solar_forecast_model.pkl`
and shown in the dashboard Model tab, including all-hours MAE, daylight MAE,
peak-hour MAE, R2 score, training rows, and training window.

Train or refresh the forecasting model:

```bash
python -m src.train_forecast_model
```

The trained bundle is saved to:

```text
model/solar_forecast_model.pkl
```

## Run Dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

The root `app.py` is the stable deployment entrypoint. It loads the Streamlit
dashboard from `dashboard/app.py`.

## API

The API is optional and meant for local development or future extension. Install
development dependencies first:

```bash
pip install -r requirements-dev.txt
```

Start FastAPI:

```bash
uvicorn src.api:app --reload
```

Forecast endpoint:

```text
/forecast?lat=13.9338&lon=75.573&capacity=3500&days=2
```

## Notes

- The dashboard is intentionally minimal and operational: every chart and metric
  supports monitoring, forecasting, validation, or weather impact analysis.
- The public Streamlit app does not require the raw `data/` CSV files at runtime
  as long as `model/solar_forecast_model.pkl` is included in the repository.
- Historical validation is shown when the local training CSV files are available.
- LSTM is not enabled by default because this project currently avoids heavy deep
  learning dependencies. The pipeline is structured so an LSTM model can be added
  later behind the same forecast dataframe interface.

## Render Deployment

This repository includes deployment-friendly defaults:

- `app.py` as the Streamlit entrypoint
- `render.yaml` for Render web service setup
- `runtime.txt` pinned to Python 3.11
- `.streamlit/config.toml` for headless hosting
- `.env.example` for local environment variables
- `requirements.txt` limited to dashboard runtime dependencies

Render settings if configuring manually:

```text
Build command: pip install -r requirements.txt && python -m scripts.smoke_check
Start command: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Useful environment variables:

```text
DEFAULT_CITY=Shimoga
SOLAR_MODEL_PATH=model/solar_forecast_model.pkl
NOMINATIM_USER_AGENT=Helios Solar Forecast
REQUEST_TIMEOUT_SECONDS=20
```

Run the deployment smoke check locally:

```bash
python -m scripts.smoke_check
```
