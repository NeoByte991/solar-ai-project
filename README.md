# Helios - Solar Power Prediction System

## Overview
Helios is an AI-based solar power prediction system that estimates real-time solar energy generation using live data and machine learning.

It combines multiple data sources with a trained model to provide accurate and explainable predictions.

---

## Features

- Real-time weather data (temperature and cloud cover)
- Solar irradiance from NASA POWER API
- Wind data integration (ERA5)
- AI-based power prediction (Random Forest)
- Dynamic sunrise and sunset calculation
- Interactive dashboard (Streamlit)
- Feature importance (model explainability)
- CSV data export

---

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn (Random Forest)
- NASA POWER API
- OpenWeather API
- ERA5 Dataset
- Altair

---

## How It Works

1. User enters a city  
2. City is converted to latitude and longitude  
3. Weather data is fetched in real time  
4. NASA API provides solar irradiance  
5. Irradiance is adjusted using cloud cover  
6. Machine learning model predicts solar power output  

---

## Model Details

- Algorithm: Random Forest Regressor  
- Inputs:
  - Irradiance  
  - Temperature  
  - Module Temperature  
  - Hour of day  
- Output:
  - Predicted solar power  

---

## How to Run

### Install dependencies

pip install -r requirements.txt

### Run the application

streamlit run dashboard/app.py

---

## Output

- Temperature  
- Cloud coverage  
- Wind speed  
- Solar irradiation (raw and adjusted)  
- Predicted solar power  
- Power trend graph  
- Feature importance chart  

---

## Notes

- Large datasets (ERA5, NASA CSV) are not included due to size limitations  
- The model is trained using a Kaggle solar plant dataset  

---

## Author

Helios Project • Batch 6
