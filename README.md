# ☀️ Solar AI Power Prediction System

## 🚀 Overview
This project predicts solar power generation using real-time weather data and machine learning.

## 🔧 Features
- Real-time weather data (OpenWeather API)
- Solar irradiation estimation
- AI-based power prediction
- Interactive dashboard (Streamlit)

## 🛠 Tech Stack
- Python
- FastAPI
- Streamlit
- Machine Learning (Kaggle dataset)

## ▶️ How to Run

### Backend
uvicorn src.api:app --reload

### Frontend
python -m streamlit run dashboard/app.py

## 📊 Output
- Temperature
- Cloud coverage
- Irradiation
- Predicted solar power