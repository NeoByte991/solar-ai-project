import requests

API_KEY = "2bbb0e41470c6e4257fdd15553bc59f1"


def get_weather_by_city(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    res = requests.get(url)
    data = res.json()

    print("Weather API response:", data)

    # 🔥 ADD THIS CHECK
    if data.get("cod") != 200:
        raise Exception("Invalid city name. Please enter a valid city.")

    temp = data["main"]["temp"]
    clouds = data["clouds"]["all"]

    return temp, clouds