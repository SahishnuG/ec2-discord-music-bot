from langchain_core.tools import tool
import httpx

from config.settings import Settings
settings = Settings()
api_key = settings.weather_api_key
# --------- Define a custom tool ----------
@tool
def get_weather(city: str) -> str:
    """Return a real weather report for a given city."""
    real_time_url = "https://api.tomorrow.io/v4/weather/realtime"

    print("Fetching real-time data for location: ", city)
    try:
        with httpx.Client(base_url=real_time_url, timeout=100) as client:
            response = client.get(
                real_time_url,
                params={"location": city, "apikey": api_key}
            )
            if response.status_code != 200:
                print("Failed to fetch real-time data: ", response.text)
                return False, {"error": response.text}
            return True, response.json()
    except Exception as e:
        print("Unexpected error while fetching real-time data: ", e)
        return False, {"error": str(e)}