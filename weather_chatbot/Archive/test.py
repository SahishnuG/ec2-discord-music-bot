import requests

URL = "http://localhost:8000/generate"

payload = {
    "user_input": "what citys weather did i ask for rn",
    "thread_id": "demo-session-1",
    "stream": False
}

response = requests.post(URL, json=payload)
print("Status:", response.status_code)
print("Response:", response.json())
