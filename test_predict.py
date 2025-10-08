import requests

# Test data from test_generation.py
coordinates = [
    [77.8746679495518, 27.364271021314064],
    [77.9066583443935, 27.364271021314064],
    [77.9066583443935, 27.3926781149196],
    [77.8746679495518, 27.3926781149196],
    [77.8746679495518, 27.364271021314064]
]

url = "http://localhost:8000/predict"
data = {"coordinates": coordinates}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())