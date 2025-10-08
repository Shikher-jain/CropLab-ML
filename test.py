import requests

coordinates = [
    [77.8746679495518, 27.364271021314064],
    [77.9066583443935, 27.364271021314064],
    [77.9066583443935, 27.3926781149196],
    [77.8746679495518, 27.3926781149196],
    [77.8746679495518, 27.364271021314064]
]

response = requests.post("http://localhost:8000/predict", json={
    "coordinates": coordinates,
    "date": "2018-10-01"
})
print(response.json())