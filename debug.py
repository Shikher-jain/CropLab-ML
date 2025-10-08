import merged_processor

# Test GEE initialization
print("Initializing GEE...")
success = merged_processor.initialize_earth_engine()
print(f"GEE init: {success}")

# Test coordinates
coordinates = [
    [77.8746679495518, 27.364271021314064],
    [77.9066583443935, 27.364271021314064],
    [77.9066583443935, 27.3926781149196],
    [77.8746679495518, 27.3926781149196],
    [77.8746679495518, 27.364271021314064]
]

geojson_feature = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [coordinates]
    }
}

print("Testing generate_ndvi_and_sensor_npy...")
try:
    ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(geojson_feature, "2018-10-01")
    print(f"NDVI data: {ndvi_data is not None}, shape: {ndvi_data.shape if ndvi_data is not None else None}")
    print(f"Sensor data: {sensor_data is not None}, shape: {sensor_data.shape if sensor_data is not None else None}")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())