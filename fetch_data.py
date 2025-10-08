import merged_processor
import numpy as np
import json

# User's coordinates from admin_farm_data.json
coordinates = [
    [77.85527222628666, 27.299857432730597],
    [77.88724402705607, 27.299857432730597],
    [77.88724402705607, 27.328264526336135],
    [77.85527222628666, 27.328264526336135],
    [77.85527222628666, 27.299857432730597]
]

date_str = "2020-10-01"  # Using a date with available satellite data

# Create GeoJSON feature
geojson_feature = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [coordinates]
    }
}

# Initialize GEE
if not merged_processor.initialize_earth_engine():
    print("Failed to initialize GEE")
    exit(1)

# Generate data
ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(geojson_feature, date_str)

if ndvi_data is None or sensor_data is None:
    print("Failed to generate data")
    exit(1)

# Save as .npy
np.save("ndvi_heatmap.npy", ndvi_data)
np.save("sensor.npy", sensor_data)

print("Saved ndvi_heatmap.npy and sensor.npy")