import merged_processor
import numpy as np

# Initialize EE
if not merged_processor.initialize_earth_engine():
    print("Failed to initialize Earth Engine")
    exit(1)

# Test data
geojson_feature = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [77.8746679495518, 27.364271021314064],
            [77.9066583443935, 27.364271021314064],
            [77.9066583443935, 27.3926781149196],
            [77.8746679495518, 27.3926781149196],
            [77.8746679495518, 27.364271021314064]
        ]]
    }
}

date_str = "2018-10-01"

try:
    ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(geojson_feature, date_str)
    if ndvi_data is None or sensor_data is None:
        print("Generation failed: returned None")
    else:
        print(f"NDVI shape: {ndvi_data.shape}")
        print(f"Sensor shape: {sensor_data.shape}")
        np.save('ndvi.npy', ndvi_data)
        np.save('sensor.npy', sensor_data)
        print("Files saved")



except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()



'''
me tujhe roi ke coords dunga 
tu present date ke hissab se date fetch krio
phir coords ki help se aur gee api ka use krke date ke hissab se gee se sensor data aur ndvi_heatmap ka data .npy format me lekr direct usse fast api ko de aur yield predict model.h5 ki help se jo input me ndvi_heatmap and sensor data leta h phir predicted yield ko tresholding ke ssth arrange krke heatmap api (fastapi) me use hota h aur gee wali present ndvi heatmap ko change krta h
'''