# merged_processor.py
# Merged functionality from ndvi_heatmap.py and main_sensor.py
# Generates NDVI and Sensor .npy files in memory without saving images

import ee
import numpy as np
import json
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SERVICE_ACCOUNT_PATH = 'earth-engine-service-account.json'
DATE_RANGE_START = '2017-10-01'
DATE_RANGE_END = '2018-03-31'
CLOUD_THRESHOLD = 20
MAX_PIXELS = 1e10

# Sensor assets (same as main_sensor.py)
SENSOR_ASSETS = {
    'ECe': 'projects/pk07007/assets/ECe',
    'N': 'projects/pk07007/assets/N',
    'P': 'projects/pk07007/assets/P',
    'pH': 'projects/pk07007/assets/pH',
    'OC': 'projects/pk07007/assets/OC'
}

def initialize_earth_engine():
    """Initialize Google Earth Engine with service account authentication"""
    try:
        with open(SERVICE_ACCOUNT_PATH, 'r') as f:
            service_account = json.load(f)

        logger.info(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")

        credentials = ee.ServiceAccountCredentials(
            service_account['client_email'],
            SERVICE_ACCOUNT_PATH
        )
        ee.Initialize(credentials)
        logger.info("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def create_geometry_from_geojson(geojson_feature):
    """Create GEE geometry from GeoJSON feature - from ndvi_heatmap.py"""
    try:
        coordinates = geojson_feature['geometry']['coordinates']
        geometry_type = geojson_feature['geometry']['type']

        if geometry_type == 'Polygon':
            if isinstance(coordinates, list) and len(coordinates) > 0:
                if isinstance(coordinates[0], list) and len(coordinates[0]) > 0:
                    if isinstance(coordinates[0][0], list) and len(coordinates[0][0]) == 2:
                        return ee.Geometry.Polygon(coordinates)
                    elif isinstance(coordinates[0][0], (int, float)):
                        if len(coordinates[0]) % 2 == 0:
                            reshaped = [[coordinates[0][i], coordinates[0][i+1]] for i in range(0, len(coordinates[0]), 2)]
                            return ee.Geometry.Polygon([reshaped])
            return ee.Geometry.Polygon(coordinates)
        elif geometry_type == 'MultiPolygon':
            return ee.Geometry.MultiPolygon(coordinates)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        return None

def search_satellite_image(polygon, start_date, end_date, cloud_threshold=20):
    """Search for satellite image in GEE - from ndvi_heatmap.py"""
    try:
        logger.info(f"Searching for Sentinel-2 images from {start_date} to {end_date} with cloud threshold {cloud_threshold}%")

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))

        size = collection.size().getInfo()
        logger.info(f"Found {size} images in collection")

        if size == 0:
            logger.info("No images found with current filters, trying without cloud filter...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(polygon)
                         .filterDate(start_date, end_date)
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))
            size = collection.size().getInfo()
            logger.info(f"Found {size} images without cloud filter")

            if size == 0:
                logger.info("No images found, trying with buffer periods...")
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                    extended_start = start_dt - relativedelta(months=3)
                    extended_start_str = extended_start.strftime('%Y-%m-%d')

                    extended_end = end_dt + relativedelta(months=3)
                    extended_end_str = extended_end.strftime('%Y-%m-%d')

                    logger.info(f"Trying extended date range: {extended_start_str} to {extended_end_str}")
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(polygon)
                                 .filterDate(extended_start_str, extended_end_str)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))

                    size = collection.size().getInfo()
                    logger.info(f"Found {size} images in extended date range")

                    if size == 0:
                        logger.info("No images found with cloud filter in extended range, trying without cloud filter...")
                        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                     .filterBounds(polygon)
                                     .filterDate(extended_start_str, extended_end_str)
                                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
                        size = collection.size().getInfo()
                        logger.info(f"Found {size} images in extended range without cloud filter")

                        if size == 0:
                            logger.info("No images found even with extended date range and no cloud filter")
                            return None
                except Exception as date_error:
                    logger.error(f"Error with date extension: {date_error}")
                    return None

        image = collection.first()
        info = image.getInfo()
        if info and 'id' in info:
            logger.info(f"Selected image: {info['id']}")
            return image
        else:
            logger.info("No suitable image found")
            return None
    except Exception as e:
        logger.error(f"Error searching for satellite image: {e}")
        return None

def select_ndvi_bands(image):
    """Select NDVI bands (B8-NIR, B4-Red) from Sentinel-2 image - from ndvi_heatmap.py"""
    try:
        ndvi_bands = image.select(['B8', 'B4']).rename(['NIR', 'Red'])
        return ndvi_bands
    except Exception as e:
        logger.error(f"Error selecting NDVI bands: {e}")
        return None

def calculate_ndvi(image):
    """Calculate NDVI from NIR and Red bands - from ndvi_heatmap.py"""
    try:
        nir = image.select('NIR')
        red = image.select('Red')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return ndvi
    except Exception as e:
        logger.error(f"Error calculating NDVI: {e}")
        return None

def export_image_data(image, region, scale=10, band_names=None):
    """Export image data as numpy array - from ndvi_heatmap.py"""
    try:
        logger.info(f"Exporting image data with scale {scale} meters per pixel...")

        clipped = image.clip(region)

        if band_names is None:
            band_info = image.getInfo()
            if 'bands' in band_info:
                band_names = [band['id'] for band in band_info['bands']]
            else:
                band_names = ['band']

        logger.info(f"Exporting bands: {band_names}")

        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]

        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139

        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)

        logger.info(f"Calculated image dimensions: {width}x{height} pixels")

        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height

        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': band_names,
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x, 'shearX': 0, 'translateX': min_lon,
                    'shearY': 0, 'scaleY': -scale_y, 'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }

        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)

        if pixel_data is not None:
            logger.info(f"Successfully fetched pixel data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch pixel data")
            return None

    except Exception as e:
        logger.error(f"Error exporting image data: {e}")
        return None

def crop_to_256(data):
    """Crop data to 256x256 pixels, centered if possible - from ndvi_heatmap.py"""
    try:
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            if 'NDVI' in data.dtype.names:
                height, width = data['NDVI'].shape
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)

                cropped_data = np.empty((end_y - start_y, end_x - start_x), dtype=data.dtype)
                for name in data.dtype.names:
                    cropped_data[name] = data[name][start_y:end_y, start_x:end_x]
                return cropped_data
        else:
            if data.ndim >= 2:
                height, width = data.shape[:2]
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)

                if data.ndim == 2:
                    return data[start_y:end_y, start_x:end_x]
                else:
                    return data[start_y:end_y, start_x:end_x, :]
        return data
    except Exception as e:
        logger.error(f"Error cropping data: {e}")
        return data

def get_sensor_data(region):
    """Get sensor data for all 5 sensors - from main_sensor.py"""
    try:
        logger.info("Fetching sensor data...")
        sensor_data = {}
        valid_sensors = []

        for sensor_name, asset_id in SENSOR_ASSETS.items():
            try:
                logger.info(f"Loading {sensor_name} sensor data from {asset_id}...")
                sensor_image = ee.Image(asset_id)

                image_info = sensor_image.getInfo()
                if not image_info:
                    logger.warning(f"Failed to load {sensor_name}: No image info")
                    continue

                available_bands = image_info.get('bands', [])
                if len(available_bands) == 0:
                    logger.warning(f"No bands available for {sensor_name}")
                    continue

                actual_band_ids = [band['id'] for band in available_bands[:4]]
                logger.info(f"Available bands for {sensor_name}: {actual_band_ids}")

                selected_image = sensor_image.select(actual_band_ids)
                valid_sensors.append(selected_image)
                sensor_data[sensor_name] = {'bands': actual_band_ids}

                logger.info(f"✅ {sensor_name}: {len(actual_band_ids)} bands loaded")

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {sensor_name}: {e}")
                continue

        if len(valid_sensors) == 0:
            logger.error("No sensor data could be loaded")
            return None

        combined_sensor_image = ee.Image.cat(valid_sensors)
        combined_sensor_image = combined_sensor_image.reproject('EPSG:4326', scale=10).resample('bilinear')
        combined_sensor_image = combined_sensor_image.clip(region)

        logger.info(f"✅ Combined sensor data: {len(valid_sensors)} sensors loaded")
        return combined_sensor_image

    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return None

def export_sensor_data(image, region, scale=10):
    """Export sensor data as numpy array - from main_sensor.py"""
    try:
        logger.info(f"Exporting sensor data with scale {scale} meters per pixel...")

        clipped = image.clip(region)

        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]

        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139

        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)

        if width * height > MAX_PIXELS:
            logger.warning(f"Image size ({width}x{height}) exceeds GEE limit")
            ratio = width / height
            new_width = int(np.sqrt(MAX_PIXELS * ratio))
            new_height = int(MAX_PIXELS / new_width)
            width, height = new_width, new_height
            logger.info(f"Reduced dimensions to {width}x{height}")

        logger.info(f"Calculated image dimensions: {width}x{height} pixels")

        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height

        image_info = image.getInfo()
        band_names = [band['id'] for band in image_info.get('bands', [])]

        if not band_names:
            logger.error("No bands found in sensor image")
            return None

        logger.info(f"Sensor data bands: {band_names}")

        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': band_names,
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x, 'shearX': 0, 'translateX': min_lon,
                    'shearY': 0, 'scaleY': -scale_y, 'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }

        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)

        if pixel_data is not None:
            logger.info(f"Successfully fetched sensor data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch sensor data")
            return None

    except Exception as e:
        logger.error(f"Error exporting sensor data: {e}")
        return None

def combine_ndvi_sensor_data(ndvi_data, sensor_data):
    """Combine NDVI and sensor data into a single 3D array - from main_sensor.py"""
    try:
        logger.info("Combining NDVI and sensor data into 21-band array...")

        if hasattr(ndvi_data, 'dtype') and ndvi_data.dtype.names is not None:
            ndvi_values = ndvi_data['NDVI']
            ndvi_3d = np.expand_dims(ndvi_values, axis=2)
        else:
            if len(ndvi_data.shape) == 2:
                ndvi_3d = np.expand_dims(ndvi_data, axis=2)
            else:
                ndvi_3d = ndvi_data[:, :, :1]

        if hasattr(sensor_data, 'dtype') and sensor_data.dtype.names is not None:
            band_names = sensor_data.dtype.names
            sensor_bands = []
            for band_name in band_names:
                band_data = sensor_data[band_name]
                if len(band_data.shape) == 2:
                    band_3d = np.expand_dims(band_data, axis=2)
                else:
                    band_3d = band_data
                sensor_bands.append(band_3d)
            sensor_3d = np.concatenate(sensor_bands, axis=2)
        else:
            sensor_3d = sensor_data

        combined_array = np.concatenate([ndvi_3d, sensor_3d], axis=2)

        logger.info(f"Combined array shape: {combined_array.shape}")
        logger.info(f"✅ Combined NDVI (1 band) + Sensor ({sensor_3d.shape[2]} bands) = {combined_array.shape[2]} bands total")

        return combined_array

    except Exception as e:
        logger.error(f"Error combining and processing data: {e}")
        return None

def generate_ndvi_and_sensor_npy(geojson_feature, date_str="2018-10-01"):
    """Generate NDVI and Sensor .npy data in memory from GeoJSON feature"""
    try:
        logger.info("Generating NDVI and Sensor data from GeoJSON...")

        # Create Earth Engine polygon
        polygon = create_geometry_from_geojson(geojson_feature)
        if polygon is None:
            logger.error("Failed to create polygon")
            return None, None

        # Parse date and create date range
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = (target_date - timedelta(days=15)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=15)).strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            start_date = date_str
            end_date = date_str

        # Search for satellite image
        logger.info(f"Searching for satellite image between {start_date} and {end_date}")
        image = search_satellite_image(polygon, start_date, end_date)
        if image is None:
            logger.error("No suitable satellite image found")
            return None, None

        # Generate NDVI data
        logger.info("Calculating NDVI...")
        ndvi_bands = select_ndvi_bands(image)
        if ndvi_bands is None:
            logger.error("Failed to select NDVI bands")
            return None, None

        ndvi_image = calculate_ndvi(ndvi_bands)
        if ndvi_image is None:
            logger.error("Failed to calculate NDVI")
            return None, None

        logger.info("Exporting NDVI data...")
        ndvi_data = export_image_data(ndvi_image, polygon, scale=10, band_names=['NDVI'])
        if ndvi_data is None:
            logger.error("Failed to export NDVI data")
            return None, None

        # Keep NDVI full size, no crop

        # Extract NDVI values
        if hasattr(ndvi_data, 'dtype') and ndvi_data.dtype.names is not None:
            if 'NDVI' in ndvi_data.dtype.names:
                ndvi_values = ndvi_data['NDVI']
            else:
                field_name = ndvi_data.dtype.names[0]
                ndvi_values = ndvi_data[field_name]
        else:
            ndvi_values = ndvi_data

        # Ensure proper data type
        if ndvi_values.dtype != np.float32:
            ndvi_values = ndvi_values.astype(np.float32)

        # Get sensor data
        logger.info("Fetching sensor data...")
        sensor_image = get_sensor_data(polygon)
        if sensor_image is None:
            logger.error("Failed to get sensor data")
            return None, None

        logger.info("Exporting sensor data...")
        sensor_data = export_sensor_data(sensor_image, polygon, scale=10)
        if sensor_data is None:
            logger.error("Failed to export sensor data")
            return None, None

        # Prepare sensor data as 3D array (full size, no crop)
        if hasattr(sensor_data, 'dtype') and sensor_data.dtype.names is not None:
            band_names = sensor_data.dtype.names
            sensor_bands = []
            for band_name in band_names:
                band_data = sensor_data[band_name]
                if len(band_data.shape) == 2:
                    band_3d = np.expand_dims(band_data, axis=2)
                else:
                    band_3d = band_data
                sensor_bands.append(band_3d)
            sensor_3d = np.concatenate(sensor_bands, axis=2)
        else:
            sensor_3d = sensor_data

        logger.info(f"✅ Successfully generated NDVI data with shape: {ndvi_values.shape}")
        logger.info(f"✅ Successfully generated sensor data with shape: {sensor_3d.shape}")

        return ndvi_values, sensor_3d

    except Exception as e:
        logger.error(f"Error generating NDVI and Sensor data: {e}")
        return None, None

def create_yield_heatmap_overlay(ndvi_data, predicted_yield, t1=30, t2=50):
    """
    Create a heatmap overlay with red, yellow, and green masks based on predicted yield.
    Uses NDVI as base image and applies color coding based on yield thresholds.

    Args:
        ndvi_data: 2D NDVI array
        predicted_yield: Predicted yield value (float)
        t1: Threshold 1 for low yield (default: 30)
        t2: Threshold 2 for high yield (default: 50)

    Returns:
        RGBA numpy array for PNG overlay
    """
    try:
        import numpy as np
        # Ensure NDVI is float
        nd = np.array(ndvi_data, dtype=float)
        if nd.ndim == 3 and nd.shape[2] == 1:
            nd = nd[..., 0]
        if nd.ndim != 2:
            nd = np.squeeze(nd)
        h, w = nd.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)  # default transparent

        # Mask valid NDVI
        valid_mask = np.isfinite(nd)
        if not np.any(valid_mask):
            return rgba
        # Thresholds
        v1 = t1
        v2 = t2
        # Classify
        low_mask = valid_mask & (nd < v1)
        mid_mask = valid_mask & (nd >= v1) & (nd < v2)
        high_mask = valid_mask & (nd >= v2)
        alpha_val = 200  # overlay alpha
        # Pure Red for low yield
        rgba[low_mask, 0] = 255   # R
        rgba[low_mask, 1] = 0     # G
        rgba[low_mask, 2] = 0     # B
        rgba[low_mask, 3] = alpha_val
        # Pure Yellow for mid yield
        rgba[mid_mask, 0] = 255
        rgba[mid_mask, 1] = 255
        rgba[mid_mask, 2] = 0
        rgba[mid_mask, 3] = alpha_val
        # Pure Green for high yield
        rgba[high_mask, 0] = 0
        rgba[high_mask, 1] = 255
        rgba[high_mask, 2] = 0
        rgba[high_mask, 3] = alpha_val
        return rgba
    except Exception as e:
        logger.error(f"Error creating yield heatmap overlay: {e}")
        return None