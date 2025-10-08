# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
import tensorflow as tf
import joblib
import tempfile
import io
import base64
from utils.preprocess import preprocess_input
from typing import Tuple, Optional
import merged_processor
from pydantic import BaseModel

app = FastAPI(title="Crop Yield Prediction API")

# --- Load model and scaler with error handling ---
import logging
model = None
scaler = None
model_error = None
scaler_error = None

try:
    model = tf.keras.models.load_model("model.h5", compile=False)
except Exception as e:
    model_error = str(e)
    logging.warning(f"Error loading model: {e}")

try:
    scaler = joblib.load("scaler.save")
except Exception as e:
    scaler_error = str(e)
    logging.warning(f"Error loading scaler: {e}")

# Pydantic models
from typing import List
from datetime import datetime

def get_corresponding_date():
    """Fetch corresponding date based on current date"""
    current = datetime.now()
    # Assuming corresponding is current year - 3, October 1st
    year = current.year - 3
    return f"{year}-10-01"

class PredictRequest(BaseModel):
    coordinates: List[List[float]]  # List of [longitude, latitude] points

class HeatmapRequest(BaseModel):
    coordinates: List[List[float]]  # List of [longitude, latitude] points
    t1: float = 3.0  # Threshold for low yield
    t2: float = 4.5  # Threshold for high yield

@app.get("/health")
def health():
    status = {"model_loaded": model is not None, "scaler_loaded": scaler is not None}
    if model_error:
        status["model_error"] = model_error
    if scaler_error:
        status["scaler_error"] = scaler_error
    return status

@app.get("/")
def root():
    return {"message": "🌾 Crop Yield API is up! Send coordinates to /predict for yield predictions or /generate_heatmap for visualization."}

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict crop yield from coordinates using data fetched from Google Earth Engine.

    Takes a list of [longitude, latitude] points, generates NDVI and sensor data automatically,
    and returns the predicted yield value.

    """
    # --- Check model and scaler loaded ---
    if model is None or scaler is None:
        msg = "Model or scaler not loaded. "
        if model_error:
            msg += f"Model error: {model_error}. "
        if scaler_error:
            msg += f"Scaler error: {scaler_error}. "
        raise HTTPException(status_code=500, detail=msg.strip())

    try:
        # --- Initialize Earth Engine ---
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        # --- Get corresponding date ---
        date_str = get_corresponding_date()
        logging.info(f"Using date: {date_str}")

        # --- Generate NDVI and Sensor data ---
        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }
        try:
            ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(
                geojson_dict, date_str
            )
        except Exception as e:
            import traceback
            raise HTTPException(status_code=400, detail=f"Error generating NDVI and sensor data: {str(e)}\n{traceback.format_exc()}")

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate NDVI and sensor data from coordinates (returned None)")

    # Keep arrays in-memory (do not save .npy files). Use these arrays directly
        logging.info(f"Generated NDVI in-memory with shape: {ndvi_data.shape}")
        logging.info(f"Generated Sensor in-memory with shape: {sensor_data.shape}")

        # --- Prepare data for prediction ---
        # NDVI preprocessing
        if ndvi_data.ndim == 2:
            ndvi_processed = ndvi_data[..., np.newaxis]  # (H, W, 1)
        else:
            ndvi_processed = ndvi_data

        ndvi_processed = np.expand_dims(ndvi_processed, axis=0)    # (1, H, W, C)
        ndvi_processed = np.expand_dims(ndvi_processed, axis=1)    # (1, 1, H, W, C)

        # Sensor preprocessing
        if sensor_data.ndim == 2:
            sensor_processed = sensor_data[..., np.newaxis]  # (H, W, 1)
        else:
            sensor_processed = sensor_data

        sensor_processed = np.expand_dims(sensor_processed, axis=0)
        sensor_processed = np.expand_dims(sensor_processed, axis=1)

        # --- Align sensor channels to scaler expectations to avoid feature mismatches ---
        try:
            expected_features = getattr(scaler, "n_features_in_", None)
            if expected_features is not None:
                expected_features = int(expected_features)
        except Exception:
            expected_features = None

        if expected_features is not None:
            current_channels = sensor_processed.shape[-1]
            if current_channels != expected_features:
                logging.warning(f"Sensor channels ({current_channels}) != scaler expected ({expected_features}); trimming or padding to match.")
                if current_channels > expected_features:
                    # trim extra channels
                    sensor_processed = sensor_processed[..., :expected_features]
                else:
                    # pad with zeros for missing channels
                    pad_width = expected_features - current_channels
                    pad_shape = list(sensor_processed.shape[:-1]) + [pad_width]
                    pad = np.zeros(tuple(pad_shape), dtype=sensor_processed.dtype)
                    sensor_processed = np.concatenate([sensor_processed, pad], axis=-1)

        # --- Preprocess inputs ---
        ndvi_processed, sensor_processed = preprocess_input(ndvi_processed, sensor_processed, scaler)

        # --- Predict yield ---
        prediction = model.predict([ndvi_processed, sensor_processed])
        predicted_yield = float(prediction[0][0])  # Single yield value

        # --- Update GEE with predicted yield ---
        import ee
        polygon = merged_processor.create_geometry_from_geojson(geojson_dict)
        yield_image = ee.Image.constant(predicted_yield).clip(polygon)
        asset_id = f"projects/pk07007/assets/predicted_yield_{int(datetime.now().timestamp())}"
        task = ee.batch.Export.image.toAsset(
            image=yield_image,
            description='Predicted Yield',
            assetId=asset_id,
            scale=10,
            region=polygon,
            maxPixels=1e10
        )
        task.start()
        logging.info(f"Started export to GEE asset: {asset_id}")

        return {"predicted_yield": predicted_yield, "gee_asset_id": asset_id, "ndvi_shape": ndvi_data.shape, "sensor_shape": sensor_data.shape}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logging.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/generate_heatmap")
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate yield prediction heatmap overlay from coordinates.

    Takes a list of [longitude, latitude] points, generates NDVI and sensor data,
    predicts yield using the CNN+LSTM model, and returns a color-coded
    heatmap overlay (red/yellow/green based on yield thresholds).
    """
    # --- Check model and scaler loaded ---
    if model is None or scaler is None:
        msg = "Model or scaler not loaded. "
        if model_error:
            msg += f"Model error: {model_error}. "
        if scaler_error:
            msg += f"Scaler error: {scaler_error}. "
        raise HTTPException(status_code=500, detail=msg.strip())

    try:
        # --- Initialize Earth Engine ---
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        # --- Get corresponding date ---
        date_str = get_corresponding_date()
        logging.info(f"Using date: {date_str}")

        # --- Generate NDVI and Sensor data ---
        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }
        ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(
            geojson_dict, date_str
        )

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate NDVI and sensor data from coordinates")

        # --- Prepare data for prediction ---
        # NDVI preprocessing
        if ndvi_data.ndim == 2:
            ndvi_processed = ndvi_data[..., np.newaxis]  # (H, W, 1)
        else:
            ndvi_processed = ndvi_data

        ndvi_processed = np.expand_dims(ndvi_processed, axis=0)    # (1, H, W, C)
        ndvi_processed = np.expand_dims(ndvi_processed, axis=1)    # (1, 1, H, W, C)

        # Sensor preprocessing
        if sensor_data.ndim == 2:
            sensor_processed = sensor_data[..., np.newaxis]  # (H, W, 1)
        else:
            sensor_processed = sensor_data

        sensor_processed = np.expand_dims(sensor_processed, axis=0)
        sensor_processed = np.expand_dims(sensor_processed, axis=1)

        # --- Align sensor channels to scaler expectations to avoid feature mismatches ---
        try:
            expected_features = getattr(scaler, "n_features_in_", None)
            if expected_features is not None:
                expected_features = int(expected_features)
        except Exception:
            expected_features = None

        if expected_features is not None:
            current_channels = sensor_processed.shape[-1]
            if current_channels != expected_features:
                logging.warning(f"Sensor channels ({current_channels}) != scaler expected ({expected_features}); trimming or padding to match.")
                if current_channels > expected_features:
                    # trim extra channels
                    sensor_processed = sensor_processed[..., :expected_features]
                else:
                    # pad with zeros for missing channels
                    pad_width = expected_features - current_channels
                    pad_shape = list(sensor_processed.shape[:-1]) + [pad_width]
                    pad = np.zeros(tuple(pad_shape), dtype=sensor_processed.dtype)
                    sensor_processed = np.concatenate([sensor_processed, pad], axis=-1)

        # --- Preprocess inputs ---
        ndvi_processed, sensor_processed = preprocess_input(ndvi_processed, sensor_processed, scaler)

        # --- Predict yield ---
        prediction = model.predict([ndvi_processed, sensor_processed])[0][0]
        predicted_yield = float(prediction)

        # --- Generate heatmap overlay ---
        heatmap_overlay = merged_processor.create_yield_heatmap_overlay(
            ndvi_data, predicted_yield, request.t1, request.t2
        )

        if heatmap_overlay is None:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap overlay")

        # --- Convert to PNG and return both yield and image (base64) in JSON ---
        import PIL.Image
        img = PIL.Image.fromarray(heatmap_overlay, "RGBA")

        # Save to bytes buffer
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Encode PNG to base64 for JSON transport
        png_bytes = buf.read()
        png_b64 = base64.b64encode(png_bytes).decode('ascii')

        # Compute pixel counts per class (using alpha>0 as mask)
        alpha = heatmap_overlay[..., 3]
        valid_pixels = np.sum(alpha > 0)
        red_pixels = int(np.sum((heatmap_overlay[..., 0] > 0) & (alpha > 0)))
        yellow_pixels = int(np.sum((heatmap_overlay[..., 1] > 0) & (alpha > 0)))
        green_pixels = int(np.sum((heatmap_overlay[..., 2] > 0) & (alpha > 0)))

        response = {
            "predicted_yield": predicted_yield,
            "ndvi_shape": ndvi_data.shape,
            "sensor_shape": sensor_data.shape,
            "image_base64": png_b64,
            "pixel_counts": {
                "valid": int(valid_pixels),
                "red": red_pixels,
                "yellow": yellow_pixels,
                "green": green_pixels
            },
            "thresholds": {"t1": request.t1, "t2": request.t2}
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logging.error(f"Heatmap generation error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@app.post("/export_arrays")
async def export_arrays(request: HeatmapRequest):
    """
    Utility endpoint: generate NDVI and sensor arrays for the provided coordinates
    and return them as a .npz file in-memory (no disk writes).
    """
    try:
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        date_str = get_corresponding_date()

        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }

        ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(geojson_dict, date_str)

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate arrays from coordinates")

        # Pack to in-memory .npz
        buf = io.BytesIO()
        np.savez(buf, ndvi=ndvi_data, sensor=sensor_data)
        buf.seek(0)

        return StreamingResponse(buf, media_type="application/octet-stream",
                                 headers={"Content-Disposition": "attachment; filename=arrays.npz"})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logging.error(f"Export arrays error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Export arrays failed: {str(e)}")