import os
import base64
import requests
import rasterio
import numpy as np
import cv2

# Service URL
BASE_URL = "https://mrcodes06--depthwizard-sam2-service-fastapi-app-dev.modal.run"
BASE_URL = "http://localhost:8000"
REFINE_URL = f"{BASE_URL}/api/v1/refine"
HEALTH_URL = f"{BASE_URL}/api/v1/health"

# Check Health First
health_resp = requests.get(HEALTH_URL)
print(f"Health Check Status: {health_resp.status_code}")
print(f"Health Check Data: {health_resp.json()}\n")

# Load a test GeoTIFF crop or generate a mock image payload
tif_path = os.path.join("test_images", "nyc_test.tif")

if os.path.exists(tif_path):
    with rasterio.open(tif_path) as src:
        image_data = src.read([1, 2, 3])
        image_rgb = np.transpose(image_data, (1, 2, 0))
        if image_rgb.dtype != np.uint8:
            image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
else:
    # Fallback to mock image array
    image_bgr = np.full((300, 300, 3), 120, dtype=np.uint8)
    image_bgr[50:200, 50:200] = [80, 80, 200]

# Encode Image to Base64
_, buffer = cv2.imencode(".jpg", image_bgr)
img_base64 = base64.b64encode(buffer).decode("utf-8")

# Build Request Payload conforming to depthwizard.roofgraph.v1
payload = {
    "image_base64": img_base64,
    "candidates": [
        {
            "building_id": "candidate-17",
            "bbox": [50.0, 50.0, 200.0, 200.0],
            "positive_points": [[125.0, 125.0]],
            "negative_points": [[20.0, 20.0], [250.0, 250.0]]
        }
    ]
}

# Post Request
print("Sending refinement request to Uvicorn server...")
response = requests.post(REFINE_URL, json=payload)

print(f"Response Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("\n--- JSON Refinement Response ---")
    print(f"Schema Version: {data.get('schema_version')}")
    print(f"Processing Time: {data.get('processing_time_ms')} ms")
    
    buildings = data.get("buildings", [])
    if buildings:
        b = buildings[0]
        print(f"Building ID: {b.get('id')}")
        print(f"SAM 2.1 Score: {b.get('scores', {}).get('sam2')}")
        print(f"Footprint Polygon Vertices: {len(b.get('footprint_proposal', []))}")
        print(f"First 3 Vertices: {b.get('footprint_proposal', [])[:3]}")
else:
    print(f"Error Response: {response.text}")