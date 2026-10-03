import os
import asyncio
import base64
import time
import httpx
import rasterio
import numpy as np
import cv2

# Replace with your active Modal URL
URL = "https://mrcodes06--depthwizard-api.modal.run/api/v1/refine"

# Path to your test image in D:\SIH\SAM2\test_images\nyc_test.tif
tif_path = os.path.join("test_images", "nyc_test.tif")

if not os.path.exists(tif_path):
    raise FileNotFoundError(f"Could not find GeoTIFF at {tif_path}")

# 1. Load and Encode GeoTIFF Once
print(f"Loading {tif_path} into memory...")
with rasterio.open(tif_path) as src:
    image_data = src.read([1, 2, 3])
    image_rgb = np.transpose(image_data, (1, 2, 0))
    if image_rgb.dtype != np.uint8:
        image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
_, buffer = cv2.imencode(".jpg", image_bgr)
img_base64 = base64.b64encode(buffer).decode("utf-8")

# 2. Build Payload with Real Building Candidate Coordinates
payload = {
    "image_base64": img_base64,
    "candidates": [
        {
            "building_id": "nyc-building-candidate-1",
            "bbox": [200.0, 200.0, 400.0, 400.0],
            "positive_points": [[300.0, 300.0]],
            "negative_points": [[150.0, 150.0], [450.0, 450.0]]
        }
    ]
}

# 3. Async Request Execution
async def send_request(client: httpx.AsyncClient, req_id: int):
    start = time.time()
    try:
        resp = await client.post(URL, json=payload, timeout=60.0)
        elapsed_ms = (time.time() - start) * 1000.0
        
        if resp.status_code == 200:
            data = resp.json()
            processing_time = data.get("processing_time_ms", 0.0)
            print(f"Req {req_id:02d} | Status 200 | Total Roundtrip: {elapsed_ms:7.2f} ms | GPU Processing: {processing_time:7.2f} ms")
        else:
            print(f"Req {req_id:02d} | Failed with Status {resp.status_code}: {resp.text}")
    except Exception as err:
        print(f"Req {req_id:02d} | Request Error: {err}")

async def main():
    num_requests = 4  # Adjust the concurrent batch size here
    print(f"Firing {num_requests} concurrent requests with nyc-test.tif to Modal...\n")
    
    # Increase HTTP client timeout to accommodate container cold starts if needed
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [send_request(client, i + 1) for i in range(num_requests)]
        start_total = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_total
        print(f"\nCompleted {num_requests} concurrent requests in {total_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())