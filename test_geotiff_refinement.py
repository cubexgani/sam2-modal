import os
import torch
import rasterio
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Setup
device = torch.device("cpu")
sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
ckpt_path = os.path.join("third_party", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
tif_path = os.path.join("test_images", "nyc_test.tif")

# Load model
sam2_model = build_sam2(sam2_cfg, ckpt_path, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# 2. Read GeoTIFF
with rasterio.open(tif_path) as src:
    # Read RGB (Bands 1, 2, 3)
    image_data = src.read([1, 2, 3])
    image_rgb = np.transpose(image_data, (1, 2, 0))

    if image_rgb.dtype != np.uint8:
        image_rgb = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

h, w, _ = image_rgb.shape

# 3. Define a Tight Building Prompt (Targeting a candidate building region)
# Box around a building candidate: [x_min, y_min, x_max, y_max]
bbox = [200, 200, 400, 400]
pos_points = [[300, 300]]             # Positive point inside roof
neg_points = [[150, 150], [450, 450]] # Negative points outside on surrounding terrain

# 4. Run SAM 2.1 Inference
predictor.set_image(image_rgb)

point_coords = np.array(pos_points + neg_points, dtype=np.float32)
point_labels = np.array([1]*len(pos_points) + [0]*len(neg_points), dtype=np.int32)
box_arr = np.array(bbox, dtype=np.float32)

masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=box_arr,
    multimask_output=False
)

best_mask = masks[0].astype(np.uint8)
score = float(scores[0])

# 5. Filter & Polygonize
contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Retain only contours with significant area to remove minor noise
valid_polygons = []
for cnt in contours:
    if cv2.contourArea(cnt) > 50:  # Minimum pixel threshold
        valid_polygons.append(cnt.reshape(-1, 2).tolist())

print(f"\n--- Targeted Refinement Results ---")
print(f"SAM 2.1 Score: {score:.4f}")
print(f"Valid Building Polygons: {len(valid_polygons)}")

# 6. Save Overlay Image for Visual Inspection
overlay = image_rgb.copy()
cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2) # Green polygon contour
cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2) # Blue prompt box

output_img_path = os.path.join("sam2_geotiff_result.png")
cv2.imwrite(output_img_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"Saved visual overlay image to: {output_img_path}")