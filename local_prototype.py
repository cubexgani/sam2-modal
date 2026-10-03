import os
import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Device Selection (Forcing CPU for local prototyping)
device = torch.device("cpu")
print(f"Running SAM 2.1 on: {device}")

# 2. Paths and Configs
# SAM2 repository structure requires path relative to sam2 package or configs
sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
ckpt_path = "third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# Initialize Model and Predictor
sam2_model = build_sam2(sam2_cfg, ckpt_path, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def run_local_refinement(image_np: np.ndarray, bbox: list, pos_points: list, neg_points: list):
    """
    Refines a building candidate using SAM 2.1 image predictor.
    
    :param image_np: RGB image array (H, W, 3)
    :param bbox: [x_min, y_min, x_max, y_max] in pixel-edge coordinates
    :param pos_points: List of [x, y] positive prompt points (e.g., nDSM peaks / roof pixels)
    :param neg_points: List of [x, y] negative prompt points (e.g., road / vegetation)
    """
    # Load image into predictor
    predictor.set_image(image_np)

    # Prepare Prompt Arrays
    box_arr = np.array(bbox, dtype=np.float32) if bbox else None
    
    point_coords = []
    point_labels = []

    for pt in pos_points:
        point_coords.append(pt)
        point_labels.append(1)  # 1 = Foreground (Building)

    for pt in neg_points:
        point_coords.append(pt)
        point_labels.append(0)  # 0 = Background (Non-building)

    coords_arr = np.array(point_coords, dtype=np.float32) if point_coords else None
    labels_arr = np.array(point_labels, dtype=np.int32) if point_labels else None

    # Perform Inference (Predict masks)
    masks, scores, logits = predictor.predict(
        point_coords=coords_arr,
        point_labels=labels_arr,
        box=box_arr,
        multimask_output=False
    )

    best_mask = masks[0].astype(np.uint8)
    best_score = float(scores[0])

    # Convert binary mask to vector polygon using pixel-edge convention
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Squeeze contour array to Nx2: coordinates correspond to pixel edges
        pts = cnt.reshape(-1, 2).tolist()
        polygons.append(pts)

    return {
        "score": best_score,
        "polygons": polygons
    }

if __name__ == "__main__":
    # Mock Image: 200x200 RGB canvas with a mock structure
    mock_image = np.full((200, 200, 3), 120, dtype=np.uint8)
    mock_image[50:150, 50:150] = [200, 80, 80]  # Simulated building footprint

    # Test Candidate Prompts
    test_bbox = [45, 45, 155, 155]       # Bounding box with context margin
    test_pos_points = [[100, 100]]       # Peak/center point
    test_neg_points = [[20, 20], [180, 180]] # Non-building surround points

    print("Executing CPU inference test...")
    result = run_local_refinement(mock_image, test_bbox, test_pos_points, test_neg_points)
    
    print(f"\n--- Refinement Success ---")
    print(f"Predicted SAM 2.1 Score: {result['score']:.4f}")
    print(f"Extracted Outer Ring Vertices: {len(result['polygons'][0]) if result['polygons'] else 0}")
    print(f"First 3 Polygon Vertices (pixel edge): {result['polygons'][0][:3] if result['polygons'] else None}")