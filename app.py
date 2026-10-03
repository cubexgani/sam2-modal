import os
import time
import base64
import torch
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- FastAPI App Setup ---
app = FastAPI(title="DepthWizard SAM 2.1 Refinement Service", version="1.0.0")

# --- Global Predictor Initialization ---
DEVICE = torch.device("cpu") # Will be overridden to "cuda" on Modal
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CKPT_PATH = os.path.join("third_party", "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")

predictor: Optional[SAM2ImagePredictor] = None

@app.on_event("startup")
def load_model():
    global predictor
    if os.path.exists(CKPT_PATH):
        print(f"Loading SAM 2.1 model on {DEVICE}...")
        model = build_sam2(SAM2_CFG, CKPT_PATH, device=DEVICE)
        predictor = SAM2ImagePredictor(model)
        print("SAM 2.1 Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {CKPT_PATH}. Service will fail on refine calls.")

# --- Schema Definitions (depthwizard.roofgraph.v1 compliant) ---
class CandidatePrompt(BaseModel):
    building_id: str
    bbox: List[float] = Field(..., description="[x_min, y_min, x_max, y_max] in pixel-edge space")
    positive_points: List[List[float]] = Field(default_factory=list, description="[[x, y], ...]")
    negative_points: List[List[float]] = Field(default_factory=list, description="[[x, y], ...]")

class RefineRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG image crop or tile")
    candidates: List[CandidatePrompt]

class RoofSection(BaseModel):
    id: str
    polygon: List[List[float]]
    score: float

class BuildingOutput(BaseModel):
    id: str
    footprint_proposal: List[List[float]]
    roof_sections: List[RoofSection]
    scores: dict
    provenance: List[str]

class RefineResponse(BaseModel):
    schema_version: str = "depthwizard.roofgraph.v1"
    buildings: List[BuildingOutput]
    processing_time_ms: float

# --- Endpoints ---
@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": predictor is not None
    }

@app.post("/api/v1/refine", response_model=RefineResponse)
def refine_buildings(request: RefineRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="SAM 2.1 model is not initialized.")

    start_time = time.time()

    # 1. Decode Base64 Image
    try:
        img_bytes = base64.b64decode(request.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode image.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {str(e)}")

    # 2. Load Image into SAM 2 Predictor
    predictor.set_image(image_rgb)

    results = []

    # 3. Process Each Candidate Prompt
    for cand in request.candidates:
        # Build prompt arrays
        point_coords = []
        point_labels = []

        for pt in cand.positive_points:
            point_coords.append(pt)
            point_labels.append(1) # Foreground

        for pt in cand.negative_points:
            point_coords.append(pt)
            point_labels.append(0) # Background

        coords_arr = np.array(point_coords, dtype=np.float32) if point_coords else None
        labels_arr = np.array(point_labels, dtype=np.int32) if point_labels else None
        box_arr = np.array(cand.bbox, dtype=np.float32) if cand.bbox else None

        # Predict Mask
        try:
            masks, scores, _ = predictor.predict(
                point_coords=coords_arr,
                point_labels=labels_arr,
                box=box_arr,
                multimask_output=False
            )
            best_mask = masks[0].astype(np.uint8)
            best_score = float(scores[0])

            # Extract Contours (pixel_edge_column_row)
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Select largest contour as primary footprint
            primary_polygon = []
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if len(largest_cnt) >= 3:
                    primary_polygon = largest_cnt.reshape(-1, 2).tolist()

            building_res = BuildingOutput(
                id=cand.building_id,
                footprint_proposal=primary_polygon,
                roof_sections=[],
                scores={"sam2": best_score},
                provenance=["sam2.1_hiera_base_plus"]
            )
            results.append(building_res)

        except Exception as err:
            # Fallback for individual building candidate failures
            results.append(BuildingOutput(
                id=cand.building_id,
                footprint_proposal=[],
                roof_sections=[],
                scores={"sam2": 0.0},
                provenance=["sam2_failed"]
            ))

    elapsed_ms = (time.time() - start_time) * 1000.0

    return RefineResponse(
        buildings=results,
        processing_time_ms=round(elapsed_ms, 2)
    )