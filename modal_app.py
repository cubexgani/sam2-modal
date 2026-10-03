import os
import time
import base64
import modal
import numpy as np
import cv2
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Modal Container Image Definition ---
sam2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "opencv-python-headless",
        "pillow",
        "shapely",
        "rasterio"
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /opt/sam2",
        "cd /opt/sam2 && pip install -e ."
    )
)

# --- Define Modal App ---
app = modal.App("depthwizard-sam2-service")
weights_volume = modal.Volume.from_name("depthwizard-sam2-weights", create_if_missing=True)

# --- Schema Definitions ---
class CandidatePrompt(BaseModel):
    building_id: str
    bbox: List[float]
    positive_points: List[List[float]] = Field(default_factory=list)
    negative_points: List[List[float]] = Field(default_factory=list)

class RefineRequest(BaseModel):
    image_base64: str
    candidates: List[CandidatePrompt]

class BuildingOutput(BaseModel):
    id: str
    footprint_proposal: List[List[float]]
    roof_sections: List[dict] = Field(default_factory=list)
    scores: dict
    provenance: List[str]

class RefineResponse(BaseModel):
    schema_version: str = "depthwizard.roofgraph.v1"
    buildings: List[BuildingOutput]
    processing_time_ms: float

# --- Modal Class with Model Lifecycle ---
@app.cls(
    gpu="L40S",  # High-performance GPU for rapid inference
    image=sam2_image,
    volumes={"/weights": weights_volume},
    timeout=60,
    scaledown_window=120,
    max_containers=10  # Maximum auto-scaled GPU containers
)
@modal.concurrent(max_inputs=8)
class SAM2Service:
    @modal.enter()
    def load_model(self):
        import os
        import sys
        import torch

        # Prevent SAM 2 repository shadowing by switching to the cloned directory at /opt/sam2
        os.chdir("/opt/sam2")
        
        if "/root" in sys.path:
            sys.path.remove("/root")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing SAM 2.1 on GPU: {self.device}")

        sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        ckpt_path = "/weights/sam2.1_hiera_base_plus.pt"

        sam2_model = build_sam2(sam2_cfg, ckpt_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("SAM 2.1 loaded on GPU inside Modal container.")

    @modal.method()
    def process_refinement(self, image_base64: str, candidates: list) -> dict:
        start_time = time.time()

        # Decode Image
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image_rgb)
        results = []

        for cand in candidates:
            point_coords = []
            point_labels = []

            for pt in cand.get("positive_points", []):
                point_coords.append(pt)
                point_labels.append(1)

            for pt in cand.get("negative_points", []):
                point_coords.append(pt)
                point_labels.append(0)

            coords_arr = np.array(point_coords, dtype=np.float32) if point_coords else None
            labels_arr = np.array(point_labels, dtype=np.int32) if point_labels else None
            box_arr = np.array(cand.get("bbox"), dtype=np.float32) if cand.get("bbox") else None

            masks, scores, _ = self.predictor.predict(
                point_coords=coords_arr,
                point_labels=labels_arr,
                box=box_arr,
                multimask_output=False
            )

            best_mask = masks[0].astype(np.uint8)
            best_score = float(scores[0])

            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            primary_polygon = []
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                if len(largest_cnt) >= 3:
                    primary_polygon = largest_cnt.reshape(-1, 2).tolist()

            results.append({
                "id": cand.get("building_id"),
                "footprint_proposal": primary_polygon,
                "roof_sections": [],
                "scores": {"sam2": best_score},
                "provenance": ["sam2.1_hiera_base_plus_gpu"]
            })

        elapsed_ms = (time.time() - start_time) * 1000.0
        return {
            "schema_version": "depthwizard.roofgraph.v1",
            "buildings": results,
            "processing_time_ms": round(elapsed_ms, 2)
        }


# --- ASGI Web Endpoint for C++ Integration ---
@app.function(
    image=sam2_image,
    max_containers=10
)
@modal.asgi_app(label="depthwizard-api")  # Sets explicit URL prefix
def fastapi_app():
    from fastapi import FastAPI

    web_app = FastAPI(title="DepthWizard SAM 2.1 Refinement API")

    @web_app.get("/api/v1/health")
    def health():
        return {"status": "healthy", "service": "modal_sam2"}

    @web_app.post("/api/v1/refine", response_model=RefineResponse)
    async def refine(req: RefineRequest):
        service = SAM2Service()
        cand_dicts = [c.model_dump() for c in req.candidates]
        res = await service.process_refinement.remote.aio(req.image_base64, cand_dicts)
        return res

    return web_app