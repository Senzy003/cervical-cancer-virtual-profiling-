"""FastAPI backend for the Virtual Protein Profiling project.

Endpoints
    GET  /                 API is running
    GET  /health           model status, targets, whether trained weights loaded
    POST /predict          upload one H&E patch -> per-protein prediction
    GET  /demo-cases       list precomputed demo cases (whole-slide results)
    GET  /demo-cases/{id}  one precomputed case: predictions, true values, heatmaps

Whole-slide inference is too slow to run live, so the demo cases are
precomputed on the GPU cluster and saved as JSON + images in DEMO_DIR.
"""

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model.model_architecture import ProteinExpressionModel
from preprocessing.process import preprocess_image_bytes


# ---------------------------------------------------------------- settings
# Final targets are confirmed once the data teammate checks which proteins
# are in the TCGA-CESC RPPA file. Order MUST match the model's outputs.
TARGETS = os.getenv("VPP_TARGETS", "PD-L1,p16,Cyclin B1,phospho-Rb,E-cadherin").split(",")
WEIGHTS_PATH = Path(os.getenv("VPP_WEIGHTS", "weights/model.pt"))
DEMO_DIR = Path(os.getenv("VPP_DEMO_DIR", "demo_cases"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
DECISION_THRESHOLD = 0.5

DISCLAIMER = ("Research prototype for decision support only. "
              "Not a diagnosis; confirm with laboratory testing.")

# Plain-language meaning of a "High" call, shown in the UI.
MEANINGS = {
    "PD-L1": "Possible immunotherapy candidate: prioritise confirmatory PD-L1 (CPS) testing.",
    "p16": "Pattern consistent with HPV-driven disease.",
    "Cyclin B1": "High proliferation: more aggressive tumour biology.",
    "phospho-Rb": "Active cell-cycle signalling, consistent with HPV E7 disrupting Rb.",
    "E-cadherin": "Cells remain cohesive; low levels suggest invasive behaviour.",
}


# ---------------------------------------------------------------- app
app = FastAPI(
    title="Cervical Cancer Virtual Protein Profiling API",
    description="Research prototype for analysing cervical cancer H&E images.",
    version="0.2.0",
)

# Lets the front end (running on another port) call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before anything beyond the hackathon demo
    allow_methods=["*"],
    allow_headers=["*"],
)

if DEMO_DIR.exists():
    # Heatmaps and thumbnails served at /static/<case_id>/...
    app.mount("/static", StaticFiles(directory=DEMO_DIR), name="static")


# ---------------------------------------------------------------- model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinExpressionModel()
trained = False

if WEIGHTS_PATH.exists():
    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    trained = True

model.to(device)
model.eval()  # prediction mode: no dropout, fixed batch-norm


# ---------------------------------------------------------------- routes
@app.get("/")
def home():
    """Check that the API is running."""
    return {"message": "Virtual Protein Profiling API is running"}


@app.get("/health")
def health():
    """Check that the API and model are available."""
    return {
        "status": "healthy",
        "model": "EfficientNet-B0",
        "targets": TARGETS,
        "trained": trained,
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict high vs low expression per protein for one H&E patch."""

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large; upload a patch, not a whole slide.")

    try:
        image = preprocess_image_bytes(data).to(device)
    except ValueError as error:
        raise HTTPException(422, str(error))

    with torch.no_grad():
        logits = model(image)  # expected shape: [1, len(TARGETS)]
    probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    if len(probabilities) != len(TARGETS):
        raise HTTPException(500, "Model outputs do not match the target list.")

    predictions = []
    for target, probability in zip(TARGETS, probabilities):
        high = probability >= DECISION_THRESHOLD
        predictions.append({
            "protein": target,
            "call": "High" if high else "Low",
            "probability_high": round(probability, 3),
            "meaning": MEANINGS.get(target, "") if high else "",
        })

    return {
        "filename": file.filename,
        "trained": trained,
        "predictions": predictions,
        "disclaimer": DISCLAIMER if trained else
            "Model not trained yet: predictions are random. " + DISCLAIMER,
    }


@app.get("/demo-cases")
def list_demo_cases():
    """List precomputed demo cases (one folder per patient with result.json)."""
    if not DEMO_DIR.exists():
        return {"cases": []}
    cases = []
    for folder in sorted(DEMO_DIR.iterdir()):
        result = folder / "result.json"
        if result.exists():
            info = json.loads(result.read_text())
            cases.append({"case_id": folder.name,
                          "thumbnail": f"/static/{folder.name}/thumbnail.png",
                          "summary": info.get("summary", "")})
    return {"cases": cases}


@app.get("/demo-cases/{case_id}")
def get_demo_case(case_id: str):
    """Full precomputed result for one case: predictions, true lab values,
    and heatmap image paths per protein."""
    result = DEMO_DIR / case_id / "result.json"
    # Block path tricks like ../../ in the case id
    if not result.resolve().is_relative_to(DEMO_DIR.resolve()) or not result.exists():
        raise HTTPException(404, "Case not found.")
    return json.loads(result.read_text())
