"""
FastAPI service exposing the claim image authenticity check as a single
endpoint: upload a photo + claim narrative, get back a risk score,
explanation, and routing decision.

Run:
    uvicorn src.api.main:app --reload
"""
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, UploadFile, Form
from torchvision import transforms
from PIL import Image

from src.models.forensic_classifier import build_model
from src.models.fusion import fuse_scores
from src.utils.metadata_check import check_metadata

app = FastAPI(title="SynthFraud Detect", version="0.1.0")

with open("configs/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_model = None
_transform = transforms.Compose([
    transforms.Resize((CONFIG["data"]["image_size"], CONFIG["data"]["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_model():
    global _model
    if _model is None:
        _model = build_model(
            backbone=CONFIG["model"]["backbone"],
            pretrained=False,
            num_classes=CONFIG["model"]["num_classes"],
        )
        _model.load_state_dict(torch.load("models_output/forensic_classifier.pt", map_location="cpu"))
        _model.eval()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check_claim_image")
async def check_claim_image(
    image: UploadFile = File(...),
    claim_narrative: str = Form(""),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    # 1. Forensic classifier score
    model = get_model()
    pil_image = Image.open(tmp_path).convert("RGB")
    tensor = _transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        forensic_score = torch.softmax(output, dim=1)[0, 1].item()

    # 2. Metadata score
    metadata_result = check_metadata(tmp_path)
    metadata_score = metadata_result["metadata_suspicion_score"]

    # 3. Semantic score (optional -- requires claim_narrative and API key)
    semantic_score = 0.0
    semantic_explanation = "not evaluated (no claim narrative provided)"
    if claim_narrative:
        try:
            from src.models.semantic_check import check_semantic_consistency
            semantic_result = check_semantic_consistency(tmp_path, claim_narrative)
            if semantic_result["consistent"] == "no":
                semantic_score = semantic_result["confidence"] or 0.5
            semantic_explanation = semantic_result["explanation"]
        except Exception as e:
            semantic_explanation = f"semantic check failed: {e}"

    Path(tmp_path).unlink(missing_ok=True)

    fusion_result = fuse_scores(forensic_score, metadata_score, semantic_score, CONFIG)
    fusion_result["semantic_explanation"] = semantic_explanation
    fusion_result["metadata_flags"] = metadata_result["flags"]

    return fusion_result
