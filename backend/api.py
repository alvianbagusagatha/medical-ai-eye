"""
API FastAPI untuk model klasifikasi 4 penyakit mata (CNN).

Menjalankan:
    pip install -r requirements.txt
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn api:app --reload --port 8000

Endpoint:
    GET  /health            -> status server & daftar kelas
    POST /predict           -> multipart/form-data, field "file" (jpg/png)
                                mengembalikan prediksi + probabilitas tiap kelas
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from eye_cnn import CLASSES
from eye_cnn.data import build_transforms
from eye_cnn.models import build_model

CHECKPOINT_PATH = Path(__file__).parent / "runs" / "eyecnn" / "best_model.pt"

# Label ramah-pengguna (Bahasa Indonesia) untuk ditampilkan di front-end.
LABEL_ID = {
    "cataract": "Katarak",
    "diabetic_retinopathy": "Retinopati Diabetik",
    "glaucoma": "Glaukoma",
    "normal": "Normal / Sehat",
}

app = FastAPI(title="Medical-AI Eye Classifier API")

# Dibuka untuk semua origin karena front-end adalah file statis (file:// atau
# server statis terpisah di port lain). Persempit allow_origins di produksi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"model": None, "transform": None, "device": None, "classes": None}


def _load_model() -> None:
    if _state["model"] is not None:
        return

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    classes = tuple(checkpoint["classes"])
    if classes != CLASSES:
        raise RuntimeError("Urutan kelas checkpoint tidak cocok dengan pipeline eye_cnn.")

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(classes),
        dropout=checkpoint.get("dropout", 0.35),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _state["model"] = model
    _state["transform"] = build_transforms(checkpoint["image_size"], train=False)
    _state["device"] = device
    _state["classes"] = classes


@app.on_event("startup")
def on_startup() -> None:
    # Dimuat sekali saat server start supaya tiap request cepat.
    _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classes": list(_state["classes"] or CLASSES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Format file harus JPG atau PNG.")

    _load_model()

    raw_bytes = await file.read()
    try:
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.") from exc

    tensor = _state["transform"](image).unsqueeze(0).to(_state["device"])
    with torch.inference_mode():
        probabilities = torch.softmax(_state["model"](tensor), dim=1)[0].cpu().tolist()

    ranking = sorted(
        zip(_state["classes"], probabilities), key=lambda item: item[1], reverse=True
    )

    return {
        "prediction": ranking[0][0],
        "prediction_label": LABEL_ID.get(ranking[0][0], ranking[0][0]),
        "confidence": ranking[0][1],
        "probabilities": [
            {
                "class": class_name,
                "label": LABEL_ID.get(class_name, class_name),
                "probability": probability,
            }
            for class_name, probability in ranking
        ],
    }
