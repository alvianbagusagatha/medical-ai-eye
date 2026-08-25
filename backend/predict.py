from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from eye_cnn import CLASSES
from eye_cnn.data import build_transforms
from eye_cnn.models import build_model
from train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi satu citra fundus.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    classes = tuple(checkpoint["classes"])
    if classes != CLASSES:
        raise ValueError("Urutan kelas checkpoint tidak cocok dengan pipeline.")

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(classes),
        dropout=checkpoint.get("dropout", 0.35),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    transform = build_transforms(checkpoint["image_size"], train=False)
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    ranking = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    print(f"Gambar: {args.image}")
    print(f"Prediksi: {ranking[0][0]} ({ranking[0][1] * 100:.2f}%)")
    for class_name, probability in ranking:
        print(f"  {class_name:22s} {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()

