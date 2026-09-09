from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eye_cnn import CLASSES
from eye_cnn.data import FundusDataset
from eye_cnn.metrics import classification_metrics, confusion_matrix
from eye_cnn.models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mengevaluasi satu checkpoint pada split manifest tertentu, "
        "dengan format keluaran yang sama seperti train.py."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Batasi thread CPU supaya tidak mengganggu training yang sedang jalan.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Folder tujuan test_metrics.json + confusion_matrix.csv "
        "(default: folder checkpoint).",
    )
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "PyTorch belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    torch.set_num_threads(max(args.threads, 1))
    device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = tuple(checkpoint["classes"])
    if classes != CLASSES:
        raise SystemExit("Urutan kelas checkpoint tidak cocok dengan pipeline.")

    model = build_model(
        checkpoint["model_name"],
        num_classes=len(classes),
        dropout=checkpoint.get("dropout", 0.35),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FundusDataset(
        args.manifest, args.split, checkpoint["image_size"], train=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    targets_all: list[int] = []
    predictions_all: list[int] = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1)
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.cpu().tolist())

    matrix = confusion_matrix(targets_all, predictions_all, len(classes))
    metrics = classification_metrics(matrix)

    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics,
        "classes": list(classes),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_macro_f1": checkpoint.get("val_metrics", {}).get("macro_f1"),
        "model": checkpoint["model_name"],
        "parameters": count_parameters(model),
        "device": str(device),
        "evaluated_split": args.split,
        "manifest": str(args.manifest),
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (out_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *classes])
        for name, row in zip(classes, matrix.tolist()):
            writer.writerow([name, *row])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split} ({len(dataset)} gambar)")
    print(f"Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"Ditulis ke : {out_dir}")


if __name__ == "__main__":
    main()
