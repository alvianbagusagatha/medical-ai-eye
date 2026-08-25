from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from eye_cnn import CLASSES
from eye_cnn.data import FundusDataset
from eye_cnn.metrics import classification_metrics, confusion_matrix
from eye_cnn.models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Melatih model CNN citra fundus.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.csv")
    )
    parser.add_argument(
        "--model",
        choices=("eyecnn", "resnet18", "efficientnet_b0"),
        default="eyecnn",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta, tetapi tidak tersedia.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def class_weights(dataset: FundusDataset):
    import torch

    counts = Counter(row["label"] for row in dataset.rows)
    total = sum(counts.values())
    weights = [
        total / (len(CLASSES) * max(counts[class_name], 1))
        for class_name in CLASSES
    ]
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
) -> tuple[float, dict, list[dict]]:
    import torch

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    probability_rows: list[dict] = []

    context = torch.enable_grad if is_train else torch.inference_mode
    with context():
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_enabled = device.type == "cuda"
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = probabilities.argmax(dim=1)
            total_loss += loss.item() * targets.size(0)
            targets_cpu = targets.detach().cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            targets_all.extend(targets_cpu)
            predictions_all.extend(predictions_cpu)

            for path, target, prediction, probs in zip(
                paths, targets_cpu, predictions_cpu, probabilities_cpu
            ):
                row = {
                    "path": path,
                    "target": CLASSES[target],
                    "prediction": CLASSES[prediction],
                }
                row.update(
                    {
                        f"prob_{class_name}": float(probability)
                        for class_name, probability in zip(CLASSES, probs)
                    }
                )
                probability_rows.append(row)

    matrix = confusion_matrix(targets_all, predictions_all, len(CLASSES))
    metrics = classification_metrics(matrix)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics["loss"], metrics, probability_rows


def save_predictions(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASSES])
        for class_name, row in zip(CLASSES, matrix.tolist()):
            writer.writerow([class_name, *row])


def main() -> None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "PyTorch belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = args.run_dir or Path("runs") / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FundusDataset(
        args.manifest, "train", args.image_size, train=True
    )
    val_dataset = FundusDataset(args.manifest, "val", args.image_size, train=False)
    test_dataset = FundusDataset(
        args.manifest, "test", args.image_size, train=False
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(
        args.model,
        num_classes=len(CLASSES),
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset).to(device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"Device: {device}")
    print(f"Model: {args.model} ({count_parameters(model):,} parameter trainable)")
    print(
        f"Data: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    history: list[dict] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        _, train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        _, val_metrics, _ = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={row['train_loss']:.4f} f1={row['train_macro_f1']:.4f} | "
            f"val loss={row['val_loss']:.4f} f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "classes": list(CLASSES),
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping pada epoch {epoch}.")
                break

    with (run_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, test_metrics, prediction_rows = run_epoch(
        model, test_loader, criterion, device
    )
    targets = [CLASSES.index(row["target"]) for row in prediction_rows]
    predictions = [
        CLASSES.index(row["prediction"]) for row in prediction_rows
    ]
    matrix = confusion_matrix(targets, predictions, len(CLASSES))

    metrics_payload = {
        **test_metrics,
        "classes": list(CLASSES),
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_metrics"]["macro_f1"],
        "model": args.model,
        "parameters": count_parameters(model),
        "device": str(device),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    save_predictions(prediction_rows, run_dir / "test_predictions.csv")
    save_confusion_matrix(matrix, run_dir / "confusion_matrix.csv")
    print(
        f"Test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro F1={test_metrics['macro_f1']:.4f}"
    )
    print(f"Checkpoint terbaik: {checkpoint_path}")


if __name__ == "__main__":
    main()

