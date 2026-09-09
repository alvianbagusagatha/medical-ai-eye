from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eye_cnn import CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Membaca hasil training jadi laporan yang mudah dibaca."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Folder hasil training, mis. runs/resnet18_pretrained",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Folder run lain untuk dibandingkan (opsional).",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "test_metrics.json"
    if not path.exists():
        raise SystemExit(
            f"Belum ada {path}.\n"
            "Training belum selesai, atau dihentikan sebelum tahap evaluasi test."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix(run_dir: Path) -> list[list[int]]:
    path = run_dir / "confusion_matrix.csv"
    if not path.exists():
        raise SystemExit(f"Belum ada {path}.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return [[int(value) for value in row[1:]] for row in rows[1:]]


def print_per_class(metrics: dict) -> None:
    classes = metrics.get("classes", list(CLASSES))
    print(f"{'kelas':24s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'data':>6s}")
    print("-" * 55)
    for index, name in enumerate(classes):
        print(
            f"{name:24s} "
            f"{metrics['per_class_precision'][index]:7.3f} "
            f"{metrics['per_class_recall'][index]:7.3f} "
            f"{metrics['per_class_f1'][index]:7.3f} "
            f"{metrics['support'][index]:6d}"
        )


def print_matrix(matrix: list[list[int]], classes: list[str]) -> None:
    width = max(len(name) for name in classes) + 2
    header = "".join(f"{name[:8]:>10s}" for name in classes)
    print(f"{'aktual v / prediksi >':<{width}s}{header}")
    for name, row in zip(classes, matrix):
        cells = "".join(f"{value:>10d}" for value in row)
        print(f"{name:<{width}s}{cells}")


def print_confusions(matrix: list[list[int]], classes: list[str]) -> None:
    """Daftar kesalahan terbesar: aktual X tapi diprediksi Y."""
    confusions = []
    for actual, row in enumerate(matrix):
        total = sum(row)
        for predicted, count in enumerate(row):
            if actual != predicted and count > 0:
                share = count / total if total else 0.0
                confusions.append((count, share, actual, predicted))
    confusions.sort(reverse=True)

    print("Kesalahan terbesar (aktual -> diprediksi):")
    if not confusions:
        print("  tidak ada kesalahan sama sekali - periksa kebocoran data!")
        return
    for count, share, actual, predicted in confusions[:6]:
        print(
            f"  {classes[actual]:22s} -> {classes[predicted]:22s} "
            f"{count:4d} kasus ({share*100:5.1f}% dari kelas itu)"
        )


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.run_dir)
    classes = metrics.get("classes", list(CLASSES))
    matrix = load_matrix(args.run_dir)

    print("=" * 55)
    print(f"LAPORAN HASIL TEST - {args.run_dir}")
    print("=" * 55)
    print(f"Arsitektur      : {metrics.get('model')}")
    print(f"Parameter       : {metrics.get('parameters', 0):,}")
    print(f"Epoch terbaik   : {metrics.get('best_epoch')}")
    print(f"Val macro F1    : {metrics.get('best_val_macro_f1', 0):.4f}")
    print()
    print(f"TEST accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"TEST macro F1   : {metrics['macro_f1']:.4f}")
    print()
    print_per_class(metrics)
    print()
    print_matrix(matrix, classes)
    print()
    print_confusions(matrix, classes)

    if args.baseline:
        base = load_metrics(args.baseline)
        print()
        print("=" * 55)
        print(f"PERBANDINGAN vs {args.baseline}")
        print("=" * 55)
        delta_acc = (metrics["accuracy"] - base["accuracy"]) * 100
        delta_f1 = metrics["macro_f1"] - base["macro_f1"]
        print(
            f"accuracy : {base['accuracy']*100:6.2f}% -> "
            f"{metrics['accuracy']*100:6.2f}%  ({delta_acc:+.2f} poin)"
        )
        print(
            f"macro F1 : {base['macro_f1']:6.4f}  -> "
            f"{metrics['macro_f1']:6.4f}   ({delta_f1:+.4f})"
        )
        for index, name in enumerate(classes):
            before = base["per_class_f1"][index]
            after = metrics["per_class_f1"][index]
            print(f"  F1 {name:22s} {before:.3f} -> {after:.3f} ({after-before:+.3f})")


if __name__ == "__main__":
    main()
