from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from eye_cnn import CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Membuat artifacts/manifest.csv dari folder dataset per kelas."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder induk yang berisi satu subfolder per kelas.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.csv"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def verify_image(path: Path) -> str:
    """Kembalikan 'ok' bila file benar-benar bisa dibuka sebagai gambar."""
    try:
        with Image.open(path) as image:
            image.verify()
        return "ok"
    except Exception:
        return "corrupt"


def collect_class_files(data_dir: Path, class_name: str) -> list[Path]:
    class_dir = data_dir / class_name
    if not class_dir.is_dir():
        raise SystemExit(
            f"Subfolder kelas tidak ditemukan: {class_dir}\n"
            f"Folder --data-dir harus berisi tepat subfolder: {', '.join(CLASSES)}"
        )
    files = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"Tidak ada gambar (.jpg/.jpeg/.png) di {class_dir}")
    return files


def assign_splits(
    count: int, val_ratio: float, test_ratio: float
) -> list[str]:
    """Split stratified per kelas: sisanya masuk train."""
    n_val = int(round(count * val_ratio))
    n_test = int(round(count * test_ratio))
    n_train = count - n_val - n_test
    if n_train <= 0:
        raise SystemExit(
            f"Rasio val+test terlalu besar untuk kelas dengan {count} gambar."
        )
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio + args.test_ratio < 1:
        raise SystemExit("val-ratio + test-ratio harus di antara 0 dan 1.")

    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    # eye_cnn.data.FundusDataset menghitung path relatif terhadap
    # manifest.parent.parent, jadi dataset wajib berada di dalam folder itu.
    root = output.parent.parent
    try:
        data_dir.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"Dataset harus berada di dalam {root}\n"
            f"Sekarang: {data_dir}\n"
            "Pindahkan folder dataset ke dalam folder tersebut "
            "(contoh: backend/data/) lalu jalankan ulang."
        )

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    for class_name in CLASSES:
        files = collect_class_files(data_dir, class_name)
        splits = assign_splits(len(files), args.val_ratio, args.test_ratio)
        rng.shuffle(splits)
        for path, split in zip(files, splits):
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "label": class_name,
                    "split": split,
                    "status": verify_image(path),
                }
            )

    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "split", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    corrupt = [row for row in rows if row["status"] != "ok"]
    print(f"Manifest ditulis: {output}")
    print(f"Total gambar: {len(rows)} (dilewati karena rusak: {len(corrupt)})")
    for split in ("train", "val", "test"):
        counts = Counter(
            row["label"]
            for row in rows
            if row["split"] == split and row["status"] == "ok"
        )
        detail = ", ".join(f"{name}={counts[name]}" for name in CLASSES)
        print(f"  {split:5s} total={sum(counts.values()):5d} | {detail}")
    for row in corrupt:
        print(f"  RUSAK: {row['path']}")


if __name__ == "__main__":
    main()
