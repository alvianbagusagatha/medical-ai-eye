from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from eye_cnn import CLASSES


def crop_black_border(image: Image.Image, threshold: int = 8) -> Image.Image:
    """Crop nearly-black outer borders without removing the retinal field."""
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    mask = gray.point(lambda value: 255 if value > threshold else 0)
    bbox = mask.getbbox()
    return rgb.crop(bbox) if bbox else rgb


def read_manifest(manifest_path: str | Path, split: str) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == split and row["status"] == "ok"
        ]
    if not rows:
        raise ValueError(f"Tidak ada data split '{split}' di {manifest_path}")
    return rows


def build_transforms(image_size: int, train: bool) -> Callable:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc

    operations: list[Callable] = [
        transforms.Lambda(crop_black_border),
        transforms.Resize((image_size + 24, image_size + 24)),
    ]
    if train:
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(
                    brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02
                ),
            ]
        )
    else:
        operations.extend(
            [
                transforms.Resize((image_size, image_size)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class FundusDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 224,
        train: bool = False,
    ) -> None:
        try:
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch belum terpasang. Jalankan: pip install -r requirements.txt"
            ) from exc

        # Registering as a virtual subclass is unnecessary; DataLoader only
        # requires __len__ and __getitem__.
        _ = Dataset
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent.parent
        self.rows = read_manifest(self.manifest_path, split)
        self.transform = build_transforms(image_size, train=train)
        self.class_to_index = {name: index for index, name in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.root / Path(row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = self.class_to_index[row["label"]]
        return tensor, label, row["path"]

