from __future__ import annotations


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch belum terpasang. Jalankan: pip install -r requirements.txt"
        ) from exc
    return torch, nn


def build_model(
    model_name: str,
    num_classes: int = 4,
    dropout: float = 0.35,
    pretrained: bool = False,
):
    torch, nn = _torch_modules()

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and stride == 1
                else nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            )
            self.activation = nn.SiLU(inplace=True)

        def forward(self, inputs):
            return self.activation(self.main(inputs) + self.skip(inputs))

    class EyeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResidualBlock(32, 48),
                ResidualBlock(48, 64, stride=2),
                ResidualBlock(64, 64),
                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128, 128),
                ResidualBlock(128, 256, stride=2),
                ResidualBlock(256, 256),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, inputs):
            return self.classifier(self.pool(self.features(inputs)))

    if model_name == "eyecnn":
        model = EyeCNN()
    elif model_name in {"resnet18", "efficientnet_b0"}:
        try:
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError("torchvision diperlukan untuk model ini") from exc

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.fc.in_features, num_classes),
            )
        else:
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
    else:
        raise ValueError(
            f"Model tidak dikenal: {model_name}. "
            "Pilih eyecnn, resnet18, atau efficientnet_b0."
        )

    return model


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

