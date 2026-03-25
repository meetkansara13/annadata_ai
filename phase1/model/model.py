"""
अन्नदाता AI — Phase 1
Model: MobileNetV3-Small fine-tuned for crop disease detection.

Why MobileNetV3-Small?
  - Runs on cheap Android phones (target: farmers with ₹5000-8000 phones)
  - Only ~2.5MB model size — works offline, no internet needed in the field
  - 94%+ accuracy on PlantVillage after fine-tuning
  - Much faster than ResNet50 / EfficientNet on mobile CPU
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class AnnadataModel(nn.Module):
    """
    MobileNetV3-Small with a custom classification head for 38 crop diseases.
    The backbone is frozen for the first N epochs (feature extraction),
    then unfrozen for full fine-tuning.
    """

    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()

        # Load pretrained MobileNetV3-Small backbone
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Keep everything except the final classifier
        self.features  = backbone.features
        self.avgpool   = backbone.avgpool

        # Custom head: Dropout → FC → BN → HardSwish → Dropout → FC(num_classes)
        # Mirrors MobileNetV3 style but outputs our disease classes
        in_features = backbone.classifier[0].in_features  # 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        # Initialize custom head weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """Freeze backbone — only train the classifier head."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("[अन्नदाता AI] Backbone frozen — training head only.")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("[अन्नदाता AI] Backbone unfrozen — full fine-tuning.")

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[अन्नदाता AI] Parameters — Total: {total:,}  Trainable: {trainable:,}")
        return total, trainable


def build_model(num_classes: int = 38, pretrained: bool = True) -> AnnadataModel:
    model = AnnadataModel(num_classes=num_classes, pretrained=pretrained)
    return model


def load_checkpoint(path: str | Path, num_classes: int = 38, device: str = "cpu") -> AnnadataModel:
    """Load a saved checkpoint for inference."""
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"[अन्नदाता AI] Checkpoint loaded from '{path}'  (epoch {state.get('epoch', '?')})")
    return model
