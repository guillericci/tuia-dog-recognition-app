"""Modelo A: ResNet18 preentrenada con cabeza lineal por raza.

La lista de razas viaja DENTRO del modelo (`model.classes`) para que la app no dependa del dataset.
Convención: la cabeza se llama `fc` (la usa el entrenamiento para separar learning rates).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class BreedResNet(nn.Module):
    def __init__(self, classes: list[str], pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])    # hasta avgpool -> (B, 512, 1, 1)
        self.embedding_dim = net.fc.in_features                       # 512
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embedding_dim, len(classes))
        self.classes = list(classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding de la capa previa al clasificador, (B, 512)."""
        return torch.flatten(self.backbone(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.embed(x)))