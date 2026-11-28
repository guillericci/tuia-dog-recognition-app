"""Modelo B: CNN propia entrenada desde cero. Convención: head `fc`, método `embed`, atributo `classes`."""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv(c_in: int, c_out: int, stride: int = 1, bn: bool = True) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, 3, stride, 1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers.append(nn.ReLU(inplace=True))
    return layers


class BreedCNN(nn.Module):
    """224x224 -> stem (stride 2) -> maxpool -> 4 bloques de 2 convs -> GAP -> embedding -> clasificador."""

    def __init__(self, classes: list[str], embedding_dim: int = 512, dropout: float = 0.3,
                 widths: tuple = (32, 64, 128, 256, 256), use_bn: bool = True):
        super().__init__()
        stem, *blocks = widths
        layers = _conv(3, stem, stride=2, bn=use_bn) + [nn.MaxPool2d(2)]          # 224 -> 112 -> 56
        c = stem
        for i, w in enumerate(blocks):
            layers += _conv(c, w, bn=use_bn) + _conv(w, w, bn=use_bn)
            if i < len(blocks) - 1:
                layers.append(nn.MaxPool2d(2))                                    # 56 -> 28 -> 14 -> 7
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        emb: list[nn.Module] = [nn.Linear(c, embedding_dim)]
        if use_bn:
            emb.append(nn.BatchNorm1d(embedding_dim))
        emb.append(nn.ReLU(inplace=True))
        self.embedding = nn.Sequential(*emb)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, len(classes))
        self.embedding_dim = embedding_dim
        self.classes = list(classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(torch.flatten(self.pool(self.features(x)), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.embed(x)))