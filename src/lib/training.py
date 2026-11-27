"""Entrenamiento de clasificadores de razas (Modelo A y Modelo B).

Usa los splits guardados en 1-pre-processing.ipynb y el mismo preprocesamiento que la inferencia
(lib.preprocessing). Guarda el mejor modelo según validación (modelo completo) y el historial en JSON.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lib.augmentation import build_train_augmentation
from lib.preprocessing import normalize_rgb_uint8, preprocess_bgr, resize_rgb


@dataclass
class TrainConfig:
    strategy: str = "finetune"        # "finetune" (todo el modelo) | "head_only" (backbone congelado)
    epochs: int = 15
    batch_size: int = 64
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4         # solo en "finetune"
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.2
    patience: int = 4                 # early stopping sobre accuracy de validación
    use_class_weights: bool = False
    seed: int = 42
    num_workers: int = 2
    max_batches: int | None = None    # solo para pruebas rápidas (smoke test)
    tag: str = "resnet18_finetuned"   # nombre del historial
    save_as_active: bool = True       # False: guarda en output/<tag>.pth y no pisa el checkpoint de la app
    model_kwargs: dict | None = None  # argumentos extra del constructor del modelo

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


class BreedDataset(Dataset):
    """Lee imágenes con cv2 (BGR) y aplica el pipeline compartido. `augment=None` => evaluación."""

    def __init__(self, csv_path: Path, data_root: Path, classes: list[str], image_size: int, augment=None):
        df = pd.read_csv(csv_path)
        idx = {c: i for i, c in enumerate(classes)}
        self.labels = df["breed"].map(idx).to_numpy()
        assert not np.isnan(self.labels.astype(float)).any(), f"Razas fuera de `classes` en {csv_path}"
        if "label" in df:
            assert (df["label"].to_numpy() == self.labels).all(), "label del CSV no coincide con classes"
        self.paths = [data_root / p for p in df["path"]]
        assert self.paths[0].exists(), f"No existe {self.paths[0]} (revisá data_root / rutas del CSV)"
        self.size, self.augment = image_size, augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        bgr = cv2.imread(str(self.paths[i]))
        if bgr is None:
            raise RuntimeError(f"No se pudo leer {self.paths[i]}")
        if self.augment is None:
            x = preprocess_bgr(bgr, self.size)                        # == pipeline de inferencia
        else:
            rgb = resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size)
            x = normalize_rgb_uint8(self.augment(image=rgb)["image"])  # augmentation -> misma normalización
        return torch.from_numpy(x), int(self.labels[i])


def _run_epoch(model, loader, device, criterion=None, optimizer=None, scaler=None, use_amp=False, max_batches=None):
    train = optimizer is not None
    ce_sum, correct, n = 0.0, 0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            if train:
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            ce_sum += F.cross_entropy(logits, y, reduction="sum").item()   # CE sin suavizado: comparable train/val
            correct += (logits.argmax(1) == y).sum().item()
            n += y.numel()
    return ce_sum / n, correct / n


def run_training(build_model: Callable[[list[str]], nn.Module], cfg: TrainConfig, splits_dir: Path,
                 data_root: Path, image_size: int, checkpoint_path: Path, history_path: Path) -> dict:
    set_seed(cfg.seed)
    meta = json.loads((splits_dir / "meta.json").read_text(encoding="utf-8"))
    classes = meta["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = BreedDataset(splits_dir / "train.csv", data_root, classes, image_size, build_train_augmentation())
    val_ds = BreedDataset(splits_dir / "valid.csv", data_root, classes, image_size, None)
    g = torch.Generator().manual_seed(cfg.seed)
    kw = dict(num_workers=cfg.num_workers, pin_memory=use_amp, worker_init_fn=_seed_worker,
              persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, drop_last=True, generator=g, **kw)
    val_dl = DataLoader(val_ds, cfg.batch_size * 2, shuffle=False, **kw)

    model = build_model(classes).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    back = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    if cfg.strategy == "head_only":
        for p in back:
            p.requires_grad = False
        groups = [{"params": head, "lr": cfg.lr_head}]
    elif cfg.strategy == "finetune":
        groups = [{"params": head, "lr": cfg.lr_head}, {"params": back, "lr": cfg.lr_backbone}]
    elif cfg.strategy == "scratch":      # sin preentrenamiento: todo el modelo con un único lr (lr_head)
        groups = [{"params": head + back, "lr": cfg.lr_head}]
    else:
        raise ValueError(f"strategy desconocida: {cfg.strategy!r}")
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    weights = torch.tensor(meta["class_weights"], dtype=torch.float32, device=device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr_head": []}
    best, best_state, best_epoch, bad = (-1.0, 0.0), None, 0, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        if cfg.strategy == "head_only":
            model.backbone.eval()                      # BatchNorm del backbone congelado no debe actualizarse
        tl, ta = _run_epoch(model, train_dl, device, criterion, optimizer, scaler, use_amp, cfg.max_batches)
        model.eval()
        vl, va = _run_epoch(model, val_dl, device, max_batches=cfg.max_batches)
        scheduler.step()
        for k, v in zip(hist, [ep, tl, ta, vl, va, optimizer.param_groups[0]["lr"]]):
            hist[k].append(v)
        print(f"[{cfg.tag}] ep {ep:02d}/{cfg.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s", flush=True)
        if (va, -vl) > best:
            best, best_epoch, bad = (va, -vl), ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping: sin mejora en {cfg.patience} épocas. Mejor epoch: {best_epoch}")
                break

    model.load_state_dict(best_state)
    model.cpu().eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, checkpoint_path)                                  # modelo COMPLETO (ver 5.6)
    out = {"config": asdict(cfg), "best_epoch": best_epoch, "best_val_acc": best[0],
           "seconds": round(time.time() - t0), "device": str(device), "torch": torch.__version__,
           "n_train": len(train_ds), "n_val": len(val_ds), "history": hist, "checkpoint": str(checkpoint_path)}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out