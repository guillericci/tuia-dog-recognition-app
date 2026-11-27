from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

import cv2
import pandas as pd
from lib.preprocessing import preprocess_to_tensor

logger = logging.getLogger(__name__)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------
    def train_classifier(self) -> None:
        """Entrena el modelo activo. Hiperparámetros: `self.train_config` (dict opcional, ver TrainConfig)."""
        from lib.training import TrainConfig, run_training

        cfg = TrainConfig(**{"tag": self.active_model_name, **getattr(self, "train_config", {})})
        if self.active_model_name == "resnet18_finetuned":
            from lib.models.resnet_classifier import BreedResNet
            build = lambda classes: BreedResNet(classes, pretrained=True, dropout=cfg.dropout)
        elif self.active_model_name == "cnn_custom":
            from lib.models.cnn_custom import BreedCNN
            emb_dim = getattr(self, "embedding_dim", 512)          # debe coincidir con EMBEDDING_DIM
            build = lambda classes: BreedCNN(classes, embedding_dim=emb_dim, dropout=cfg.dropout,
                                             **(cfg.model_kwargs or {}))
        else:
            raise ValueError(f"Modelo desconocido: {self.active_model_name!r}")

        data_root = self.dataset_path.parent
        splits_dir = self._find_splits_dir()
        ckpt = self.active_checkpoint if cfg.save_as_active else self.output_path / f"{cfg.tag}.pth"
        run_training(build_model=build, cfg=cfg, splits_dir=splits_dir, data_root=data_root,
                     image_size=self.image_size, checkpoint_path=ckpt,
                     history_path=self.output_path / f"history_{cfg.tag}.json")
        if cfg.save_as_active:
            self._loaded.pop(self.active_model_name, None)
            
    def evaluate_classifier(self) -> dict[str, float]:
        """Evalúa el modelo activo sobre `test` (splits de 1-pre-processing)."""
        d = pd.read_csv(self._find_splits_dir() / "test.csv")
        data_root = self.dataset_path.parent
        return self.evaluate_paths([data_root / p for p in d.path], d.breed.tolist(), name="test")

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """Embedding (capa previa al clasificador) de una imagen BGR uint8, normalizado L2."""
        model = self._inference_model()
        with torch.no_grad():
            emb = model.embed(preprocess_to_tensor(image, self.image_size))[0].float()
            emb = torch.nn.functional.normalize(emb, dim=0)
        return emb.tolist()
    
    # ------------------------------------------------------------------
    # Auxiliares de inferencia y evaluación
    # ------------------------------------------------------------------
    def _find_splits_dir(self) -> Path:
        for cand in (self.dataset_path.parent / "splits", self.output_path / "splits"):
            if (cand / "train.csv").exists():
                return cand
        raise FileNotFoundError("No encuentro los splits: copiá train/valid/test.csv a data/splits/.")

    def _inference_model(self):
        model = self.load_model()
        if not isinstance(model, torch.nn.Module):
            raise NotImplementedError("Este flujo usa checkpoints .pth (modelo completo de PyTorch).")
        return model.eval()

    def _predict_paths(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        """Probabilidades (N, C) del modelo activo con el mismo preprocesamiento que la app."""
        model = self._inference_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = []
        try:
            model.to(device)
            with torch.no_grad():
                for s in range(0, len(paths), batch_size):
                    batch = []
                    for p in paths[s:s + batch_size]:
                        img = cv2.imread(str(p))
                        if img is None:
                            raise RuntimeError(f"No se pudo leer {p}")
                        batch.append(preprocess_to_tensor(img, self.image_size))
                    logits = model(torch.cat(batch).to(device))
                    out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        finally:
            model.to("cpu")            # el modelo cacheado queda en CPU, como espera la app
        return np.concatenate(out)

    @staticmethod
    def _metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
        from lib.evaluation.metrics import precision_recall_f1, specificity
        cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
        tp = np.diag(cm)
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        tn = cm.sum() - tp - fp - fn
        present = np.flatnonzero(cm.sum(1) > 0)           # solo clases con imágenes reales en este conjunto
        prf = np.array([precision_recall_f1(int(tp[c]), int(fp[c]), int(fn[c])) for c in present])
        spec = [specificity(int(tn[c]), int(fp[c])) for c in present]
        return {"accuracy": float(tp.sum() / cm.sum()), "precision": float(prf[:, 0].mean()),
                "recall": float(prf[:, 1].mean()), "specificity": float(np.mean(spec)),
                "f1": float(prf[:, 2].mean())}

    def evaluate_paths(self, paths: list[Path], breeds: list[str], name: str = "custom") -> dict[str, float]:
        """Evalúa el modelo activo sobre cualquier lista de imágenes con su raza (test, independiente...).
        Guarda predicciones y probabilidades en self.last_eval['<modelo>:<name>'] para el análisis de errores."""
        model = self._inference_model()
        idx = {c: i for i, c in enumerate(model.classes)}          # mapeo que viaja dentro del modelo
        y_true = np.array([idx[b] for b in breeds])                # KeyError si una raza no existe en el modelo
        probs = self._predict_paths(paths)
        y_pred = probs.argmax(1)
        if not hasattr(self, "last_eval"):
            self.last_eval = {}
        self.last_eval[f"{self.active_model_name}:{name}"] = {
            "paths": [str(p) for p in paths], "y_true": y_true, "y_pred": y_pred,
            "probs": probs, "classes": list(model.classes)}
        return self._metrics_from_predictions(y_true, y_pred, len(model.classes))

