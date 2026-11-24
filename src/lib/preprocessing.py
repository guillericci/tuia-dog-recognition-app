"""Preprocesamiento compartido entre notebooks, entrenamiento y servicios de la app.

La entrada es siempre una imagen BGR uint8 (como la devuelve cv2.imread
y como la entrega la app). La salida es un array float32 (3, H, W) normalizado con
las estadísticas de ImageNet, en orden RGB.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # RGB
# Color de relleno para el modo "pad": la media de ImageNet en uint8 (RGB).
# Tras normalizar, el relleno queda en ~0 y no introduce un sesgo de color.
_PAD_RGB = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)


def _ensure_bgr3(image: np.ndarray) -> np.ndarray:
    """Acepta gris (H,W), (H,W,1) o BGRA (H,W,4) y devuelve BGR (H,W,3) uint8."""
    if image is None or image.size == 0:
        raise ValueError("Imagen vacía o no legible.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resize_rgb(rgb: np.ndarray, size: int, mode: str = "resize") -> np.ndarray:
    """Lleva una imagen RGB uint8 a (size, size).

    mode="resize": resize directo (deforma la relación de aspecto).
    mode="pad":    letterbox, conserva la relación de aspecto y rellena con la media.
    """
    h, w = rgb.shape[:2]
    if mode == "resize":
        interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_LINEAR
        return cv2.resize(rgb, (size, size), interpolation=interp)
    if mode == "pad":
        scale = size / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
        canvas = np.full((size, size, 3), _PAD_RGB, dtype=np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas
    raise ValueError(f"mode desconocido: {mode!r} (esperado 'resize' o 'pad')")


def normalize_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 (3,H,W) normalizado. Lo usa también el
    pipeline de entrenamiento DESPUÉS de la augmentation (que opera sobre uint8)."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_bgr(image_bgr: np.ndarray, image_size: int = 224, mode: str = "resize") -> np.ndarray:
    """Pipeline de inferencia completo: BGR uint8 -> (3, S, S) float32 normalizado."""
    bgr = _ensure_bgr3(image_bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = resize_rgb(rgb, image_size, mode)
    return normalize_rgb_uint8(rgb)


def preprocess_to_tensor(image_bgr: np.ndarray, image_size: int = 224, mode: str = "resize") -> torch.Tensor:
    """Igual que preprocess_bgr pero devuelve un tensor (1, 3, S, S) listo para el modelo."""
    return torch.from_numpy(preprocess_bgr(image_bgr, image_size, mode)).unsqueeze(0)