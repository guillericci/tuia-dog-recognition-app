"""Data augmentation de entrenamiento (solo sobre train). Opera sobre RGB uint8 (H, W, 3)."""
import cv2
import albumentations as A

def build_train_augmentation() -> A.Compose:
    try:                                   # albumentations >= 2.0
        noise = A.GaussNoise(std_range=(0.02, 0.08), p=0.3)
    except Exception:                      # albumentations 1.x
        noise = A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        noise,
    ])