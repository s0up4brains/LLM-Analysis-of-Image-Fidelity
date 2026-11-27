import numpy as np
import cv2
from pathlib import Path

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_gray_pair(path0: str, path1: str):
    a, b = load_gray(path0), load_gray(path1)
    if a.shape != b.shape:
        raise ValueError(f"Image sizes differ: {a.shape} vs {b.shape}")
    return a, b


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_info(path: str) -> dict:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    channels = 1 if img.ndim == 2 else img.shape[2]
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": int(channels),
        "has_alpha": channels == 4,
        "dtype": str(img.dtype),
        "format": Path(path).suffix.lower().lstrip("."),
        "file_size_bytes": Path(path).stat().st_size,
    }