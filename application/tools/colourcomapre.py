import cv2
import numpy as np
from tools.imagecheck import load_rgb
from skimage.color import rgb2lab, deltaE_ciede2000



def delta_e_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")
    de = deltaE_ciede2000(rgb2lab(a), rgb2lab(b))  # per-pixel map
    return {
        "mean_delta_e": float(np.mean(de)),
        "p95_delta_e": float(np.percentile(de, 95)),
        "max_delta_e": float(np.max(de)),
    }


def delta_e_compare(path0: str, path1: str) -> dict:
    return delta_e_arrays(load_rgb(path0), load_rgb(path1))