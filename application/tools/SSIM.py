import numpy as np
from tools.imagecheck import load_gray, load_gray_pair
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


def ssim_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    score = ssim(a, b, data_range=255)
    return {"ssim": float(score)}


def ssim_compare(path0: str, path1: str) -> dict:
    return ssim_arrays(*load_gray_pair(path0, path1))

