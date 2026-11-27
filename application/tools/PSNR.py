import cv2
import numpy as np
from tools.imagecheck import load_gray, load_gray_pair
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr


def psnr_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    if np.array_equal(a, b):
        return {"psnr_db": None, "identical": True}
    score = psnr(a, b, data_range=255)
    return {"psnr_db": float(score), "identical": False}


def psnr_compare(path0: str, path1: str) -> dict:
    return psnr_arrays(*load_gray_pair(path0, path1))