import numpy as np

def mean_brightness(img: np.ndarray) -> float:
    """Mean pixel value of an image array (0-255)."""
    return float(np.mean(img))