import numpy as np
from tools.imagecheck import load_rgb

def _norm_hist(channel: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(channel, bins=256, range=(0, 256))
    return h / h.sum()


def histogram_arrays(ref_rgb: np.ndarray, edited_rgb: np.ndarray) -> dict:
    """Per-channel histogram comparison.

    correlation: 1.0 = same shape of distribution.
    emd: earth mover's distance in grey levels, i.e. roughly how far
         pixel values were moved on average (0 = identical).
    """
    out = {}
    for i, name in enumerate(["red", "green", "blue"]):
        h0 = _norm_hist(ref_rgb[..., i])
        h1 = _norm_hist(edited_rgb[..., i])
        corr = float(np.corrcoef(h0, h1)[0, 1])
        emd = float(np.sum(np.abs(np.cumsum(h0) - np.cumsum(h1))))
        out[name] = {"correlation": corr, "emd": emd}
    return out

def histogram_compare(reference_path: str, edited_path: str) -> dict:
    return histogram_arrays(load_rgb(reference_path), load_rgb(edited_path))