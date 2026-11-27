from pathlib import Path
import numpy as np
from tools.imagecheck import load_gray

def contrast_arrays(ref: np.ndarray, edited: np.ndarray) -> dict:
    """Contrast = standard deviation of luminance (0-255 scale)."""
    c_ref = float(np.std(ref.astype(float)))
    c_edit = float(np.std(edited.astype(float)))
    ratio = c_edit / c_ref if c_ref > 0 else None
    if ratio is None:
        change = "undefined (reference has no contrast)"
    elif ratio > 1.05:
        change = "increased"
    elif ratio < 0.95:
        change = "decreased"
    else:
        change = "unchanged"
    return {
        "reference_contrast": c_ref,
        "edited_contrast": c_edit,
        "contrast_ratio": ratio,
        "change": change,
    }
    
    
def contrast_compare(reference_path: str, edited_path: str) -> dict:
    return contrast_arrays(load_gray(reference_path), load_gray(edited_path))