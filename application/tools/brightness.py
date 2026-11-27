import numpy as np
from tools.imagecheck import load_gray

def brightness_arrays(ref: np.ndarray, edited: np.ndarray) -> dict:
    """Brightness = mean luminance (0-255 scale)."""
    b_ref = float(np.mean(ref.astype(float)))
    b_edit = float(np.mean(edited.astype(float)))
    diff = b_edit - b_ref
    if diff > 2:
        change = "brighter"
    elif diff < -2:
        change = "darker"
    else:
        change = "unchanged"
    return {
        "reference_brightness": b_ref,
        "edited_brightness": b_edit,
        "difference": diff,
        "change": change,
    }
    
def brightness_compare(reference_path: str, edited_path: str) -> dict:
    return brightness_arrays(load_gray(reference_path), load_gray(edited_path))
