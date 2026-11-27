import numpy as np
import pytest

from application.analysis import mean_brightness  

def test_black_image_is_zero():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert mean_brightness(img) == 0.0

def test_white_image_is_255():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert mean_brightness(img) == pytest.approx(255.0)