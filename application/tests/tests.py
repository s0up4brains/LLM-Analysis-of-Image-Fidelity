import numpy as np
import cv2
import pytest
from tools.SSIM import ssim_arrays
from tools.imagecheck import image_info
from tools.PSNR import psnr_arrays
from tools.contrast import contrast_arrays
from tools.brightness import brightness_arrays
from tools.histogram import histogram_arrays


def test_psnr_identical():
    img = np.full((16, 16), 100, dtype=np.uint8)
    assert psnr_arrays(img, img) == {"psnr_db": None, "identical": True}

def test_psnr_constant_offset_known_value():
    a = np.full((16, 16), 100, dtype=np.uint8)
    b = np.full((16, 16), 105, dtype=np.uint8)   # MSE = 25
    assert psnr_arrays(a, b)["psnr_db"] == pytest.approx(34.15, abs=0.01)

def test_ssim_identical_is_one():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    assert ssim_arrays(img, img)["ssim"] == pytest.approx(1.0)

def test_ssim_decreases_as_noise_increases():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    def noisy(amount):
        n = rng.integers(-amount, amount, img.shape)
        return np.clip(img + n, 0, 255).astype(np.uint8)

    mild = ssim_arrays(img, noisy(10))["ssim"]
    strong = ssim_arrays(img, noisy(80))["ssim"]
    assert 1.0 > mild > strong

def test_contrast_ratio_doubles():
    ref = np.array([[100, 150] * 4] * 8, dtype=np.uint8)      # std 25
    edited = np.array([[75, 175] * 4] * 8, dtype=np.uint8)    # std 50
    r = contrast_arrays(ref, edited)
    assert r["contrast_ratio"] == pytest.approx(2.0)
    assert r["change"] == "increased"


def test_contrast_of_flat_reference_is_undefined():
    flat = np.full((8, 8), 100, dtype=np.uint8)
    assert contrast_arrays(flat, flat)["contrast_ratio"] is None


def test_brightness_difference():
    ref = np.full((8, 8), 100, dtype=np.uint8)
    edited = np.full((8, 8), 120, dtype=np.uint8)
    r = brightness_arrays(ref, edited)
    assert r["difference"] == pytest.approx(20.0)
    assert r["change"] == "brighter"


def test_histogram_identical():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    r = histogram_arrays(img, img)
    assert r["red"]["correlation"] == pytest.approx(1.0)
    assert r["red"]["emd"] == pytest.approx(0.0)


def test_histogram_emd_equals_uniform_shift():
    a = np.full((8, 8, 3), 100, dtype=np.uint8)
    b = np.full((8, 8, 3), 110, dtype=np.uint8)
    assert histogram_arrays(a, b)["green"]["emd"] == pytest.approx(10.0)


def test_image_info(tmp_path):
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))
    info = image_info(str(path))
    assert (info["width"], info["height"], info["channels"]) == (30, 20, 3)
    assert info["format"] == "png"