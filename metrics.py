"""Evaluation metrics and degradation helpers for the Auto-Clip CLAHE study."""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk

_BR_MODEL = "brisque_model/brisque_model_live.yml"
_BR_RANGE = "brisque_model/brisque_range_live.yml"


def to_L(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def entropy(bgr):
    """Shannon entropy (bits) of the L channel."""
    L = to_L(bgr)
    hist = np.bincount(L.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def psnr(test_bgr, ref_bgr):
    return float(psnr_sk(ref_bgr, test_bgr, data_range=255))


def ssim(test_bgr, ref_bgr):
    a = to_L(test_bgr)
    b = to_L(ref_bgr)
    return float(ssim_sk(b, a, data_range=255))


def brisque(bgr):
    L = to_L(bgr)
    return float(cv2.quality.QualityBRISQUE_compute(L, _BR_MODEL, _BR_RANGE)[0])


# --------------------------------------------------------------------------- #
# Degradation: low-contrast simulation + additive Gaussian noise
# --------------------------------------------------------------------------- #
def reduce_contrast(bgr, factor=0.55):
    """Compress dynamic range of the L channel toward mid-grey.

    Simulates a low-contrast acquisition. factor<1 lowers contrast.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    L = 128.0 + (L - 128.0) * factor
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def add_gaussian_noise(bgr, sigma, rng=None):
    """Add zero-mean Gaussian noise (std `sigma`, 0-255 scale) to all channels."""
    if rng is None:
        rng = np.random.default_rng()
    noisy = bgr.astype(np.float32) + rng.normal(0, sigma, bgr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
