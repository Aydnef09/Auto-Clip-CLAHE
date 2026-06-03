"""
Auto-Clip CLAHE: Noise-Aware Automatic Parameter Selection for
Adaptive Contrast Enhancement.

This module implements:
  - estimate_noise_mad : global noise floor (sigma) via the Median Absolute
    Deviation (MAD) of the finest-scale diagonal wavelet sub-band
    (Donoho & Johnstone, 1994).
  - tile_entropy        : per-tile Shannon entropy (local texture energy).
  - clahe_per_tile      : a custom CLAHE that accepts a DIFFERENT clip limit
    for every tile (OpenCV's CLAHE only supports a single global clip limit),
    with bilinear interpolation of the tile mappings (Zuiderveld, 1994).
  - auto_clip_clahe     : the proposed method. Computes C_{i,j} = alpha +
    beta * (E_{i,j} / (sigma + eps)) per tile and feeds it to clahe_per_tile.

All enhancement is performed on the L (lightness) channel in CIELAB space so
that chrominance is preserved; the colour image is reconstructed afterwards.

Author: Yagizefe Aydin (2211051071), COMP430 Digital Image Processing.
"""

import numpy as np
import cv2
import pywt


# --------------------------------------------------------------------------- #
# 1. Global noise-floor estimation (MAD on diagonal wavelet coefficients)
# --------------------------------------------------------------------------- #
def estimate_noise_mad(gray):
    """Estimate the global noise standard deviation (0-255 scale).

    Uses the robust MAD estimator on the diagonal (HH) detail coefficients of
    a single-level 'db1' (Haar) wavelet decomposition:

        sigma = median(|HH|) / 0.6745

    This is the canonical wavelet noise estimator of Donoho & Johnstone (1994).
    The diagonal band is dominated by high-frequency content; for natural
    images its robust spread is an effective proxy for the sensor noise floor.
    """
    g = gray.astype(np.float64)
    # Single-level 2-D DWT; cD = diagonal detail coefficients (HH band).
    _, (_, _, cD) = pywt.dwt2(g, "db1")
    mad = np.median(np.abs(cD - np.median(cD)))
    sigma = mad / 0.6745
    return float(sigma)


# --------------------------------------------------------------------------- #
# 2. Per-tile Shannon entropy (local texture energy E_{i,j})
# --------------------------------------------------------------------------- #
def _shannon_entropy(tile):
    """Shannon entropy (bits) of an 8-bit tile from its 256-bin histogram."""
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def tile_entropy(gray, grid):
    """Return a (gy, gx) array of per-tile Shannon entropies."""
    h, w = gray.shape
    gy, gx = grid
    ent = np.zeros((gy, gx), dtype=np.float64)
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    for i in range(gy):
        for j in range(gx):
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            ent[i, j] = _shannon_entropy(tile)
    return ent


# --------------------------------------------------------------------------- #
# 3. Custom CLAHE with a per-tile clip limit + bilinear interpolation
# --------------------------------------------------------------------------- #
def _tile_lut(tile, clip_count):
    """Build the 256-entry mapping LUT for one tile.

    clip_count : maximum number of pixels allowed in any histogram bin.
                 The excess is clipped and redistributed uniformly
                 (Zuiderveld, 1994).
    """
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    clip_count = max(1.0, float(clip_count))
    excess = np.maximum(hist - clip_count, 0.0).sum()
    hist = np.minimum(hist, clip_count)
    # Redistribute the clipped excess uniformly across all 256 bins.
    hist += excess / 256.0
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total <= 0:
        return np.arange(256, dtype=np.float64)
    lut = (cdf - cdf.min()) / (total - cdf.min() + 1e-12) * 255.0
    return lut


def clahe_per_tile(gray, clip_map, grid):
    """CLAHE allowing a distinct clip limit per tile, with bilinear blending.

    Parameters
    ----------
    gray     : uint8 image (single channel).
    clip_map : (gy, gx) array. clip_map[i, j] is the NORMALISED clip limit for
               tile (i, j), expressed as a fraction of the tile's pixel count
               (e.g. 0.01 == clip every bin at 1% of the tile pixels).
    grid     : (gy, gx) number of tiles.

    Returns
    -------
    uint8 enhanced image of the same shape.
    """
    h, w = gray.shape
    gy, gx = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)

    # Build a LUT for every tile.
    luts = np.zeros((gy, gx, 256), dtype=np.float64)
    cy = np.zeros(gy)
    cx = np.zeros(gx)
    for i in range(gy):
        cy[i] = 0.5 * (ys[i] + ys[i + 1])
        for j in range(gx):
            cx[j] = 0.5 * (xs[j] + xs[j + 1])
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            n_pix = tile.size
            clip_count = clip_map[i, j] * n_pix
            luts[i, j] = _tile_lut(tile, clip_count)

    # Bilinear interpolation of the four neighbouring tile-centre mappings.
    out = np.zeros_like(gray, dtype=np.float64)
    yy = np.arange(h, dtype=np.float64)
    xx = np.arange(w, dtype=np.float64)

    # Row tile indices / weights
    iy = np.clip(np.searchsorted(cy, yy) - 1, 0, gy - 2) if gy > 1 else np.zeros(h, int)
    ix = np.clip(np.searchsorted(cx, xx) - 1, 0, gx - 2) if gx > 1 else np.zeros(w, int)

    for r in range(h):
        if gy > 1:
            i0 = iy[r]; i1 = i0 + 1
            ty = (yy[r] - cy[i0]) / (cy[i1] - cy[i0] + 1e-12)
            ty = min(max(ty, 0.0), 1.0)
        else:
            i0 = i1 = 0; ty = 0.0
        row = gray[r]
        # Vectorised over the row:
        if gx > 1:
            j0 = ix; j1 = ix + 1
            tx = (xx - cx[j0]) / (cx[j1] - cx[j0] + 1e-12)
            tx = np.clip(tx, 0.0, 1.0)
        else:
            j0 = np.zeros(w, int); j1 = j0; tx = np.zeros(w)
        v = row
        m00 = luts[i0, j0, v]
        m01 = luts[i0, j1, v]
        m10 = luts[i1, j0, v]
        m11 = luts[i1, j1, v]
        top = m00 * (1 - tx) + m01 * tx
        bot = m10 * (1 - tx) + m11 * tx
        out[r] = top * (1 - ty) + bot * ty

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4. The proposed Auto-Clip CLAHE
# --------------------------------------------------------------------------- #
def compute_clip_map(gray, sigma, grid, alpha, beta, eps=1.0,
                     c_min=0.001, c_max=0.05, texture="entropy"):
    """Compute the per-tile adaptive clip limit C_{i,j}.

        C_{i,j} = alpha + beta * ( E_{i,j} / (sigma + eps) )

    The result is clamped to [c_min, c_max] for numerical stability.
    """
    if texture == "entropy":
        E = tile_entropy(gray, grid)
    elif texture == "variance":
        # Normalised local variance as an alternative texture descriptor.
        h, w = gray.shape
        gy, gx = grid
        E = np.zeros((gy, gx))
        ys = np.linspace(0, h, gy + 1).astype(int)
        xs = np.linspace(0, w, gx + 1).astype(int)
        for i in range(gy):
            for j in range(gx):
                t = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].astype(np.float64)
                E[i, j] = np.log1p(t.var())
    else:
        raise ValueError(texture)

    C = alpha + beta * (E / (sigma + eps))
    return np.clip(C, c_min, c_max), E


def auto_clip_clahe(bgr, grid=(8, 8), alpha=0.005, beta=0.004, eps=1.0,
                    c_min=0.001, c_max=0.05, texture="entropy",
                    return_info=False):
    """Apply the proposed Auto-Clip CLAHE to a colour (BGR) image.

    Enhancement is performed on the L channel in CIELAB space.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    sigma = estimate_noise_mad(L)
    clip_map, E = compute_clip_map(L, sigma, grid, alpha, beta, eps,
                                   c_min, c_max, texture)
    L_enh = clahe_per_tile(L, clip_map, grid)
    lab[:, :, 0] = L_enh
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if return_info:
        return out, {"sigma": sigma, "clip_map": clip_map, "entropy": E}
    return out


# --------------------------------------------------------------------------- #
# 5. Baselines
# --------------------------------------------------------------------------- #
def global_he(bgr):
    """Global Histogram Equalisation on the L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fixed_clahe(bgr, clip=0.01, grid=(8, 8)):
    """Fixed-parameter CLAHE using the same custom engine with a CONSTANT clip.

    `clip` is the normalised clip limit (fraction of tile pixels) applied to
    every tile, so the comparison against the adaptive method is apples-to-
    apples (same CLAHE engine, only the clip map differs).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clip_map = np.full(grid, clip, dtype=np.float64)
    lab[:, :, 0] = clahe_per_tile(L, clip_map, grid)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
