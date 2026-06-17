"""Round-2 addition: per-image runtime benchmark.

This script does NOT modify the project code; it imports the existing
functions and times them. Runtime of these operations is dominated by image
size (histogram building, LUTs, bilinear interpolation, wavelet transform),
not pixel content, so images are resized to the Kodak working resolution
512x768 to reflect the runtimes reported in the paper.
"""
import time, glob
import numpy as np
import cv2
from autoclahe import auto_clip_clahe, fixed_clahe, estimate_noise_mad
from metrics import reduce_contrast, add_gaussian_noise

W, H = 768, 512          # Kodak working resolution (cols, rows)
GRID = (8, 8)
ALPHA, BETA = 0.0, 0.02  # calibrated operating point
REPEATS = 5
rng = np.random.default_rng(2025)

# Standard library CLAHE (OpenCV C++), applied on the L channel.
def opencv_clahe(bgr, clip=2.0, grid=GRID):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

imgs = []
for f in sorted(glob.glob("data/kodak/kodim*.png")):
    im = cv2.imread(f)
    if im is None:
        continue
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    base = reduce_contrast(im, factor=0.7)
    noisy = add_gaussian_noise(base, 20, rng=rng)
    imgs.append(noisy)
print(f"benchmark images: {len(imgs)} at {W}x{H}")

def timeit(fn, images):
    per = []
    for im in images:
        t = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn(im)
            t.append((time.perf_counter() - t0) * 1000.0)  # ms
        per.append(min(t))   # best-of to reduce scheduler noise
    return np.array(per)

methods = {
    "OpenCV CLAHE (standard, C++)": lambda im: opencv_clahe(im),
    "Fixed-CLAHE (paper engine, Python)": lambda im: fixed_clahe(im, clip=0.02, grid=GRID),
    "Auto-Clip (ours, Python)": lambda im: auto_clip_clahe(im, grid=GRID, alpha=ALPHA, beta=BETA),
}

print(f"\n{'method':40s} {'mean ms':>9s} {'std':>7s}")
results = {}
for name, fn in methods.items():
    fn(imgs[0])  # warm-up
    per = timeit(fn, imgs)
    results[name] = (per.mean(), per.std())
    print(f"{name:40s} {per.mean():9.1f} {per.std():7.1f}")

# Breakdown of Auto-Clip's overhead (noise estimation only, vs full)
t_noise = []
for im in imgs:
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0]
    t = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); estimate_noise_mad(L); t.append((time.perf_counter()-t0)*1000)
    t_noise.append(min(t))
print(f"\nAuto-Clip noise-estimation step alone: {np.mean(t_noise):.2f} ms "
      f"({100*np.mean(t_noise)/results['Auto-Clip (ours, Python)'][0]:.1f}% of total)")

import json
json.dump({k: list(v) for k, v in results.items()} |
          {"noise_ms": float(np.mean(t_noise))}, open("runtime.json", "w"), indent=2)
print("\nsaved runtime.json")
