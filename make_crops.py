"""Round-2 addition: input vs Auto-Clip output crops at a noisy level,
to sit next to the clip-map figure. Imports existing code; no modification.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autoclahe import auto_clip_clahe
from metrics import reduce_contrast, add_gaussian_noise

W, H = 768, 512
GRID = (8, 8)
ALPHA, BETA = 0.0, 0.02
SIGMA = 20
rng = np.random.default_rng(2025)

im = cv2.imread("data/kodak/kodim19.png")
im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
base = reduce_contrast(im, factor=0.7)
noisy = add_gaussian_noise(base, SIGMA, rng=rng)
out = auto_clip_clahe(noisy, grid=GRID, alpha=ALPHA, beta=BETA)

def rgb(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

# Zoom crop: a textured region (fence / building edge) -> rows, cols
r0, r1, c0, c1 = 300, 440, 120, 300
def crop(b):
    return rgb(b[r0:r1, c0:c1])

fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.4))
ax[0, 0].imshow(rgb(noisy));  ax[0, 0].set_title(f"Noisy low-contrast input ($\\sigma_n={SIGMA}$)", fontsize=11)
ax[0, 1].imshow(rgb(out));    ax[0, 1].set_title("Auto-Clip output", fontsize=11)
# draw crop rectangle on the full views
for a in (ax[0, 0], ax[0, 1]):
    a.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, edgecolor="yellow", facecolor="none", lw=1.5))
ax[1, 0].imshow(crop(noisy)); ax[1, 0].set_title("input (zoom)", fontsize=10)
ax[1, 1].imshow(crop(out));   ax[1, 1].set_title("Auto-Clip (zoom)", fontsize=10)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig("fig_crops.png", dpi=150, bbox_inches="tight")
print("saved fig_crops.png", cv2.imread("fig_crops.png").shape if False else "")
