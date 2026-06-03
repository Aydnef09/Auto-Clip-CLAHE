# Auto-Clip CLAHE

**Noise-Aware Automatic Clip-Limit Selection for Adaptive Contrast Enhancement**

COMP430 — Digital Image Processing, final term project.
Author: Yağızefe Aydın (Student ID: 2211051071) — yagizefe.aydin@agu.edu.tr

**Repository:** https://github.com/Aydnef09/Auto-Clip-CLAHE

```bash
git clone https://github.com/Aydnef09/Auto-Clip-CLAHE.git
```

---

## Overview

Contrast Limited Adaptive Histogram Equalization (CLAHE) is one of the most
widely used local contrast-enhancement techniques, but its key parameter — the
**clip limit** — is normally fixed by hand. A clip limit tuned for clean images
amplifies sensor noise into visible artifacts on noisy images, while a clip
limit chosen to be safe under noise leaves clean images under-enhanced.

**Auto-Clip CLAHE** removes this manual tuning by choosing a **per-tile** clip
limit automatically from two measured image statistics:

1. the **global noise floor** `σ`, estimated robustly via the wavelet
   median-absolute-deviation (MAD) estimator (Donoho & Johnstone, 1994), and
2. the **local Shannon entropy** `E_{i,j}` of each tile (texture energy).

The per-tile clip limit is

```
C_{i,j} = clip( α + β · E_{i,j} / (σ + ε),  C_min, C_max )
```

so enhancement grows with local texture and shrinks as the estimated noise
rises. The method enhances aggressively on clean, detailed regions and backs off
automatically as noise increases — with no parameter changes.

## Key result

On the 24-image Kodak suite, degraded with reduced contrast and additive
Gaussian noise at six levels (σ ∈ {0,…,25}):

| | clean (σ=0) | heavy noise (σ=25) |
|---|---|---|
| **Entropy gain ΔH** (higher = more enhancement) | 1.16 (≈ aggressive CLAHE's 1.18) | 0.48 (backs off) |
| **BRISQUE** (lower = fewer artifacts) | 22.8 | **71.0 — lowest of any enhancer** |

No single fixed clip limit reproduces this profile: the aggressive setting is
best on clean images but worst under noise (BRISQUE 85.8 @ σ=25); the
conservative setting is the reverse. Auto-Clip gets the best of both by
transitioning automatically. An ablation confirms the **noise term is the source
of the robustness**: removing it collapses the method to the over-enhancing
fixed-high baseline.

## Repository structure

```
auto-clip-clahe/
├── README.md
├── requirements.txt
├── src/
│   ├── autoclahe.py      # noise estimation, entropy, per-tile CLAHE engine, the method + baselines
│   ├── metrics.py        # PSNR, SSIM, BRISQUE, entropy, degradation helpers
│   ├── experiment.py     # grid search, main evaluation, ablations
│   └── make_figures.py   # regenerates all paper figures
└── paper/
    ├── main.tex          # IEEE conference paper
    ├── references.bib    # references (verified DOIs)
    ├── IEEEtran.cls      # IEEE class file
    ├── main.pdf          # compiled paper
    └── figures/          # generated figures
```

## Method components (`src/autoclahe.py`)

- `estimate_noise_mad(gray)` — global noise σ from the diagonal (HH) sub-band of
  a single-level Haar DWT: `σ = median(|HH − median(HH)|) / 0.6745`.
- `tile_entropy(gray, grid)` — per-tile Shannon entropy `E_{i,j}`.
- `clahe_per_tile(gray, clip_map, grid)` — a custom CLAHE that accepts a
  **different** clip limit per tile (OpenCV's CLAHE only supports a single global
  clip limit), with bilinear interpolation between tile mappings.
- `compute_clip_map(...)` — builds `C_{i,j}` from Eq. above.
- `auto_clip_clahe(bgr, ...)` — the proposed method (enhances the L channel in
  CIELAB; chrominance is preserved).
- Baselines: `global_he`, `fixed_clahe(clip=…)`.

## Setup

Requires Python 3.10+ and `opencv-contrib-python` (for the BRISQUE quality
module).

```bash
pip install -r requirements.txt
```

Download the 24 Kodak images into `data/kodak/` as `kodim01.png … kodim24.png`,
and place the BRISQUE model files (`brisque_model_live.yml`,
`brisque_range_live.yml`, from the opencv_contrib repository) into
`brisque_model/`.

## Usage

Enhance a single image:

```python
import cv2
from src.autoclahe import auto_clip_clahe

img = cv2.imread("input.png")
out = auto_clip_clahe(img, grid=(8, 8), alpha=0.0, beta=0.02)
cv2.imwrite("enhanced.png", out)
```

Reproduce the full study (writes JSON results to `results/`):

```bash
python src/experiment.py          # grid search + main eval + ablations
python src/make_figures.py        # regenerate all figures from results/
```

All randomness is seeded (`SEED = 2025`) and every (image, noise) pair uses a
single shared noisy realisation, so all method comparisons are paired and the
reported numbers are reproducible.

## Building the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Calibrated parameters

`(α, β) = (0.0, 0.02)`, selected by a grid search on a six-image calibration
subset using the parameter-free objective `J = G − λP` (mean entropy gain minus
a noise-quality penalty, with `λ = std(G)/std(P)`). Other fixed settings:
`ε = 1`, clip clamp `[0.001, 0.05]`, `8×8` tile grid.

## Note on tools

An AI assistant (Anthropic Claude) was used to help structure the code, debug
the pipeline, and draft/edit the paper. All numerical results and figures were
produced by the code in this repository on the stated dataset, and all cited
references were checked against their primary sources.

## License

MIT
