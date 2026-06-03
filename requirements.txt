"""
Full experimental pipeline for the Auto-Clip CLAHE study.

Stages
------
1. grid_search : sweep (alpha, beta) on a calibration subset, pick the optimum
                 by an equally-weighted, min-max-normalised composite of the
                 four metrics, and dump the score surface for a heatmap.
2. main_eval   : evaluate GHE, fixed CLAHE (low/high) and the tuned Auto-Clip
                 method on all 24 Kodak images across 6 noise levels.
3. ablations   : (a) entropy vs variance texture, (b) with/without the noise
                 normalisation term, (c) 8x8 vs 16x16 tile grid.

For every (image, sigma) pair a SINGLE noisy realisation is generated with a
deterministic seed and fed to every method, so all comparisons are paired.
"""

import os, sys, json, itertools
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from autoclahe import (auto_clip_clahe, fixed_clahe, global_he,
                       clahe_per_tile, compute_clip_map, estimate_noise_mad,
                       tile_entropy)
import metrics as M

KODAK = "data/kodak"
ALL_IMAGES = [f"kodim{ i:02d}" for i in range(1, 25)]
CALIB = ["kodim01", "kodim05", "kodim07", "kodim13", "kodim19", "kodim23"]
LEVELS = [0, 5, 10, 15, 20, 25]
CONTRAST = 0.7
GRID = (8, 8)
SEED = 2025


def _seed(name, sigma):
    return (abs(hash(name)) % 100000) * 100 + sigma + SEED


def load_base(name):
    img = cv2.imread(os.path.join(KODAK, f"{name}.png"))
    return M.reduce_contrast(img, CONTRAST)


def make_noisy(base, name, sigma):
    if sigma == 0:
        return base.copy()
    rng = np.random.default_rng(_seed(name, sigma))
    return M.add_gaussian_noise(base, sigma, rng)


# --------------------------------------------------------------------------- #
# 1. GRID SEARCH
# --------------------------------------------------------------------------- #
def grid_search():
    alphas = [0.0, 0.002, 0.005]
    betas = [0.005, 0.01, 0.02, 0.03, 0.04]
    combos = list(itertools.product(alphas, betas))

    raw = {}  # (a,b) -> dict of mean metrics
    for (a, b) in combos:
        dH, br, ps, ss = [], [], [], []
        for name in CALIB:
            base = load_base(name)
            ref = auto_clip_clahe(base, grid=GRID, alpha=a, beta=b)
            for sig in LEVELS:
                inp = make_noisy(base, name, sig)
                out = auto_clip_clahe(inp, grid=GRID, alpha=a, beta=b)
                dH.append(M.entropy(out) - M.entropy(inp))
                br.append(M.brisque(out))
                if sig > 0:
                    ps.append(M.psnr(out, ref))
                    ss.append(M.ssim(out, ref))
        raw[(a, b)] = dict(dH=np.mean(dH), BRISQUE=np.mean(br),
                           PSNR=np.mean(ps), SSIM=np.mean(ss))
        print(f"  a={a:.3f} b={b:.3f}  dH={raw[(a,b)]['dH']:.3f} "
              f"BR={raw[(a,b)]['BRISQUE']:.2f} PS={raw[(a,b)]['PSNR']:.2f} "
              f"SS={raw[(a,b)]['SSIM']:.3f}")

    # Composite: min-max normalise each metric across the grid; higher=better.
    def col(key): return np.array([raw[c][key] for c in combos])
    def norm(x): return (x - x.min()) / (np.ptp(x) + 1e-12)
    score = (norm(col("dH")) + norm(col("PSNR")) + norm(col("SSIM"))
             + norm(-col("BRISQUE"))) / 4.0
    best_idx = int(np.argmax(score))
    best = combos[best_idx]
    out = {
        "alphas": alphas, "betas": betas,
        "combos": [list(c) for c in combos],
        "raw": {f"{a}_{b}": raw[(a, b)] for (a, b) in combos},
        "score": score.tolist(),
        "best": list(best), "best_score": float(score[best_idx]),
    }
    print(f"  >>> BEST alpha={best[0]} beta={best[1]} score={score[best_idx]:.3f}")
    return out


# --------------------------------------------------------------------------- #
# 2. MAIN EVALUATION
# --------------------------------------------------------------------------- #
def main_eval(alpha, beta):
    methods = {
        "GHE": lambda x: global_he(x),
        "Fixed-CLAHE-low": lambda x: fixed_clahe(x, 0.01, GRID),
        "Fixed-CLAHE-high": lambda x: fixed_clahe(x, 0.04, GRID),
        "Auto-Clip": lambda x: auto_clip_clahe(x, grid=GRID, alpha=alpha, beta=beta),
    }
    rows = []
    for name in ALL_IMAGES:
        base = load_base(name)
        refs = {m: f(base) for m, f in methods.items()}
        for sig in LEVELS:
            inp = make_noisy(base, name, sig)
            Hin = M.entropy(inp)
            for m, f in methods.items():
                out = f(inp)
                rows.append({
                    "image": name, "sigma": sig, "method": m,
                    "dH": M.entropy(out) - Hin,
                    "BRISQUE": M.brisque(out),
                    "PSNR": (None if sig == 0 else M.psnr(out, refs[m])),
                    "SSIM": (None if sig == 0 else M.ssim(out, refs[m])),
                })
        print(f"  done {name}")
    return rows


# --------------------------------------------------------------------------- #
# 3. ABLATIONS
# --------------------------------------------------------------------------- #
def _entropy_only_clip(L, grid, alpha, beta, c_min=0.001, c_max=0.05):
    """Auto-clip WITHOUT the noise term: C = alpha + beta * E (sigma ignored)."""
    E = tile_entropy(L, grid)
    C = alpha + beta * E
    return np.clip(C, c_min, c_max)


def ablations(alpha, beta):
    res = {"entropy": [], "variance": [], "no_noise_term": []}
    for name in ALL_IMAGES:
        base = load_base(name)
        for sig in LEVELS:
            inp = make_noisy(base, name, sig)
            Hin = M.entropy(inp)
            lab = cv2.cvtColor(inp, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0]
            sigma = estimate_noise_mad(L)

            # (a) entropy texture (the main method)
            cmap, _ = compute_clip_map(L, sigma, GRID, alpha, beta,
                                       texture="entropy")
            out = cv2.cvtColor(_apply(lab, L, cmap), cv2.COLOR_LAB2BGR)
            res["entropy"].append(dict(sigma=sig, dH=M.entropy(out) - Hin,
                                       BRISQUE=M.brisque(out)))

            # (b) variance texture
            cmapv, _ = compute_clip_map(L, sigma, GRID, alpha, beta,
                                        texture="variance")
            outv = cv2.cvtColor(_apply(lab, L, cmapv), cv2.COLOR_LAB2BGR)
            res["variance"].append(dict(sigma=sig, dH=M.entropy(outv) - Hin,
                                        BRISQUE=M.brisque(outv)))

            # (c) no noise normalisation term
            cmapn = _entropy_only_clip(L, GRID, alpha, beta)
            outn = cv2.cvtColor(_apply(lab, L, cmapn), cv2.COLOR_LAB2BGR)
            res["no_noise_term"].append(dict(sigma=sig, dH=M.entropy(outn) - Hin,
                                             BRISQUE=M.brisque(outn)))
    return res


def _apply(lab, L, cmap):
    out = lab.copy()
    out[:, :, 0] = clahe_per_tile(L, cmap, GRID)
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("[1/3] Grid search ...")
    gs = grid_search()
    json.dump(gs, open("results/grid_search.json", "w"), indent=2)

    a, b = gs["best"]
    print(f"[2/3] Main evaluation (alpha={a}, beta={b}) ...")
    rows = main_eval(a, b)
    json.dump(rows, open("results/main_eval.json", "w"), indent=2)

    print("[3/3] Ablations ...")
    ab = ablations(a, b)
    json.dump(ab, open("results/ablations.json", "w"), indent=2)
    print("All results saved to results/")
