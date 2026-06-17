"""Generate all publication figures from the saved result JSONs."""
import os, sys, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
sys.path.insert(0, "src")
from autoclahe import (auto_clip_clahe, fixed_clahe, global_he,
                       compute_clip_map, estimate_noise_mad)
import metrics as M
from experiment import load_base, make_noisy, GRID

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
os.makedirs("figures", exist_ok=True)
LEVELS = [0, 5, 10, 15, 20, 25]
COL = {"GHE": "#7f7f7f", "Fixed-CLAHE-low": "#1f77b4",
       "Fixed-CLAHE-high": "#d62728", "Auto-Clip": "#2ca02c"}
MK = {"GHE": "o", "Fixed-CLAHE-low": "s", "Fixed-CLAHE-high": "^", "Auto-Clip": "D"}


def agg_main():
    rows = json.load(open("results/main_eval.json"))
    a = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows:
        for k in ["dH", "BRISQUE", "PSNR", "SSIM"]:
            if r[k] is not None:
                a[r["method"]][r["sigma"]][k].append(r[k])
    return a


# --------------------------------------------------------------------------- #
# FIG 1: main metrics vs sigma (BRISQUE, entropy gain, SSIM)
# --------------------------------------------------------------------------- #
def fig_metrics():
    a = agg_main()
    methods = ["GHE", "Fixed-CLAHE-low", "Fixed-CLAHE-high", "Auto-Clip"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    # BRISQUE
    for m in methods:
        y = [np.mean(a[m][s]["BRISQUE"]) for s in LEVELS]
        ax[0].plot(LEVELS, y, marker=MK[m], color=COL[m], label=m, ms=4, lw=1.5)
    ax[0].set_title("(a) BRISQUE  (lower = better)")
    ax[0].set_xlabel(r"noise level $\sigma$"); ax[0].set_ylabel("BRISQUE")
    # Entropy gain
    for m in methods:
        y = [np.mean(a[m][s]["dH"]) for s in LEVELS]
        ax[1].plot(LEVELS, y, marker=MK[m], color=COL[m], label=m, ms=4, lw=1.5)
    ax[1].set_title("(b) Entropy gain $\\Delta H$  (higher = better)")
    ax[1].set_xlabel(r"noise level $\sigma$"); ax[1].set_ylabel(r"$\Delta H$ (bits)")
    # SSIM
    lv = LEVELS[1:]
    for m in methods:
        y = [np.mean(a[m][s]["SSIM"]) for s in lv]
        ax[2].plot(lv, y, marker=MK[m], color=COL[m], label=m, ms=4, lw=1.5)
    ax[2].set_title("(c) SSIM vs. clean-enhanced (robustness)")
    ax[2].set_xlabel(r"noise level $\sigma$"); ax[2].set_ylabel("SSIM")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("figures/fig_metrics.png", bbox_inches="tight")
    plt.close(fig); print("saved fig_metrics.png")


# --------------------------------------------------------------------------- #
# FIG 2: grid-search heatmap of objective J
# --------------------------------------------------------------------------- #
def fig_gridsearch():
    gs = json.load(open("results/grid_search.json"))
    alphas, betas = gs["alphas"], gs["betas"]
    J = np.array(gs["J"]).reshape(len(alphas), len(betas))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    im = ax.imshow(J, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels(betas)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel(r"$\beta$ (scaling gain)"); ax.set_ylabel(r"$\alpha$ (baseline)")
    bi = gs["combos"].index(gs["best"])
    by, bx = divmod(bi, len(betas))
    ax.scatter([bx], [by], marker="*", s=220, c="white", edgecolor="k", zorder=3)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                    color="w" if J[i, j] < J.mean() else "k", fontsize=7)
    ax.set_title(r"Objective $J=G-\lambda P$  (★ = optimum)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="J")
    fig.tight_layout()
    fig.savefig("figures/fig_gridsearch.png", bbox_inches="tight")
    plt.close(fig); print("saved fig_gridsearch.png")


# --------------------------------------------------------------------------- #
# FIG 3: clip-map adaptivity (image + per-tile clip map at sigma 0 vs 25)
# --------------------------------------------------------------------------- #
def fig_clipmap(name="kodim19"):
    base = load_base(name)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    ax[0].imshow(rgb); ax[0].set_title("(a) low-contrast input"); ax[0].axis("off")
    vmax = 0
    maps = {}
    for sig in [0, 25]:
        inp = make_noisy(base, name, sig)
        L = cv2.cvtColor(inp, cv2.COLOR_BGR2LAB)[:, :, 0]
        s = estimate_noise_mad(L)
        cmap, _ = compute_clip_map(L, s, GRID, 0.0, 0.02)
        maps[sig] = (cmap, s)
        vmax = max(vmax, cmap.max())
    for k, sig in enumerate([0, 25]):
        cmap, s = maps[sig]
        im = ax[k + 1].imshow(cmap, cmap="inferno", vmin=0, vmax=vmax)
        ax[k + 1].set_title(f"(b{k}) clip map  $\\sigma$={sig}, $\\hat\\sigma$={s:.1f}")
        ax[k + 1].set_xticks([]); ax[k + 1].set_yticks([]); ax[k + 1].grid(False)
        fig.colorbar(im, ax=ax[k + 1], fraction=0.046, pad=0.04,
                     label="clip limit $C_{i,j}$")
    fig.tight_layout()
    fig.savefig("figures/fig_clipmap.png", bbox_inches="tight")
    plt.close(fig); print("saved fig_clipmap.png")


# --------------------------------------------------------------------------- #
# FIG 4: ablation (BRISQUE vs sigma for the three variants)
# --------------------------------------------------------------------------- #
def fig_ablation():
    ab = json.load(open("results/ablations.json"))
    labels = {"entropy": "Full (entropy + noise term)",
              "variance": "Variance texture",
              "no_noise_term": r"No noise term ($C=\alpha+\beta E$)"}
    cols = {"entropy": "#2ca02c", "variance": "#9467bd", "no_noise_term": "#d62728"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    for v in ["entropy", "variance", "no_noise_term"]:
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in ab[v]:
            by[r["sigma"]]["dH"].append(r["dH"]); by[r["sigma"]]["BR"].append(r["BRISQUE"])
        ax[0].plot(LEVELS, [np.mean(by[s]["BR"]) for s in LEVELS], marker="o",
                   color=cols[v], label=labels[v], ms=4, lw=1.5)
        ax[1].plot(LEVELS, [np.mean(by[s]["dH"]) for s in LEVELS], marker="o",
                   color=cols[v], label=labels[v], ms=4, lw=1.5)
    ax[0].set_title("(a) BRISQUE (lower = better)")
    ax[0].set_xlabel(r"$\sigma$"); ax[0].set_ylabel("BRISQUE")
    ax[1].set_title(r"(b) Entropy gain $\Delta H$")
    ax[1].set_xlabel(r"$\sigma$"); ax[1].set_ylabel(r"$\Delta H$ (bits)")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig("figures/fig_ablation.png", bbox_inches="tight")
    plt.close(fig); print("saved fig_ablation.png")


# --------------------------------------------------------------------------- #
# FIG 5: qualitative crops at a fixed noise level
# --------------------------------------------------------------------------- #
def fig_qualitative(name="kodim19", sig=20):
    base = load_base(name)
    inp = make_noisy(base, name, sig)
    outs = {
        "Noisy input": inp,
        "GHE": global_he(inp),
        "Fixed-CLAHE-high": fixed_clahe(inp, 0.04, GRID),
        "Auto-Clip": auto_clip_clahe(inp, grid=GRID, alpha=0.0, beta=0.02),
    }
    # crop a flat-sky region (top) where noise amplification shows
    y0, y1, x0, x1 = 20, 180, 60, 300
    fig, ax = plt.subplots(2, 4, figsize=(11, 4.6))
    for k, (lbl, im) in enumerate(outs.items()):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax[0, k].imshow(rgb); ax[0, k].set_title(lbl, fontsize=9)
        ax[0, k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                         ec="yellow", fc="none", lw=1.2))
        ax[0, k].axis("off")
        ax[1, k].imshow(rgb[y0:y1, x0:x1]); ax[1, k].axis("off")
        ax[1, k].set_title("sky crop (noise)", fontsize=8)
    fig.suptitle(f"Qualitative comparison on {name}, $\\sigma$={sig}", y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig_qualitative.png", bbox_inches="tight")
    plt.close(fig); print("saved fig_qualitative.png")


if __name__ == "__main__":
    fig_metrics()
    fig_gridsearch()
    fig_clipmap()
    fig_ablation()
    fig_qualitative()
    print("All figures generated.")
