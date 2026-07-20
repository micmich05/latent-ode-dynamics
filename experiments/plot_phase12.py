"""Spectrum-recovery figure across 3 systems x 3 models x 5 seeds.

Usage: python3 experiments/plot_phase12.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"jepa": ("latent loss (JEPA)", "#2a78d6"),
          "lode": ("reconstruction only", "#008300"),
          "unified": ("unified", "#4a3aa7")}
INK = "#1a1a19"


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    data = json.loads((results_dir / "phase12_geometry.json").read_text())
    systems = data["args"]["systems"]

    fig, axes = plt.subplots(1, len(systems), figsize=(4.4 * len(systems), 4.2),
                             facecolor="#fcfcfb")
    for ax, system in zip(axes, systems):
        ax.set_facecolor("#fcfcfb")
        th = data["theory"][system]
        ax.scatter([th["real"]], [th["imag"]], marker="x", s=120, color=INK,
                   linewidths=2.5, label="theory", zorder=5)
        for key, (label, color) in MODELS.items():
            rs = [r["spectrum"] for r in data["runs"]
                  if r["system"] == system and r["model"] == key]
            ok = [r for r in rs if r["fp_residual"] < 1e-4]
            bad = [r for r in rs if r["fp_residual"] >= 1e-4]
            ax.scatter([r["real"] for r in ok], [r["imag"] for r in ok],
                       s=48, color=color, alpha=0.85, label=label,
                       edgecolors="#fcfcfb", linewidths=1)
            if bad:  # Newton failed: hollow markers
                ax.scatter([r["real"] for r in bad], [r["imag"] for r in bad],
                           s=48, facecolors="none", edgecolors=color,
                           linewidths=1.4, alpha=0.8)
        ax.axvline(0, color="#e5e4dd", lw=0.8, zorder=0)
        ax.set_title(system.replace("_", "-"), fontsize=10, color=INK)
        ax.set_xlabel("Re(λ)", fontsize=9)
        ax.grid(color="#efeee8", lw=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#3d3d3a", labelsize=8)
    axes[0].set_ylabel("Im(λ)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Leading eigenvalue at the field's fixed point — 5 seeds per model "
                 "(hollow = Newton failed to find a genuine zero)",
                 fontsize=11, color=INK)
    fig.tight_layout()
    out = results_dir / "phase12_spectra.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
