"""Complex-plane figure: learned leading eigenvalue pairs vs theory.

Usage: python3 experiments/plot_phase5.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same identity colors as all previous figures.
MODELS = {"ours": ("latent-ODE JEPA (ours)", "#2a78d6"),
          "lode": ("latent ODE + decoder", "#008300")}
THEORY_COLOR = "#1a1a19"


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    data = json.loads((results_dir / "phase5_field.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), facecolor="#fcfcfb")
    for ax, system in zip(axes, ["oscillator", "lotka_volterra"]):
        ax.set_facecolor("#fcfcfb")
        th = data["theory"][system]
        ax.scatter([th["real"]], [th["imag"]], marker="x", s=110, color=THEORY_COLOR,
                   linewidths=2.5, label="theory", zorder=5)
        for key, (label, color) in MODELS.items():
            runs = [r for r in data["runs"] if r["system"] == system and r["model"] == key]
            ax.scatter([r["real"] for r in runs], [r["imag"] for r in runs],
                       s=52, color=color, alpha=0.85, label=label,
                       edgecolors="#fcfcfb", linewidths=1)
        ax.axvline(0, color="#e5e4dd", lw=0.8, zorder=0)
        ax.set_title(system.replace("_", "-"), fontsize=10, color="#1a1a19")
        ax.set_xlabel("Re(λ)  (damping)", fontsize=9)
        ax.grid(color="#efeee8", lw=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors="#3d3d3a", labelsize=8)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0.8, 2.8)
    axes[0].set_ylabel("Im(λ)  (frequency)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Leading eigenvalue of the learned latent field at its fixed point "
                 "(one dot per seed)", fontsize=11, color="#1a1a19")
    fig.tight_layout()
    out = results_dir / "phase5_eigenvalues.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
