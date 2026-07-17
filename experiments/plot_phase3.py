"""Render the H2-in-state-space figure (mean ± std over seeds) from phase3 JSON.

Usage: python3 experiments/plot_phase3.py --system oscillator
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same identity colors as phases 1-2: color follows the entity.
MODELS = {
    "ours": ("latent-ODE JEPA (ours)", "#2a78d6"),
    "lode": ("latent ODE + decoder", "#008300"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator")
    args = p.parse_args()

    results_dir = Path(__file__).resolve().parents[1] / "results"
    data = json.loads((results_dir / f"phase3_{args.system}.json").read_text())
    shares = [row["noise_share"] * 100 for row in data["sweep"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, (label, color) in MODELS.items():
        vals = np.array([[m["state_rmse"] for m in row["models"][key]] for row in data["sweep"]])
        mean, std = vals.mean(axis=1), vals.std(axis=1)
        ax.plot(shares, mean, color=color, lw=2, marker="o", ms=6, label=label)
        ax.fill_between(shares, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.annotate(label, (shares[-1], mean[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#3d3d3a")

    ax.set_xlabel("observation noise (% of total variance)", fontsize=10)
    ax.set_ylabel("state RMSE via ridge probe (rollout latents)", fontsize=10)
    ax.set_title(f"Dynamics quality in state space at s=0.9 — {args.system}\n"
                 "(readout-free; band = ±1 std over 3 seeds)", fontsize=11, color="#1a1a19")
    ax.grid(axis="y", color="#e5e4dd", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#3d3d3a", labelsize=9)
    ax.set_xlim(-2, max(shares) + 24)
    ax.set_ylim(bottom=0)

    out = results_dir / f"phase3_{args.system}_state.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
