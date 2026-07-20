"""Render the H1 gap figure from a phase1 sweep JSON.

Usage: python3 experiments/plot_phase1.py --system oscillator
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed slot order, one hue per model identity).
MODELS = {
    "ours": ("latent-ODE JEPA (ours)", "#2a78d6"),
    "djepa": ("discrete JEPA", "#1baf7a"),
    "gru": ("GRU", "#eda100"),
    "lode": ("latent ODE + decoder", "#008300"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator")
    args = p.parse_args()

    results_dir = Path(__file__).resolve().parents[1] / "results"
    data = json.loads((results_dir / f"phase1_{args.system}.json").read_text())
    jitters = [row["jitter"] for row in data["sweep"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, (label, color) in MODELS.items():
        cells = [row["models"][key] for row in data["sweep"]]
        # single-seed rows are dicts; multi-seed rows are lists of dicts
        per_seed = [[c["mean_rmse"]] if isinstance(c, dict)
                    else [fc["mean_rmse"] for fc in c] for c in cells]
        mean = [sum(v) / len(v) for v in per_seed]
        std = [(sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
               for v, m in zip(per_seed, mean)]
        ax.plot(jitters, mean, color=color, lw=2, marker="o", ms=6, label=label)
        if any(s > 0 for s in std):
            ax.fill_between(jitters, [m - s for m, s in zip(mean, std)],
                            [m + s for m, s in zip(mean, std)], color=color,
                            alpha=0.15, lw=0)
        ax.annotate(label, (jitters[-1], mean[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#3d3d3a")

    ax.set_xlabel("sampling irregularity s   (Δt ~ Δt₀·U(1−s, 1+s))", fontsize=10)
    ax.set_ylabel("forecast RMSE (50 steps, context 10)", fontsize=10)
    ax.set_title(f"Forecast error vs sampling irregularity — {args.system}",
                 fontsize=11, color="#1a1a19")
    ax.grid(axis="y", color="#e5e4dd", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#3d3d3a", labelsize=9)
    ax.set_xlim(min(jitters) - 0.03, max(jitters) + 0.42)
    ax.set_ylim(bottom=0)

    out = results_dir / f"phase1_{args.system}_gap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
