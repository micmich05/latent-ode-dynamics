"""Theoretical vs learned phase portraits, across systems.

For each system, trains a unified model on lifted noisy observations and
renders side by side:
  left  — the TRUE phase plane: vector field quiver + test trajectories.
  right — the LEARNED dynamics, projected: encoded test latents in the PCA
          plane, plus the learned field g_phi evaluated on a grid of the PCA
          plane (grid points lifted back to latent space through the PCA
          basis, g projected onto the same plane).

The learned panel is a projection of an 8-D field the model built having
never seen the true state — matching topology (spiral sink / nested cycles /
limit cycle) is the qualitative evidence that the dynamics law, not just the
trajectories, was learned.

Usage: python3 experiments/portraits.py [--systems oscillator lotka_volterra van_der_pol]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latentode import UnifiedLatentODE, make_dataset
from latentode.data import SYSTEMS
from latentode.unified import train_unified

TITLES = {"oscillator": "damped oscillator", "lotka_volterra": "Lotka-Volterra",
          "van_der_pol": "Van der Pol"}
INK, BLUE, GRID = "#1a1a19", "#2a78d6", "#c3c2b7"


def true_panel(ax, system, states):
    f = SYSTEMS[system]["f"]
    x0, x1 = states[..., 0].min(), states[..., 0].max()
    y0, y1 = states[..., 1].min(), states[..., 1].max()
    pad_x, pad_y = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    gx, gy = np.meshgrid(np.linspace(x0 - pad_x, x1 + pad_x, 18),
                         np.linspace(y0 - pad_y, y1 + pad_y, 18))
    uv = f(np.stack([gx, gy], axis=-1))
    ax.quiver(gx, gy, uv[..., 0], uv[..., 1], color=GRID, width=0.003,
              angles="xy")
    for i in range(min(8, len(states))):
        ax.plot(states[i, :, 0], states[i, :, 1], color=INK, lw=1.1, alpha=0.75)
        ax.plot(states[i, 0, 0], states[i, 0, 1], ".", color=INK, ms=5)


@torch.no_grad()
def learned_panel(ax, model, obs):
    z = model.encode(obs)                                  # [n, T, d]
    zf = z.reshape(-1, z.shape[-1])
    mean = zf.mean(0)
    _, _, V = torch.linalg.svd(zf - mean, full_matrices=False)
    P = V[:2]                                              # [2, d] PCA basis
    z2 = (z - mean) @ P.T                                  # trajectories in plane

    x0, x1 = z2[..., 0].min(), z2[..., 0].max()
    y0, y1 = z2[..., 1].min(), z2[..., 1].max()
    pad_x, pad_y = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    gx, gy = torch.meshgrid(torch.linspace(x0 - pad_x, x1 + pad_x, 18),
                            torch.linspace(y0 - pad_y, y1 + pad_y, 18),
                            indexing="xy")
    plane = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    z_grid = mean + plane @ P                              # lift plane -> latent
    g = model.field(z_grid) @ P.T                          # project field -> plane
    ax.quiver(gx.numpy(), gy.numpy(),
              g[:, 0].reshape(18, 18).numpy(), g[:, 1].reshape(18, 18).numpy(),
              color="#a9c7ec", width=0.003, angles="xy")
    for i in range(min(8, len(z2))):
        ax.plot(z2[i, :, 0], z2[i, :, 1], color=BLUE, lw=1.1, alpha=0.8)
        ax.plot(z2[i, 0, 0], z2[i, 0, 1], ".", color=BLUE, ms=5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+",
                   default=["oscillator", "lotka_volterra", "van_der_pol"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = len(args.systems)
    fig, axes = plt.subplots(n, 2, figsize=(9.4, 4.1 * n), facecolor="#fcfcfb")
    axes = axes.reshape(n, 2)

    for row, system in enumerate(args.systems):
        data = make_dataset(system=system, jitter=0.5, seed=args.seed)
        torch.manual_seed(args.seed)
        model = UnifiedLatentODE(n_obs=50, d=8)
        train_unified(model, data, epochs=args.epochs, seed=args.seed, verbose=False)
        print(f"{system}: trained", flush=True)

        states = data["test"]["states"].numpy()
        true_panel(axes[row, 0], system, states)
        learned_panel(axes[row, 1], model, data["test"]["obs"])
        axes[row, 0].set_title(f"{TITLES[system]} — true phase plane",
                               fontsize=10, color=INK)
        axes[row, 1].set_title("learned latent field (PCA projection)",
                               fontsize=10, color=INK)
        for ax in axes[row]:
            ax.set_facecolor("#fcfcfb")
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.tick_params(colors="#3d3d3a", labelsize=8)

    fig.tight_layout()
    out = ROOT / "results" / "portraits_true_vs_learned.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
