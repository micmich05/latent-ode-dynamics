"""Phase 4 / H2 with pixel observations: does decoder-free win when
reconstruction is genuinely expensive (2048 pixels vs 8 latent dims)?

Rendered pendulum (32x32, stacked frame pairs), regular sampling, sweep
pixel noise. Metric as in phase 3: readout-free ridge from rollout latents
to the true 2D state, mean ± std over seeds.

Usage: python3 experiments/phase4.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode.pixels import PixelJEPA, PixelLatentODE, make_pixel_dataset, train_pixel_model

CONTEXT = 10


@torch.no_grad()
def rolled_latents(model, data, split, device):
    obs = data[split]["obs"].to(device)
    times = data[split]["times"].to(device)
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    if isinstance(model, PixelLatentODE):
        z_c = model.encode_context(obs[:, :CONTEXT], times[:, :CONTEXT])
    else:
        z_c = model.encode(obs[:, CONTEXT - 1])
    return model.rollout(z_c, dts).cpu()


@torch.no_grad()
def state_space_metrics(model, data, device):
    z_tr = rolled_latents(model, data, "train", device)
    z_tr = z_tr.reshape(-1, z_tr.shape[-1]).double()
    z_te = rolled_latents(model, data, "test", device)
    z_te = z_te.reshape(-1, z_te.shape[-1]).double()
    s_tr = data["train"]["states"][:, CONTEXT:].reshape(-1, 2).double()
    s_te = data["test"]["states"][:, CONTEXT:].reshape(-1, 2).double()

    z_tr1 = torch.cat([z_tr, torch.ones(len(z_tr), 1, dtype=torch.float64)], dim=1)
    z_te1 = torch.cat([z_te, torch.ones(len(z_te), 1, dtype=torch.float64)], dim=1)
    reg = 1e-3 * torch.eye(z_tr1.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(z_tr1.T @ z_tr1 + reg, z_tr1.T @ s_tr)
    pred = z_te1 @ W
    rmse = ((pred - s_te) ** 2).mean().sqrt().item()
    ss_res = ((s_te - pred) ** 2).sum(0)
    ss_tot = ((s_te - s_te.mean(0)) ** 2).sum(0)
    return {"state_rmse": rmse, "state_r2": (1 - ss_res / ss_tot).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noises", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.17])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--d", type=int, default=6)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--n_train", type=int, default=256)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    results = {"args": vars(args), "sweep": []}
    for noise in args.noises:
        row = {"noise_std": noise, "models": {"ours": [], "lode": []}}
        for seed in args.seeds:
            data = make_pixel_dataset(n_train=args.n_train, noise_std=noise, seed=seed)
            row["noise_share"] = data["meta"]["noise_share"]

            torch.manual_seed(seed)
            t0 = time.time()
            # Tuned config (see phase-4 notes): EMA target encoder instead of
            # strong VICReg, long free-rollout supervision. VICReg-dominant
            # training reaches a bad equilibrium on pixels.
            ours = PixelJEPA(d=args.d, ema_decay=0.99)
            train_pixel_model(ours, data, device, "jepa", context=CONTEXT,
                              epochs=args.epochs, lr=1e-3, rollout_horizon=20,
                              lambda_var=0.1, lambda_cov=0.01, seed=seed, verbose=False)
            m = state_space_metrics(ours, data, device)
            row["models"]["ours"].append(m)
            print(f"noise={noise:.2f} seed={seed} ours state_rmse={m['state_rmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

            torch.manual_seed(seed)
            t0 = time.time()
            lode = PixelLatentODE(d=args.d)
            train_pixel_model(lode, data, device, "lode", context=CONTEXT,
                              epochs=args.epochs, seed=seed, verbose=False)
            m = state_space_metrics(lode, data, device)
            row["models"]["lode"].append(m)
            print(f"noise={noise:.2f} seed={seed} lode state_rmse={m['state_rmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results["sweep"].append(row)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / "phase4_pixels.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")

    print(f"\n{'noise':>6s} {'share':>6s} | {'ours (mean±std)':>18s} | {'lode (mean±std)':>18s}")
    for row in results["sweep"]:
        stats = {}
        for name in ("ours", "lode"):
            vals = torch.tensor([m["state_rmse"] for m in row["models"][name]])
            stats[name] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        print(f"{row['noise_std']:6.2f} {row['noise_share']:6.0%} | {stats['ours']:>18s} | {stats['lode']:>18s}")


if __name__ == "__main__":
    main()
