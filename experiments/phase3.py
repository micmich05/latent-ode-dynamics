"""Phase 3 / H2 in state space, multi-seed.

Obs-space RMSE conflates dynamics quality with readout quality (probe vs
end-to-end decoder). Here we compare the ROLLED-OUT latents of both models
directly against the true 2D state via a closed-form ridge probe: fit
z_rollout -> state on train, evaluate state RMSE and R^2 on test. No decoder
anywhere in the metric. 3 seeds per cell.

Usage: python3 experiments/phase3.py --system oscillator
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.baselines import LatentODEBaseline, train_obs_space_model

CONTEXT = 10


@torch.no_grad()
def rolled_latents(model, data, split):
    """Latent trajectory each model actually uses to forecast: context -> rollout."""
    obs, times = data[split]["obs"], data[split]["times"]
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    if isinstance(model, LatentODEBaseline):
        z_c = model.encode_context(obs[:, :CONTEXT], times[:, :CONTEXT])
    else:
        z_c = model.encode(obs[:, CONTEXT - 1])
    return model.rollout(z_c, dts)  # [B, T-CONTEXT, d]


@torch.no_grad()
def state_space_metrics(model, data):
    """Ridge z_rollout -> true state, fit on train, eval on test. Readout-free."""
    z_tr = rolled_latents(model, data, "train")
    z_tr = z_tr.reshape(-1, z_tr.shape[-1])
    z_te = rolled_latents(model, data, "test")
    z_te = z_te.reshape(-1, z_te.shape[-1])
    s_tr = data["train"]["states"][:, CONTEXT:].reshape(-1, 2)
    s_te = data["test"]["states"][:, CONTEXT:].reshape(-1, 2)

    z_tr1 = torch.cat([z_tr, torch.ones(len(z_tr), 1)], dim=1)
    z_te1 = torch.cat([z_te, torch.ones(len(z_te), 1)], dim=1)
    reg = 1e-3 * torch.eye(z_tr1.shape[1])
    W = torch.linalg.solve(z_tr1.T @ z_tr1 + reg, z_tr1.T @ s_tr)
    pred = z_te1 @ W
    rmse = ((pred - s_te) ** 2).mean().sqrt().item()
    ss_res = ((s_te - pred) ** 2).sum(0)
    ss_tot = ((s_te - s_te.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).tolist()
    return {"state_rmse": rmse, "state_r2": r2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator", choices=["oscillator", "lotka_volterra"])
    p.add_argument("--noises", type=float, nargs="+", default=[0.02, 0.1, 0.3, 0.6, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--jitter", type=float, default=0.9)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    results = {"args": vars(args), "sweep": []}
    for noise in args.noises:
        row = {"noise_std": noise, "models": {"ours": [], "lode": []}}
        for seed in args.seeds:
            data = make_dataset(system=args.system, jitter=args.jitter,
                                noise_std=noise, seed=seed)
            noise_share = (((data["test"]["obs"] - data["test"]["obs_clean"]).var()
                            / data["test"]["obs"].var()).item())
            row["noise_share"] = noise_share

            torch.manual_seed(seed)
            t0 = time.time()
            ours = LatentODEJEPA(n_obs=50, d=args.d)
            train(ours, data, epochs=args.epochs, seed=seed, verbose=False)
            m = state_space_metrics(ours, data)
            row["models"]["ours"].append(m)
            print(f"noise={noise:.2f} seed={seed} ours state_rmse={m['state_rmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

            torch.manual_seed(seed)
            t0 = time.time()
            lode = LatentODEBaseline(n_obs=50, d=args.d)
            train_obs_space_model(lode, data, context=CONTEXT, epochs=args.epochs, seed=seed)
            m = state_space_metrics(lode, data)
            row["models"]["lode"].append(m)
            print(f"noise={noise:.2f} seed={seed} lode state_rmse={m['state_rmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results["sweep"].append(row)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"phase3_{args.system}.json"
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
