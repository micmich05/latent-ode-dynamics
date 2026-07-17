"""Phase 2 / H2: is the decoder-free latent more robust to observation noise?

Fix sampling irregularity at s=0.9 (where both continuous models handle dt)
and sweep observation noise. The Latent ODE's reconstruction loss forces its
latent+decoder to model the noise; our latent prediction loss has no such
incentive. RMSE is measured against the CLEAN (noise-free) observations —
both models only ever see the noisy ones.

Usage: python3 experiments/phase2.py --system oscillator
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.baselines import (LatentODEBaseline, obs_space_forecast_rmse,
                                 train_obs_space_model)
from latentode.eval import forecast_rmse, train_decoder_probe

CONTEXT = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator", choices=["oscillator", "lotka_volterra"])
    p.add_argument("--noises", type=float, nargs="+", default=[0.02, 0.1, 0.3, 0.6, 1.0])
    p.add_argument("--jitter", type=float, default=0.9)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results = {"args": vars(args), "sweep": []}
    for noise in args.noises:
        data = make_dataset(system=args.system, jitter=args.jitter,
                            noise_std=noise, seed=args.seed)
        # Noise share of total variance in normalized coordinates (signal+noise = 1).
        clean = data["test"]["obs_clean"]
        noisy = data["test"]["obs"]
        noise_share = ((noisy - clean).var() / noisy.var()).item()

        row = {"noise_std": noise, "noise_share": noise_share, "models": {}}

        torch.manual_seed(args.seed)
        t0 = time.time()
        ours = LatentODEJEPA(n_obs=50, d=args.d)
        train(ours, data, epochs=args.epochs, seed=args.seed, verbose=False)
        decoder = train_decoder_probe(ours, data, seed=args.seed)
        row["models"]["ours"] = forecast_rmse(ours, decoder, data, CONTEXT, target="obs_clean")
        print(f"noise={noise:.2f} (share {noise_share:.0%}) ours rmse={row['models']['ours']['mean_rmse']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        torch.manual_seed(args.seed)
        t0 = time.time()
        lode = LatentODEBaseline(n_obs=50, d=args.d)
        train_obs_space_model(lode, data, context=CONTEXT, epochs=args.epochs, seed=args.seed)
        row["models"]["lode"] = obs_space_forecast_rmse(lode, data, CONTEXT, target="obs_clean")
        print(f"noise={noise:.2f} (share {noise_share:.0%}) lode rmse={row['models']['lode']['mean_rmse']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        results["sweep"].append(row)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"phase2_{args.system}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")

    print(f"\n{'noise':>6s} {'share':>6s} | {'ours':>8s} | {'lode':>8s}")
    for row in results["sweep"]:
        print(f"{row['noise_std']:6.2f} {row['noise_share']:6.0%} | "
              f"{row['models']['ours']['mean_rmse']:8.4f} | {row['models']['lode']['mean_rmse']:8.4f}")


if __name__ == "__main__":
    main()
