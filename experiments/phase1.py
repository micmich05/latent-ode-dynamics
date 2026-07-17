"""Phase 1: does the ODE inductive bias pay off as sampling gets irregular? (H1)

Sweep sampling irregularity s (dt ~ dt_base * U(1-s, 1+s)) and compare
forecast RMSE (context=10, forecast the remaining 50 steps) across:

  - ours:  latent-ODE JEPA (latent loss, continuous integration)
  - djepa: discrete JEPA   (latent loss, MLP predictor with dt as feature)
  - gru:   GRU forecaster  (obs-space loss, autoregressive, dt as feature)
  - lode:  deterministic Latent ODE (obs-space recon loss through a decoder)

H1 predicts the gap between ours and the discrete models grows with s.

Usage: python3 experiments/phase1.py --system oscillator
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.baselines import (DiscreteJEPA, GRUForecaster, LatentODEBaseline,
                                 obs_space_forecast_rmse, train_obs_space_model)
from latentode.eval import forecast_rmse, train_decoder_probe

CONTEXT = 10


def run_latent_model(model, data, seed, epochs):
    train(model, data, epochs=epochs, seed=seed, verbose=False)
    decoder = train_decoder_probe(model, data, seed=seed)
    return forecast_rmse(model, decoder, data, context=CONTEXT)


def run_obs_model(model, data, seed, epochs):
    train_obs_space_model(model, data, context=CONTEXT, epochs=epochs, seed=seed)
    return obs_space_forecast_rmse(model, data, context=CONTEXT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator", choices=["oscillator", "lotka_volterra"])
    p.add_argument("--jitters", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_obs = 50
    builders = {
        "ours": lambda: LatentODEJEPA(n_obs=n_obs, d=args.d),
        "djepa": lambda: DiscreteJEPA(n_obs=n_obs, d=args.d),
        "gru": lambda: GRUForecaster(n_obs=n_obs),
        "lode": lambda: LatentODEBaseline(n_obs=n_obs, d=args.d),
    }
    runners = {"ours": run_latent_model, "djepa": run_latent_model,
               "gru": run_obs_model, "lode": run_obs_model}

    results = {"args": vars(args), "sweep": []}
    for s in args.jitters:
        data = make_dataset(system=args.system, jitter=s, seed=args.seed)
        row = {"jitter": s, "models": {}}
        for name, build in builders.items():
            torch.manual_seed(args.seed)
            t0 = time.time()
            fc = runners[name](build(), data, args.seed, args.epochs)
            row["models"][name] = fc
            print(f"jitter={s:.1f} {name:6s} rmse={fc['mean_rmse']:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        results["sweep"].append(row)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"phase1_{args.system}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")

    print(f"\n{'jitter':>6s} | " + " | ".join(f"{m:>8s}" for m in builders))
    for row in results["sweep"]:
        print(f"{row['jitter']:6.1f} | " + " | ".join(
            f"{row['models'][m]['mean_rmse']:8.4f}" for m in builders))


if __name__ == "__main__":
    main()
