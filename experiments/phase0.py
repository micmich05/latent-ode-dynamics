"""Phase 0: sanity check on synthetic systems with regular sampling.

Success criteria:
  1. No collapse: effective rank >> 1 and stable through training.
  2. Informative latent: linear probe latent -> true state with high R^2.
  3. Usable for forecasting: rollout + decoder-probe RMSE well below the
     trivial predictor, approaching the reconstruction floor.
  4. Qualitative: latent phase portrait matches the true topology
     (spiral for damped oscillator, closed cycles for Lotka-Volterra).

Usage: python3 experiments/phase0.py --system oscillator
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.eval import forecast_rmse, linear_probe_state, phase_portrait, train_decoder_probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="oscillator", choices=["oscillator", "lotka_volterra"])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--method", default="rk4", choices=["euler", "rk4"])
    p.add_argument("--jitter", type=float, default=0.0, help="sampling irregularity s: dt ~ U(1-s, 1+s)*dt_base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    data = make_dataset(system=args.system, jitter=args.jitter, seed=args.seed)
    model = LatentODEJEPA(n_obs=data["meta"]["n_obs"], d=args.d, method=args.method)

    print(f"== Phase 0: {args.system} (jitter={args.jitter}, method={args.method}, d={args.d}) ==")
    history = train(model, data, epochs=args.epochs, seed=args.seed)

    r2 = linear_probe_state(model, data)
    decoder = train_decoder_probe(model, data, seed=args.seed)
    fc = forecast_rmse(model, decoder, data, context=10)

    # Trivial baseline: predict the last context observation forever.
    obs = data["test"]["obs"]
    persist_rmse = ((obs[:, 10:] - obs[:, 9:10]) ** 2).mean().sqrt().item()

    name = f"phase0_{args.system}{f'_j{args.jitter}' if args.jitter else ''}{args.tag}"
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    phase_portrait(model, data, results_dir / f"{name}_portrait.png")

    summary = {
        "args": vars(args),
        "eff_rank_final": history[-1]["eff_rank"],
        "state_probe_r2": r2,
        "forecast_mean_rmse": fc["mean_rmse"],
        "forecast_per_step_rmse": fc["per_step_rmse"],
        "recon_floor_rmse": fc["recon_rmse"],
        "persistence_baseline_rmse": persist_rmse,
        "history": history,
    }
    (results_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))

    print("\n== Results ==")
    print(f"effective rank (d={args.d}):      {history[-1]['eff_rank']:.2f}")
    print(f"state probe R^2 (dim0, dim1):     {r2[0]:.3f}, {r2[1]:.3f}")
    print(f"forecast RMSE (mean, 50 steps):   {fc['mean_rmse']:.4f}")
    print(f"  reconstruction floor:           {fc['recon_rmse']:.4f}")
    print(f"  persistence baseline:           {persist_rmse:.4f}")
    print(f"portrait: results/{name}_portrait.png")


if __name__ == "__main__":
    main()
