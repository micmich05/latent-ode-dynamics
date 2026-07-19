"""Phase 5 / E2 + E3: two more probes that g_phi is a genuine vector field.

E2 (interpolation at unseen times): a continuous model can be queried at ANY
time. Test trajectories live on a fine dt=0.05 grid; the model observes every
2nd sample (dt=0.1, its training regime) and must predict the state at the
held-out midpoints by integrating HALF an interval — a time offset it can
reach only through the vector field. Compared against no-dynamics baselines.

E3 (solver ablation): train identical models with Euler vs RK4 x substeps.
If forecast quality improves with integration order, g_phi is being used as a
genuine vector field rather than a one-step residual block.

Usage: python3 experiments/phase5_e23.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.data import RandomLift, generate_states
from latentode.model import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from phase3 import state_space_metrics


def ridge(z, s, reg=1e-3):
    z = z.reshape(-1, z.shape[-1]).double()
    s = s.reshape(-1, s.shape[-1]).double()
    z1 = torch.cat([z, torch.ones(len(z), 1, dtype=torch.float64)], dim=1)
    return torch.linalg.solve(z1.T @ z1 + reg * torch.eye(z1.shape[1], dtype=torch.float64),
                              z1.T @ s)


def apply_ridge(W, z):
    z = z.double()
    z1 = torch.cat([z, torch.ones(*z.shape[:-1], 1, dtype=torch.float64)], dim=-1)
    return z1 @ W


def e2_interpolation(seed, epochs=300, n_test=64):
    """Train on jittered data, evaluate midpoint-state prediction on a fine grid."""
    data = make_dataset(system="oscillator", jitter=0.5, seed=seed)
    torch.manual_seed(seed)
    model = LatentODEJEPA(n_obs=50, d=8)
    train(model, data, epochs=epochs, seed=seed, verbose=False)

    # Fine test grid: dt=0.05, observe every 2nd sample, midpoints held out.
    states, times = generate_states("oscillator", n_test, T=119, dt_base=0.05,
                                    jitter=0.0, seed=seed + 5000)
    states = torch.from_numpy(states)
    lift = RandomLift(n_obs=50)
    obs = lift(states)
    obs = obs + 0.02 * torch.randn(obs.shape, generator=torch.Generator().manual_seed(seed + 5000))
    obs = (obs - data["norm"]["mean"]) / data["norm"]["std"]

    obs_k = obs[:, 0::2]        # observed samples (dt = 0.1)
    s_mid = states[:, 1::2]     # held-out midpoint states
    K = s_mid.shape[1]

    with torch.no_grad():
        W = ridge(model.encode(data["train"]["obs"]), data["train"]["states"])
        z_k = model.encode(obs_k)                      # [B, K+1, d]
        z_flat = z_k[:, :K].reshape(-1, 8)
        z_mid_field = integrate(model.field, z_flat,
                                torch.full((len(z_flat),), 0.05),
                                model.n_substeps, model.method).reshape(-1, K, 8)

    preds = {
        "field (past only)": apply_ridge(W, z_mid_field),
        "hold last obs": apply_ridge(W, z_k[:, :K]),
        "latent midpoint (uses future)": apply_ridge(W, (z_k[:, :K] + z_k[:, 1:]) / 2),
    }
    return {name: ((p - s_mid.double()) ** 2).mean().sqrt().item()
            for name, p in preds.items()}


def e3_solver(seed, epochs=300):
    """Same model, different integrators; forecast state RMSE at jitter 0.9."""
    data = make_dataset(system="oscillator", jitter=0.9, seed=seed)
    out = {}
    for method, substeps in [("euler", 1), ("euler", 2), ("rk4", 1), ("rk4", 2)]:
        torch.manual_seed(seed)
        model = LatentODEJEPA(n_obs=50, d=8, method=method, n_substeps=substeps)
        train(model, data, epochs=epochs, seed=seed, verbose=False)
        m = state_space_metrics(model, data)
        out[f"{method}-{substeps}"] = m["state_rmse"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    results = {"e2": [], "e3": []}
    for seed in args.seeds:
        t0 = time.time()
        r = e2_interpolation(seed)
        results["e2"].append(r)
        print(f"E2 seed={seed} " + " | ".join(f"{k}: {v:.4f}" for k, v in r.items())
              + f" ({time.time()-t0:.0f}s)", flush=True)
    for seed in args.seeds:
        t0 = time.time()
        r = e3_solver(seed)
        results["e3"].append(r)
        print(f"E3 seed={seed} " + " | ".join(f"{k}: {v:.4f}" for k, v in r.items())
              + f" ({time.time()-t0:.0f}s)", flush=True)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    (results_dir / "phase5_e23.json").write_text(json.dumps(results, indent=2))

    print("\n== E2: midpoint-state RMSE (mean over seeds) ==")
    for k in results["e2"][0]:
        vals = [r[k] for r in results["e2"]]
        print(f"  {k:32s} {sum(vals)/len(vals):.4f}")
    print("\n== E3: forecast state RMSE by integrator (mean over seeds) ==")
    for k in results["e3"][0]:
        vals = [r[k] for r in results["e3"]]
        print(f"  {k:10s} {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
