"""Phase 5 / E1: is the learned latent field a GENUINE dynamical system?

Eigenvalues of the linearization at a fixed point are invariant under smooth
conjugacy (change of coordinates). So if g_phi truly learned the dynamics, the
Jacobian at its fixed point must reproduce the true system's eigenvalues —
recovering physical constants from the latent space:

  oscillator (omega=2, gamma=0.15):  lambda = -gamma/2 +- i*sqrt(omega^2 - gamma^2/4)
                                            = -0.075 +- 2.0i
  lotka_volterra (a=1.5, g=3):       lambda = +- i*sqrt(a*g) = +- 2.121i

Procedure per (model, system, seed): train, find z* solving g(z*)=0 by damped
Newton from the latent data mean, autograd Jacobian, report the leading
complex-conjugate eigenvalue pair. Sanity: z* must lie near the data manifold.

Usage: python3 experiments/phase5_field.py
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

THEORY = {
    "oscillator": {"real": -0.075, "imag": 1.9986},
    "lotka_volterra": {"real": 0.0, "imag": 2.1213},
}


def find_fixed_point(field, z0, iters=200, damping=0.5):
    """Damped Newton on g(z) = 0 starting from z0."""
    z = z0.clone().requires_grad_(False)
    for _ in range(iters):
        J = torch.autograd.functional.jacobian(field, z)
        g = field(z)
        step = torch.linalg.lstsq(J, -g).solution
        z = z + damping * step
        if g.norm() < 1e-8:
            break
    return z, field(z).norm().item()


def leading_pair(field, z_star):
    """Eigenvalues of the Jacobian at z*; return the complex pair with the
    largest |imag| (the oscillatory tangent plane), plus the full spectrum."""
    J = torch.autograd.functional.jacobian(field, z_star)
    ev = torch.linalg.eigvals(J)
    order = ev.imag.abs().argsort(descending=True)
    lead = ev[order[0]]
    return {"real": lead.real.item(), "imag": abs(lead.imag.item()),
            "spectrum": [[e.real.item(), e.imag.item()] for e in ev]}


def analyze(model, data, encode_all):
    with torch.no_grad():
        z_all = encode_all().reshape(-1, model.field.net[0].in_features)
    z0 = z_all.mean(0)
    z_star, residual = find_fixed_point(model.field, z0)
    dist = (z_all - z_star).norm(dim=1).min().item()
    scale = z_all.std(0).norm().item()
    out = leading_pair(model.field, z_star)
    out.update(fp_residual=residual, fp_dist_to_data=dist, latent_scale=scale)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["oscillator", "lotka_volterra"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--jitter", type=float, default=0.5)
    args = p.parse_args()

    results = {"args": vars(args), "theory": THEORY, "runs": []}
    for system in args.systems:
        for seed in args.seeds:
            data = make_dataset(system=system, jitter=args.jitter, seed=seed)

            torch.manual_seed(seed)
            t0 = time.time()
            ours = LatentODEJEPA(n_obs=50, d=args.d)
            train(ours, data, epochs=args.epochs, seed=seed, verbose=False)
            m = analyze(ours, data, lambda: ours.encode(data["train"]["obs"]))
            m.update(system=system, seed=seed, model="ours")
            results["runs"].append(m)
            print(f"{system} seed={seed} ours  eig={m['real']:+.4f} ± {m['imag']:.4f}i "
                  f"(res {m['fp_residual']:.1e}, dist {m['fp_dist_to_data']:.2f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

            torch.manual_seed(seed)
            t0 = time.time()
            lode = LatentODEBaseline(n_obs=50, d=args.d)
            train_obs_space_model(lode, data, epochs=args.epochs, seed=seed)

            def encode_lode():
                z_c = lode.encode_context(data["train"]["obs"][:, :10],
                                          data["train"]["times"][:, :10])
                dts = data["train"]["times"][:, 10:] - data["train"]["times"][:, 9:-1]
                return lode.rollout(z_c, dts)

            m = analyze(lode, data, encode_lode)
            m.update(system=system, seed=seed, model="lode")
            results["runs"].append(m)
            print(f"{system} seed={seed} lode  eig={m['real']:+.4f} ± {m['imag']:.4f}i "
                  f"(res {m['fp_residual']:.1e}, dist {m['fp_dist_to_data']:.2f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "phase5_field.json").write_text(json.dumps(results, indent=2))

    print("\n== Eigenvalue recovery (theory vs learned, mean over seeds) ==")
    for system in args.systems:
        th = THEORY[system]
        print(f"\n{system}: theory = {th['real']:+.4f} ± {th['imag']:.4f}i")
        for name in ("ours", "lode"):
            rs = [r for r in results["runs"] if r["system"] == system and r["model"] == name]
            re_m = sum(r["real"] for r in rs) / len(rs)
            im_m = sum(r["imag"] for r in rs) / len(rs)
            print(f"  {name}: {re_m:+.4f} ± {im_m:.4f}i")


if __name__ == "__main__":
    main()
