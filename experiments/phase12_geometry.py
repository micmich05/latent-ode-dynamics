"""Phase 12: robust demonstration of the on/off-manifold dissociation.

The claim to stress-test (finding from phase 5): WHERE the loss lives
determines WHERE the field is trustworthy. Reconstruction supervises the
field only along rollout trajectories; per-frame latent prediction
supervises it across the whole data manifold — and only the latter should
yield a field that behaves like a dynamical system off-manifold, which is
exactly where anomaly scores operate.

Two measurements, 3 systems x 3 models x 5 seeds:

A) Spectrum recovery (extends phase 5): now including Van der Pol, whose
   fixed point is UNSTABLE (theory: +0.75 +- 0.661i) and sits inside the
   limit cycle where data never lingers — recovering the instability sign is
   a strictly harder test than the attracting cases.

B) Off-manifold flow test (new): perturb on-manifold latents by eps latent
   scales in a random direction, integrate each model's field for 5 time
   units, and measure the final distance back to the data manifold. A
   well-behaved field pulls perturbed states back (or keeps them bounded); a
   field that was never supervised there lets them fly off.

Usage: python3 experiments/phase12_geometry.py [--seeds 0 1 2 3 4]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from latentode import LatentODEJEPA, UnifiedLatentODE, make_dataset, train
from latentode.baselines import LatentODEBaseline, train_obs_space_model
from latentode.model import integrate
from latentode.unified import train_unified
from phase5_field import analyze

THEORY = {
    "oscillator": {"real": -0.075, "imag": 1.9986, "stable": True},
    "lotka_volterra": {"real": 0.0, "imag": 2.1213, "stable": None},
    "van_der_pol": {"real": 0.75, "imag": 0.6614, "stable": False},
}
EPS_LIST = (0.5, 1.0, 2.0, 4.0)


@torch.no_grad()
def flow_test(model, z_data, seed, n_points=64, T=50, dt=0.1):
    """Perturb on-manifold latents by eps*scale, integrate, distance back."""
    g = torch.Generator().manual_seed(seed)
    scale = z_data.std(0).norm()
    z0 = z_data[torch.randperm(len(z_data), generator=g)[:n_points]]
    ref = z_data[torch.randperm(len(z_data), generator=g)[:1024]]
    out = {}
    for eps in EPS_LIST:
        u = torch.randn(n_points, z0.shape[1], generator=g)
        z = z0 + eps * scale * (u / u.norm(dim=1, keepdim=True))
        for _ in range(T):
            z = integrate(model.field, z, torch.full((len(z),), dt), 2, "rk4")
            z = torch.nan_to_num(z, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        d = torch.cdist(z, ref).min(dim=1).values / scale
        out[str(eps)] = {"median_dist": d.median().item(),
                         "frac_exploded": (d > 10).float().mean().item()}
    return out


def latents_of(model, data):
    if isinstance(model, LatentODEBaseline):
        with torch.no_grad():
            z_c = model.encode_context(data["train"]["obs"][:, :10],
                                       data["train"]["times"][:, :10])
            dts = data["train"]["times"][:, 10:] - data["train"]["times"][:, 9:-1]
            z = model.rollout(z_c, dts)
    else:
        with torch.no_grad():
            z = model.encode(data["train"]["obs"])
    return z.reshape(-1, z.shape[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+",
                   default=["oscillator", "lotka_volterra", "van_der_pol"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--d", type=int, default=8)
    args = p.parse_args()

    results = {"args": vars(args), "theory": THEORY, "runs": []}
    for system in args.systems:
        for seed in args.seeds:
            data = make_dataset(system=system, jitter=0.5, seed=seed)
            for name in ("jepa", "lode", "unified"):
                torch.manual_seed(seed)
                t0 = time.time()
                if name == "jepa":
                    model = LatentODEJEPA(n_obs=50, d=args.d)
                    train(model, data, epochs=args.epochs, seed=seed, verbose=False)
                elif name == "lode":
                    model = LatentODEBaseline(n_obs=50, d=args.d)
                    train_obs_space_model(model, data, epochs=args.epochs, seed=seed)
                else:
                    model = UnifiedLatentODE(n_obs=50, d=args.d)
                    train_unified(model, data, epochs=args.epochs, seed=seed, verbose=False)

                z_data = latents_of(model, data)
                spec = analyze(model, data, lambda: z_data)
                flow = flow_test(model, z_data, seed)
                rec = {"system": system, "seed": seed, "model": name,
                       "spectrum": spec, "flow": flow}
                results["runs"].append(rec)
                print(f"{system} seed={seed} {name:7s} "
                      f"eig={spec['real']:+.3f}±{spec['imag']:.3f}i "
                      f"res={spec['fp_residual']:.0e} "
                      f"dist={spec['fp_dist_to_data']/spec['latent_scale']:.1f} | "
                      f"flow eps=4: d={flow['4.0']['median_dist']:.1f} "
                      f"expl={flow['4.0']['frac_exploded']:.0%} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    (ROOT / "results" / "phase12_geometry.json").write_text(json.dumps(results, indent=2))

    print("\n== A) spectrum recovery (mean over seeds) ==")
    for system in args.systems:
        th = THEORY[system]
        print(f"\n{system}: theory {th['real']:+.3f} ± {th['imag']:.3f}i")
        for name in ("jepa", "lode", "unified"):
            rs = [r["spectrum"] for r in results["runs"]
                  if r["system"] == system and r["model"] == name]
            re_m = np.mean([r["real"] for r in rs])
            im_m = np.mean([r["imag"] for r in rs])
            ok = np.mean([r["fp_residual"] < 1e-4 for r in rs])
            dist = np.median([r["fp_dist_to_data"] / r["latent_scale"] for r in rs])
            print(f"  {name:7s} {re_m:+.3f} ± {im_m:.3f}i | Newton ok {ok:.0%} "
                  f"| fp dist (median) {dist:.1f}")

    print("\n== B) flow test: median final distance (scales) / % exploded, eps=4 ==")
    for name in ("jepa", "lode", "unified"):
        rs = [r["flow"]["4.0"] for r in results["runs"] if r["model"] == name]
        print(f"  {name:7s} dist {np.median([r['median_dist'] for r in rs]):.1f} "
              f"| exploded {np.mean([r['frac_exploded'] for r in rs]):.0%}")


if __name__ == "__main__":
    main()
