"""Phase 7: does the unified model inherit the best of both parents?

Three validations against numbers already measured:
  A) dynamics accuracy  — phase-3 setting (jitter 0.9, noise sweep, readout-free
     state RMSE): should approach lode, beat JEPA-only.
  B) field geometry     — E1 setting: fixed point on-manifold with clean Newton
     convergence, eigenvalues near theory: should match JEPA-only, beat lode.
  C) anomaly detection  — phase-6 setting: latent-rollout + decoded-rollout
     hybrid score: should approach lode's AUROC.

Usage: python3 experiments/phase7_unified.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from latentode import make_dataset
from latentode.unified import UnifiedLatentODE, train_unified
from phase3 import state_space_metrics, CONTEXT
from phase5_field import THEORY, analyze
from phase6 import JITTER as ANOM_JITTER
from phase6 import build_split, evaluate, score_ours_rollout


@torch.no_grad()
def score_decoded_rollout(model, split):
    obs, times = split["obs"], split["times"]
    z_c = model.encode(obs[:, CONTEXT - 1])
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    preds = model.decoder(model.rollout(z_c, dts))
    return (preds - obs[:, CONTEXT:]).norm(dim=-1), CONTEXT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--noises", type=float, nargs="+", default=[0.02, 0.1, 0.3, 0.6, 1.0])
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()
    results = {"args": vars(args), "accuracy": [], "field": [], "anomaly": []}

    print("== A) accuracy (phase-3 setting) ==", flush=True)
    for noise in args.noises:
        vals = []
        for seed in args.seeds:
            data = make_dataset(system="oscillator", jitter=0.9, noise_std=noise, seed=seed)
            torch.manual_seed(seed)
            m = UnifiedLatentODE(n_obs=50, d=8)
            t0 = time.time()
            train_unified(m, data, epochs=args.epochs, seed=seed, verbose=False)
            sm = state_space_metrics(m, data)
            vals.append(sm["state_rmse"])
            print(f"  noise={noise:.2f} seed={seed} state_rmse={sm['state_rmse']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results["accuracy"].append({"noise_std": noise, "state_rmse": vals})

    print("== B) field geometry (E1 setting) ==", flush=True)
    for system in ("oscillator", "lotka_volterra"):
        for seed in args.seeds:
            data = make_dataset(system=system, jitter=0.5, seed=seed)
            torch.manual_seed(seed)
            m = UnifiedLatentODE(n_obs=50, d=8)
            train_unified(m, data, epochs=args.epochs, seed=seed, verbose=False)
            a = analyze(m, data, lambda: m.encode(data["train"]["obs"]))
            a.update(system=system, seed=seed)
            results["field"].append(a)
            print(f"  {system} seed={seed} eig={a['real']:+.4f} ± {a['imag']:.4f}i "
                  f"(res {a['fp_residual']:.1e}, dist/scale "
                  f"{a['fp_dist_to_data']/a['latent_scale']:.2f})", flush=True)

    print("== C) anomaly (phase-6 setting) ==", flush=True)
    for seed in args.seeds:
        data = make_dataset(system="oscillator", jitter=ANOM_JITTER, seed=seed)
        norm = data["norm"]
        calib = build_split(64, seed + 100, norm)
        test_normal = build_split(64, seed + 200, norm)
        anom = {a: build_split(64, seed + 300, norm, anomaly=a)
                for a in ("param", "impulse", "sensor")}
        torch.manual_seed(seed)
        m = UnifiedLatentODE(n_obs=50, d=8)
        train_unified(m, data, epochs=args.epochs, seed=seed, verbose=False)
        streams = [(m, score_ours_rollout), (m, score_decoded_rollout)]
        ev = evaluate(streams, calib, test_normal, anom)
        results["anomaly"].append({"seed": seed, "metrics": ev})
        print(f"  seed={seed} " + " | ".join(
            f"{a}: AUROC {v['auroc']:.3f}" for a, v in ev.items()), flush=True)

    (ROOT / "results" / "phase7_unified.json").write_text(json.dumps(results, indent=2))

    # Side-by-side with the already-measured parents.
    print("\n== A) state RMSE vs parents (phase3) ==")
    p3 = json.loads((ROOT / "results" / "phase3_oscillator.json").read_text())
    print(f"{'noise':>6s} | {'unified':>8s} | {'jepa':>8s} | {'lode':>8s}")
    for row, row3 in zip(results["accuracy"], p3["sweep"]):
        u = sum(row["state_rmse"]) / len(row["state_rmse"])
        o = sum(m["state_rmse"] for m in row3["models"]["ours"]) / 3
        l = sum(m["state_rmse"] for m in row3["models"]["lode"]) / 3
        print(f"{row['noise_std']:6.2f} | {u:8.4f} | {o:8.4f} | {l:8.4f}")

    print("\n== B) eigenvalues (theory / unified mean) ==")
    for system in ("oscillator", "lotka_volterra"):
        rs = [r for r in results["field"] if r["system"] == system]
        re_m = sum(r["real"] for r in rs) / len(rs)
        im_m = sum(r["imag"] for r in rs) / len(rs)
        th = THEORY[system]
        print(f"  {system}: theory {th['real']:+.3f} ± {th['imag']:.3f}i | "
              f"unified {re_m:+.3f} ± {im_m:.3f}i")

    print("\n== C) AUROC vs phase6 ==")
    p6 = json.loads((ROOT / "results" / "phase6_anomaly.json").read_text())
    for a in ("param", "impulse", "sensor"):
        u = sum(r["metrics"][a]["auroc"] for r in results["anomaly"]) / len(results["anomaly"])
        l = sum(r["metrics"][a]["auroc"] for r in p6["runs"] if r["model"] == "lode") / 3
        print(f"  {a}: unified {u:.3f} | lode {l:.3f}")


if __name__ == "__main__":
    main()
