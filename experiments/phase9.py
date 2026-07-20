"""Phase 9: dynamics-consistency classification on REAL data
(CharacterTrajectories, irregularly subsampled) + open-set.

One UnifiedLatentODE per character class, trained only on that class's
series; assignment by lowest mean z-scored decoded-rollout residual.
Two characters ('w', 'z') are held out of ALL training as the unseen
regimes for open-set rejection. Baseline: supervised GRU classifier with
max-softmax rejection. Both thresholds at 95% in-distribution acceptance.

Usage: python3 experiments/phase9.py [--seeds 0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latentode.chartraj import make_splits
from latentode.model import mlp
from latentode.unified import UnifiedLatentODE, train_unified

CONTEXT = 8
UNSEEN_CHARS = ["w", "z"]


@torch.no_grad()
def residual_rollout(model, obs, times):
    z_c = model.encode(obs[:, CONTEXT - 1])
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    preds = model.decoder(model.rollout(z_c, dts))
    return (preds - obs[:, CONTEXT:]).norm(dim=-1)


@torch.no_grad()
def residual_onestep(model, obs, times):
    """Local test at every step: does this field explain this motion? No
    drift accumulation — the right scoring when whole series differ in class."""
    z = model.encode(obs)
    dts = times[:, 1:] - times[:, :-1]
    B, T, d = z.shape
    pred = model.step(z[:, :-1].reshape(-1, d), dts.reshape(-1)).reshape(B, T - 1, d)
    return (model.decoder(pred) - obs[:, 1:]).norm(dim=-1)


RESIDUALS = {"rollout": residual_rollout, "onestep": residual_onestep}
residual_series = residual_onestep  # default; overridden by --score


class ClassField:
    def __init__(self, n_obs, train_split, calib_split, seed, epochs, d):
        torch.manual_seed(seed)
        self.model = UnifiedLatentODE(n_obs=n_obs, d=d)
        data = {"train": train_split}
        train_unified(self.model, data, epochs=epochs, batch_size=64,
                      seed=seed, verbose=False)
        r = residual_series(self.model, calib_split["obs"], calib_split["times"])
        self.mu, self.sd = r.mean(0), r.std(0).clamp_min(1e-6)
        self.tau = self.score(calib_split["obs"], calib_split["times"]).quantile(0.95)

    def score(self, obs, times):
        """Mean |z| of the residual profile: a series belongs to a class if its
        residuals look TYPICAL for that class — suspiciously low is also
        evidence against (an easy/static series under a hard class's field
        z-scores far below that class's calibration and must not win)."""
        r = residual_series(self.model, obs, times)
        return ((r - self.mu) / self.sd).abs().mean(dim=1)


class GRUClassifier(nn.Module):
    def __init__(self, n_obs, n_classes, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_obs + 1, hidden, batch_first=True)
        self.head = mlp([hidden, hidden, n_classes])

    def forward(self, obs, times):
        dts = times[:, 1:] - times[:, :-1]
        dts = torch.cat([torch.zeros(obs.shape[0], 1), dts], dim=1)
        h, _ = self.gru(torch.cat([obs, dts[..., None]], dim=-1))
        return self.head(h[:, -1])


def run_seed(seed, epochs, d):
    data = make_splits(seed=seed)
    keys, n_obs = data["keys"], data["n_obs"]
    known = [c for c in range(len(keys)) if keys[c] not in UNSEEN_CHARS]
    unseen = [c for c in range(len(keys)) if keys[c] in UNSEEN_CHARS]

    t0 = time.time()
    fields = {}
    for c in known:
        fields[c] = ClassField(n_obs, data["train"][c], data["calib"][c],
                               seed, epochs, d)
    print(f"  seed={seed}: {len(known)} class fields trained ({time.time()-t0:.0f}s)",
          flush=True)

    # Supervised baseline on the same known classes.
    tr_obs = torch.cat([data["train"][c]["obs"] for c in known])
    tr_times = torch.cat([data["train"][c]["times"] for c in known])
    tr_y = torch.cat([torch.full((len(data["train"][c]["obs"]),), i)
                      for i, c in enumerate(known)])
    torch.manual_seed(seed)
    clf = GRUClassifier(n_obs, len(known))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(80):
        perm = torch.randperm(len(tr_obs))
        for i in range(0, len(tr_obs), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(tr_obs[idx], tr_times[idx]), tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cal_probs = torch.cat([
            F.softmax(clf(data["calib"][c]["obs"], data["calib"][c]["times"]), -1)
            .max(-1).values for c in known])
    clf_tau = cal_probs.quantile(0.05)

    # Closed-set accuracy over the known classes.
    n_gen = n_disc = total = 0
    for i, c in enumerate(known):
        obs, times = data["test"][c]["obs"], data["test"][c]["times"]
        scores = torch.stack([fields[k].score(obs, times) for k in known], dim=1)
        n_gen += (scores.argmin(dim=1) == i).sum().item()
        with torch.no_grad():
            n_disc += (clf(obs, times).argmax(-1) == i).sum().item()
        total += len(obs)

    # Open-set on the held-out characters.
    obs = torch.cat([data["test"][c]["obs"] for c in unseen])
    times = torch.cat([data["test"][c]["times"] for c in unseen])
    scores = torch.stack([fields[k].score(obs, times) for k in known], dim=1)
    taus = torch.stack([fields[k].tau for k in known])
    gen_reject = (scores > taus).all(dim=1).float().mean().item()
    with torch.no_grad():
        probs = F.softmax(clf(obs, times), -1).max(-1).values
    disc_reject = (probs < clf_tau).float().mean().item()

    return {"acc_gen": n_gen / total, "acc_disc": n_disc / total,
            "openset_reject_gen": gen_reject, "openset_reject_disc": disc_reject,
            "n_test": total, "n_unseen": len(obs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--d", type=int, default=12)
    p.add_argument("--score", default="onestep", choices=list(RESIDUALS))
    args = p.parse_args()

    global residual_series
    residual_series = RESIDUALS[args.score]

    results = {"args": vars(args), "unseen": UNSEEN_CHARS, "runs": []}
    for seed in args.seeds:
        r = run_seed(seed, args.epochs, args.d)
        results["runs"].append({"seed": seed, **r})
        print(f"  seed={seed} acc gen {r['acc_gen']:.3f} | disc {r['acc_disc']:.3f} | "
              f"openset reject gen {r['openset_reject_gen']:.0%} vs disc "
              f"{r['openset_reject_disc']:.0%}", flush=True)

    (ROOT / "results" / "phase9_chartraj.json").write_text(json.dumps(results, indent=2))
    print("\n== mean over seeds ==")
    for k in ("acc_gen", "acc_disc", "openset_reject_gen", "openset_reject_disc"):
        print(f"  {k:22s} {np.mean([r[k] for r in results['runs']]):.3f}")


if __name__ == "__main__":
    main()
