"""Phase 11: shared-encoder classifier on real data — the decisive test.

Same protocol as phases 9-10 (same splits, scoring, baselines) but with the
SharedDynamicsClassifier: one encoder/decoder for all classes, one field per
class. If HAR improves a lot while characters stay poor, the scope condition
(dynamics laws vs control programs) is confirmed clean of the encoding
artifact; if characters improve too, the condition gets refined.

Usage: python3 experiments/phase11.py --dataset har [--seeds 0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from latentode.shared import SharedDynamicsClassifier, train_shared
from phase9 import GRUClassifier

DATASETS = {
    "har": {"unseen": ["downstairs"]},
    "chartraj": {"unseen": ["w", "z"]},
}


def get_splits(dataset, seed):
    if dataset == "har":
        from latentode.har import make_splits
        return make_splits(seed=seed)
    from latentode.chartraj import make_splits
    return make_splits(seed=seed)


@torch.no_grad()
def residual_onestep(model, c, obs, times):
    z = model.encode(obs)
    dts = times[:, 1:] - times[:, :-1]
    B, T, d = z.shape
    pred = model.step(z[:, :-1].reshape(-1, d), dts.reshape(-1), c).reshape(B, T - 1, d)
    return (model.decoder(pred) - obs[:, 1:]).norm(dim=-1)


class ClassCal:
    """Per-class calibration of the typicality score |z| (as in phases 9-10)."""

    def __init__(self, model, c, calib):
        r = residual_onestep(model, c, calib["obs"], calib["times"])
        self.mu, self.sd = r.mean(0), r.std(0).clamp_min(1e-6)
        self.c = c
        self.tau = self.score(model, calib["obs"], calib["times"]).quantile(0.95)

    def score(self, model, obs, times):
        r = residual_onestep(model, self.c, obs, times)
        return ((r - self.mu) / self.sd).abs().mean(dim=1)


def run_seed(dataset, seed, epochs, d):
    data = get_splits(dataset, seed)
    keys, n_obs = data["keys"], data["n_obs"]
    unseen_keys = DATASETS[dataset]["unseen"]
    known = [c for c in range(len(keys)) if keys[c] not in unseen_keys]
    unseen = [c for c in range(len(keys)) if keys[c] in unseen_keys]

    t0 = time.time()
    torch.manual_seed(seed)
    model = SharedDynamicsClassifier(n_obs=n_obs, n_classes=len(known), d=d)
    per_class = {i: data["train"][c] for i, c in enumerate(known)}
    train_shared(model, per_class, epochs=epochs, seed=seed)
    cals = [ClassCal(model, i, data["calib"][c]) for i, c in enumerate(known)]
    print(f"  seed={seed}: shared model + {len(known)} fields trained "
          f"({time.time()-t0:.0f}s)", flush=True)

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

    n_gen = n_disc = total = 0
    confusion = np.zeros((len(known), len(known)), dtype=int)
    for i, c in enumerate(known):
        obs, times = data["test"][c]["obs"], data["test"][c]["times"]
        scores = torch.stack([cal.score(model, obs, times) for cal in cals], dim=1)
        pred = scores.argmin(dim=1)
        for j in range(len(known)):
            confusion[i, j] += (pred == j).sum().item()
        n_gen += (pred == i).sum().item()
        with torch.no_grad():
            n_disc += (clf(obs, times).argmax(-1) == i).sum().item()
        total += len(obs)

    obs = torch.cat([data["test"][c]["obs"] for c in unseen])
    times = torch.cat([data["test"][c]["times"] for c in unseen])
    scores = torch.stack([cal.score(model, obs, times) for cal in cals], dim=1)
    taus = torch.stack([cal.tau for cal in cals])
    gen_reject = (scores > taus).all(dim=1).float().mean().item()
    with torch.no_grad():
        probs = F.softmax(clf(obs, times), -1).max(-1).values
    disc_reject = (probs < clf_tau).float().mean().item()

    return {"acc_gen": n_gen / total, "acc_disc": n_disc / total,
            "openset_reject_gen": gen_reject, "openset_reject_disc": disc_reject,
            "confusion": confusion.tolist(),
            "known": [keys[c] for c in known]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="har", choices=list(DATASETS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--d", type=int, default=12)
    args = p.parse_args()

    results = {"args": vars(args), "runs": []}
    for seed in args.seeds:
        r = run_seed(args.dataset, seed, args.epochs, args.d)
        results["runs"].append({"seed": seed, **r})
        print(f"  seed={seed} acc gen {r['acc_gen']:.3f} | disc {r['acc_disc']:.3f} | "
              f"openset reject gen {r['openset_reject_gen']:.0%} vs disc "
              f"{r['openset_reject_disc']:.0%}", flush=True)

    (ROOT / "results" / f"phase11_shared_{args.dataset}.json").write_text(
        json.dumps(results, indent=2))
    print("\n== mean over seeds ==")
    for k in ("acc_gen", "acc_disc", "openset_reject_gen", "openset_reject_disc"):
        print(f"  {k:22s} {np.mean([r[k] for r in results['runs']]):.3f}")
    known = results["runs"][0]["known"]
    conf = np.sum([r["confusion"] for r in results["runs"]], axis=0)
    print("\nconfusion (gen; rows=true, cols=pred):")
    print("  " + " ".join(f"{c:>10s}" for c in known))
    for i, c in enumerate(known):
        print(f"  {c:>10s} " + " ".join(f"{v:10d}" for v in conf[i]))


if __name__ == "__main__":
    main()
