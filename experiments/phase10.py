"""Phase 10: dynamics-consistency classification on UCI HAR + open-set.

Activities are genuine dynamical regimes (the scope condition phase 9
identified). One UnifiedLatentODE per activity, one-step decoded residual
scoring; 'downstairs' held out of all training as the unseen regime.
Baseline: supervised GRU on the same data, max-softmax rejection, both
thresholds at 95% in-distribution acceptance.

Usage: python3 experiments/phase10.py [--seeds 0]
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
from latentode.har import ACTIVITIES, make_splits
from phase9 import ClassField, GRUClassifier

UNSEEN = ["downstairs"]


def run_seed(seed, epochs, d):
    data = make_splits(seed=seed)
    n_obs = data["n_obs"]
    known = [c for c in range(len(ACTIVITIES)) if ACTIVITIES[c] not in UNSEEN]
    unseen = [c for c in range(len(ACTIVITIES)) if ACTIVITIES[c] in UNSEEN]

    t0 = time.time()
    fields = {c: ClassField(n_obs, data["train"][c], data["calib"][c], seed, epochs, d)
              for c in known}
    print(f"  seed={seed}: {len(known)} activity fields trained ({time.time()-t0:.0f}s)",
          flush=True)

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
        scores = torch.stack([fields[k].score(obs, times) for k in known], dim=1)
        pred = scores.argmin(dim=1)
        for j in range(len(known)):
            confusion[i, j] += (pred == j).sum().item()
        n_gen += (pred == i).sum().item()
        with torch.no_grad():
            n_disc += (clf(obs, times).argmax(-1) == i).sum().item()
        total += len(obs)

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
            "confusion": confusion.tolist(), "n_test": total, "n_unseen": len(obs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--d", type=int, default=12)
    args = p.parse_args()

    results = {"args": vars(args), "unseen": UNSEEN, "runs": []}
    for seed in args.seeds:
        r = run_seed(seed, args.epochs, args.d)
        results["runs"].append({"seed": seed, **r})
        print(f"  seed={seed} acc gen {r['acc_gen']:.3f} | disc {r['acc_disc']:.3f} | "
              f"openset reject gen {r['openset_reject_gen']:.0%} vs disc "
              f"{r['openset_reject_disc']:.0%}", flush=True)

    (ROOT / "results" / "phase10_har.json").write_text(json.dumps(results, indent=2))
    print("\n== mean over seeds ==")
    for k in ("acc_gen", "acc_disc", "openset_reject_gen", "openset_reject_disc"):
        print(f"  {k:22s} {np.mean([r[k] for r in results['runs']]):.3f}")
    known_names = [a for a in ACTIVITIES if a not in UNSEEN]
    conf = np.sum([r["confusion"] for r in results["runs"]], axis=0)
    print("\nconfusion (gen; rows=true, cols=pred):")
    print("  " + " ".join(f"{c:>11s}" for c in known_names))
    for i, c in enumerate(known_names):
        print(f"  {c:>11s} " + " ".join(f"{v:11d}" for v in conf[i]))


if __name__ == "__main__":
    main()
