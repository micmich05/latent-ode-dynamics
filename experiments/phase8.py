"""Phase 8: multi-class classification by dynamics consistency + open-set.

One UnifiedLatentODE per dynamical regime, trained only on that regime's
series. A test series is assigned to the class whose field explains it best
(lowest mean z-scored decoded-rollout residual). Series from an UNSEEN regime
(Van der Pol) should be rejected by all fields — the open-set capability a
discriminative classifier lacks structurally.

Baseline: supervised GRU classifier (labels, cross-entropy, max-softmax
rejection). Both rejection thresholds calibrated to 95% in-distribution
acceptance. 3 seeds.

Usage: python3 experiments/phase8.py
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
from latentode import make_dataset
from latentode.data import SYSTEMS, RandomLift, damped_oscillator, generate_states
from latentode.model import mlp
from latentode.unified import UnifiedLatentODE, train_unified

CONTEXT = 10
JITTER = 0.5
NOISE = 0.02
N_OBS = 50

# Register the class-regime variants (van_der_pol ships in data.SYSTEMS).
SYSTEMS["osc_fast"] = {"f": lambda s: damped_oscillator(s, omega=2.8),
                       "init": SYSTEMS["oscillator"]["init"]}
SYSTEMS["osc_damped"] = {"f": lambda s: damped_oscillator(s, gamma=0.7),
                         "init": SYSTEMS["oscillator"]["init"]}

CLASSES = ["oscillator", "osc_fast", "osc_damped", "lotka_volterra"]
UNSEEN = "van_der_pol"


def build_raw(system, n, seed):
    """Un-normalized observations (normalization is each class model's job)."""
    states, times = generate_states(system, n, T=60, dt_base=0.1, jitter=JITTER, seed=seed)
    obs = RandomLift(n_obs=N_OBS)(torch.from_numpy(states))
    obs = obs + NOISE * torch.randn(obs.shape, generator=torch.Generator().manual_seed(seed))
    return obs, torch.from_numpy(times)


@torch.no_grad()
def residual_series(model, norm, obs_raw, times):
    """Mean per-step decoded-rollout residual of each series under this model."""
    obs = (obs_raw - norm["mean"]) / norm["std"]
    z_c = model.encode(obs[:, CONTEXT - 1])
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    preds = model.decoder(model.rollout(z_c, dts))
    return (preds - obs[:, CONTEXT:]).norm(dim=-1)  # [B, T-CONTEXT]


class ClassField:
    """A trained per-class model plus its calibration statistics."""

    def __init__(self, system, seed, epochs):
        data = make_dataset(system=system, jitter=JITTER, seed=seed)
        torch.manual_seed(seed)
        self.model = UnifiedLatentODE(n_obs=N_OBS, d=8)
        train_unified(self.model, data, epochs=epochs, seed=seed, verbose=False)
        self.norm = data["norm"]
        cal_obs, cal_times = build_raw(system, 64, seed + 100)
        r = residual_series(self.model, self.norm, cal_obs, cal_times)
        self.mu, self.sd = r.mean(0), r.std(0).clamp_min(1e-6)
        self.tau = self.score(cal_obs, cal_times).quantile(0.95)

    def score(self, obs_raw, times):
        r = residual_series(self.model, self.norm, obs_raw, times)
        return ((r - self.mu) / self.sd).mean(dim=1)  # mean z over the horizon


class GRUClassifier(nn.Module):
    def __init__(self, n_classes, hidden=128):
        super().__init__()
        self.gru = nn.GRU(N_OBS + 1, hidden, batch_first=True)
        self.head = mlp([hidden, hidden, n_classes])

    def forward(self, obs, times):
        dts = times[:, 1:] - times[:, :-1]
        dts = torch.cat([torch.zeros(obs.shape[0], 1), dts], dim=1)
        h, _ = self.gru(torch.cat([obs, dts[..., None]], dim=-1))
        return self.head(h[:, -1])


def run_seed(seed, epochs):
    t0 = time.time()
    fields = {c: ClassField(c, seed, epochs) for c in CLASSES}
    print(f"  seed={seed}: 4 class fields trained ({time.time()-t0:.0f}s)", flush=True)

    # Supervised baseline on pooled labeled data (128 series/class + pooled norm).
    tr_obs, tr_times, tr_y = [], [], []
    for i, c in enumerate(CLASSES):
        o, t = build_raw(c, 128, seed + 10)
        tr_obs.append(o); tr_times.append(t); tr_y.append(torch.full((len(o),), i))
    tr_obs, tr_times, tr_y = torch.cat(tr_obs), torch.cat(tr_times), torch.cat(tr_y)
    mean, std = tr_obs.reshape(-1, N_OBS).mean(0), tr_obs.reshape(-1, N_OBS).std(0).clamp_min(1e-6)

    torch.manual_seed(seed)
    clf = GRUClassifier(len(CLASSES))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for epoch in range(60):
        perm = torch.randperm(len(tr_obs))
        for i in range(0, len(tr_obs), 128):
            idx = perm[i:i + 128]
            logits = clf((tr_obs[idx] - mean) / std, tr_times[idx])
            loss = F.cross_entropy(logits, tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # Softmax rejection threshold at 95% in-distribution acceptance.
    with torch.no_grad():
        val_probs = []
        for c in CLASSES:
            o, t = build_raw(c, 64, seed + 20)
            val_probs.append(F.softmax(clf((o - mean) / std, t), -1).max(-1).values)
        clf_tau = torch.cat(val_probs).quantile(0.05)

    # Closed-set accuracy + confusion.
    n_correct = {"gen": 0, "disc": 0}
    total = 0
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for i, c in enumerate(CLASSES):
        o, t = build_raw(c, 64, seed + 30)
        scores = torch.stack([fields[k].score(o, t) for k in CLASSES], dim=1)
        pred = scores.argmin(dim=1)
        for j in range(len(CLASSES)):
            confusion[i, j] += (pred == j).sum().item()
        n_correct["gen"] += (pred == i).sum().item()
        with torch.no_grad():
            pred_d = clf((o - mean) / std, t).argmax(-1)
        n_correct["disc"] += (pred_d == i).sum().item()
        total += len(o)

    # Open-set: unseen regime should be rejected.
    o, t = build_raw(UNSEEN, 64, seed + 40)
    scores = torch.stack([fields[k].score(o, t) for k in CLASSES], dim=1)
    taus = torch.stack([fields[k].tau for k in CLASSES])
    gen_reject = (scores > taus).all(dim=1).float().mean().item()
    with torch.no_grad():
        probs = F.softmax(clf((o - mean) / std, t), -1).max(-1).values
    disc_reject = (probs < clf_tau).float().mean().item()

    return {"acc_gen": n_correct["gen"] / total, "acc_disc": n_correct["disc"] / total,
            "confusion": confusion.tolist(),
            "openset_reject_gen": gen_reject, "openset_reject_disc": disc_reject}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    results = {"args": vars(args), "classes": CLASSES, "unseen": UNSEEN, "runs": []}
    for seed in args.seeds:
        r = run_seed(seed, args.epochs)
        results["runs"].append({"seed": seed, **r})
        print(f"  seed={seed} acc gen {r['acc_gen']:.3f} | disc {r['acc_disc']:.3f} | "
              f"openset reject gen {r['openset_reject_gen']:.0%} vs disc "
              f"{r['openset_reject_disc']:.0%}", flush=True)

    (ROOT / "results" / "phase8_classifier.json").write_text(json.dumps(results, indent=2))
    print("\n== mean over seeds ==")
    for k in ("acc_gen", "acc_disc", "openset_reject_gen", "openset_reject_disc"):
        print(f"  {k:22s} {np.mean([r[k] for r in results['runs']]):.3f}")
    print("\nconfusion (gen, summed over seeds; rows=true, cols=pred):")
    conf = np.sum([r["confusion"] for r in results["runs"]], axis=0)
    print("  " + " ".join(f"{c:>12s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>12s} " + " ".join(f"{v:12d}" for v in conf[i]))


if __name__ == "__main__":
    main()
