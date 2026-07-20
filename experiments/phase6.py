"""Phase 6: anomaly detection / classification by dynamics consistency.

Train on NORMAL trajectories only (irregular sampling, jitter=0.5). Test series
carry one of three mid-series anomalies (onset uniform in steps 20-40):
  param:   regime change, omega 2.0 -> 2.5 from onset
  impulse: velocity kick (+2.0) at onset
  sensor:  observation fault, dims 0-9 frozen at their onset value

Score = per-step prediction residual, z-scored per horizon step against a
normal calibration split; series score = max_t. Each model scores the way it
was trained: ours -> one-step latent residual, GRU -> one-step obs residual,
latent ODE + decoder -> rollout-from-context obs residual.

Metrics: AUROC (normal vs anomalous) per type, detection rate and median
delay at 5% series-level FPR. 3 seeds.

Usage: python3 experiments/phase6.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latentode import LatentODEJEPA, make_dataset, train
from latentode.baselines import GRUForecaster, LatentODEBaseline, train_obs_space_model
from latentode.data import FINE_SUBSTEPS, RandomLift

CONTEXT = 10
T = 60
DT = 0.1
JITTER = 0.5
NOISE = 0.02


def gen_anomalous(n, seed, anomaly=None):
    """Integrate the oscillator with an optional mid-series anomaly.
    Returns states [n,T,2], times [n,T], onset step [n] (or None)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-2.0, 2.0, size=(n, 2))
    dts = DT * rng.uniform(1 - JITTER, 1 + JITTER, size=(n, T - 1))
    onset = rng.integers(20, 40, size=n) if anomaly else None

    states = [s]
    for i in range(T - 1):
        omega = np.full(n, 2.0)
        if anomaly == "param":
            omega = np.where(i >= onset, 2.5, 2.0)
        if anomaly == "impulse":
            kick = (i == onset)
            s = s.copy()
            s[kick, 1] += 2.0
        h = dts[:, i] / FINE_SUBSTEPS
        for _ in range(FINE_SUBSTEPS):
            x, v = s[..., 0], s[..., 1]
            def f(st):
                return np.stack([st[..., 1],
                                 -(omega**2) * st[..., 0] - 0.15 * st[..., 1]], axis=-1)
            k1 = f(s); k2 = f(s + 0.5 * h[:, None] * k1)
            k3 = f(s + 0.5 * h[:, None] * k2); k4 = f(s + h[:, None] * k3)
            s = s + h[:, None] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(s)
    states = np.stack(states, axis=1).astype(np.float32)
    times = np.concatenate([np.zeros((n, 1)), np.cumsum(dts, axis=1)], axis=1).astype(np.float32)
    return states, times, onset


def build_split(n, seed, norm, anomaly=None):
    states, times, onset = gen_anomalous(n, seed, anomaly)
    lift = RandomLift(n_obs=50)
    obs = lift(torch.from_numpy(states))
    obs = obs + NOISE * torch.randn(obs.shape, generator=torch.Generator().manual_seed(seed))
    if anomaly == "sensor":
        for j in range(n):
            obs[j, onset[j]:, :10] = obs[j, onset[j], :10]
    obs = (obs - norm["mean"]) / norm["std"]
    return {"obs": obs, "times": torch.from_numpy(times), "onset": onset}


@torch.no_grad()
def score_ours(model, split):
    z = model.encode(split["obs"])
    dts = split["times"][:, 1:] - split["times"][:, :-1]
    B, Tn, d = z.shape
    pred = model.step(z[:, :-1].reshape(-1, d), dts.reshape(-1)).reshape(B, Tn - 1, d)
    return (pred - z[:, 1:]).norm(dim=-1), 1  # residual i predicts obs index i+1


@torch.no_grad()
def score_ours_rollout(model, split):
    """Rollout-mode latent score: forecast from the (normal) context and
    accumulate deviation — stronger for persistent regime changes."""
    obs, times = split["obs"], split["times"]
    z = model.encode(obs)
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    z_roll = model.rollout(z[:, CONTEXT - 1], dts)
    return (z_roll - z[:, CONTEXT:]).norm(dim=-1), CONTEXT


@torch.no_grad()
def score_gru(model, split):
    preds = model._teacher_forced(split["obs"], split["times"])
    return (preds - split["obs"][:, 1:]).norm(dim=-1), 1


@torch.no_grad()
def score_lode(model, split):
    obs, times = split["obs"], split["times"]
    z_c = model.encode_context(obs[:, :CONTEXT], times[:, :CONTEXT])
    dts = times[:, CONTEXT:] - times[:, CONTEXT - 1:-1]
    preds = model.decoder(model.rollout(z_c, dts))
    return (preds - obs[:, CONTEXT:]).norm(dim=-1), CONTEXT


def zscore(resid, mu, sd):
    return (resid - mu) / sd


def auroc(pos, neg):
    scores = torch.cat([pos, neg])
    ranks = torch.empty(len(scores))
    ranks[scores.argsort()] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    r_pos = ranks[:len(pos)].sum()
    return ((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def evaluate(streams, calib, test_normal, anom_splits):
    """streams: list of (model, score_fn); their z-scores are combined by
    elementwise max (a series is anomalous if ANY stream flags it)."""
    stats, offs = [], []
    for model, fn in streams:
        r, off = fn(model, calib)
        stats.append((r.mean(0), r.std(0).clamp_min(1e-6)))
        offs.append(off)
    assert len(set(offs)) == 1, "combined streams must share the horizon offset"
    off = offs[0]

    def zmax(split):
        zs = [zscore(fn(model, split)[0], mu, sd)
              for (model, fn), (mu, sd) in zip(streams, stats)]
        return torch.stack(zs).max(0).values

    z_norm = zmax(test_normal)
    tau = zmax(calib).max(dim=1).values.quantile(0.95)

    out = {}
    for name, split in anom_splits.items():
        z_a = zmax(split)
        out[name] = {"auroc": auroc(z_a.max(1).values, z_norm.max(1).values)}
        delays, detected = [], 0
        for j in range(len(z_a)):
            hits = (z_a[j] > tau).nonzero()
            if len(hits):
                detected += 1
                delays.append(hits[0].item() + off - split["onset"][j])
        out[name]["detect_rate"] = detected / len(z_a)
        out[name]["median_delay"] = float(np.median(delays)) if delays else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    results = {"args": vars(args), "runs": []}
    for seed in args.seeds:
        data = make_dataset(system="oscillator", jitter=JITTER, seed=seed)
        norm = data["norm"]
        calib = build_split(64, seed + 100, norm)
        test_normal = build_split(64, seed + 200, norm)
        anom = {a: build_split(64, seed + 300, norm, anomaly=a)
                for a in ("param", "impulse", "sensor")}

        torch.manual_seed(seed)
        t0 = time.time()
        ours = LatentODEJEPA(n_obs=50, d=8)
        train(ours, data, epochs=args.epochs, seed=seed, verbose=False)
        from latentode.eval import train_decoder_probe
        probe = train_decoder_probe(ours, data, seed=seed)

        @torch.no_grad()
        def score_ours_probe(model, split):
            """Obs-space one-step residual through the post-hoc probe decoder:
            covers observation-level faults the latent stream is blind to."""
            z = model.encode(split["obs"])
            dts = split["times"][:, 1:] - split["times"][:, :-1]
            B, Tn, d = z.shape
            pred = model.step(z[:, :-1].reshape(-1, d), dts.reshape(-1)).reshape(B, Tn - 1, d)
            return (probe(pred) - split["obs"][:, 1:]).norm(dim=-1), 1

        torch.manual_seed(seed)
        gru = GRUForecaster(n_obs=50)
        train_obs_space_model(gru, data, epochs=args.epochs, seed=seed)
        torch.manual_seed(seed)
        lode = LatentODEBaseline(n_obs=50, d=8)
        train_obs_space_model(lode, data, epochs=args.epochs, seed=seed)

        models = {
            "ours": [(ours, score_ours)],
            "ours-roll": [(ours, score_ours_rollout)],
            "ours+probe": [(ours, score_ours), (ours, score_ours_probe)],
            "gru": [(gru, score_gru)],
            "lode": [(lode, score_lode)],
        }
        for name, streams in models.items():
            ev = evaluate(streams, calib, test_normal, anom)
            results["runs"].append({"seed": seed, "model": name, "metrics": ev})
            summary = " | ".join(f"{a}: AUROC {m['auroc']:.3f} det {m['detect_rate']:.0%} "
                                 f"delay {m['median_delay']}" for a, m in ev.items())
            print(f"seed={seed} {name:5s} {summary}", flush=True)
        print(f"  (seed {seed} done, {time.time()-t0:.0f}s)", flush=True)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    (results_dir / "phase6_anomaly.json").write_text(json.dumps(results, indent=2))

    print("\n== AUROC mean over seeds ==")
    print(f"{'model':>10s} | {'param':>7s} | {'impulse':>7s} | {'sensor':>7s}")
    for name in ("ours", "ours-roll", "ours+probe", "gru", "lode"):
        rs = [r["metrics"] for r in results["runs"] if r["model"] == name]
        row = [sum(r[a]["auroc"] for r in rs) / len(rs) for a in ("param", "impulse", "sensor")]
        print(f"{name:>10s} | " + " | ".join(f"{v:7.3f}" for v in row))


if __name__ == "__main__":
    main()
