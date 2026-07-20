"""CharacterTrajectories (UCI): real irregularly-subsampled benchmark.

2858 pen-tip trajectories (vx, vy, force @ 200Hz), 20 character classes.
Pipeline: subsample K random time points per sequence (irregular dt by
construction, mirroring the Neural CDE dropped-data setting), then Takens
delay-embed m consecutive OBSERVED samples plus their gaps — a character is
not autonomous in raw obs space (the same instantaneous velocity occurs at
several points of a letter); the delay embedding reconstructs a state the
per-frame encoder can work with.
"""

import numpy as np
import torch
from scipy.io import loadmat

DT_NATIVE = 0.005  # 200 Hz


def load_chartraj(path="data/mixoutALL_shifted.mat"):
    m = loadmat(path)
    seqs = [s.astype(np.float32) for s in m["mixout"][0]]          # each [3, L]
    labels = m["consts"]["charlabels"][0, 0][0].astype(int) - 1     # 0..19
    keys = [str(k[0]) for k in m["consts"]["key"][0, 0][0]]
    return seqs, labels, keys


def build_series(seqs, K=40, m=4, seed=0):
    """Subsample + delay-embed. Returns obs [N, K-m+1, 3m+(m-1)], times [N, K-m+1]."""
    rng = np.random.default_rng(seed)
    all_obs, all_times = [], []
    for s in seqs:
        L = s.shape[1]
        idx = np.sort(rng.choice(L, size=K, replace=False))
        t = idx * DT_NATIVE
        x = s[:, idx].T                                   # [K, 3]
        rows = []
        for j in range(m - 1, K):
            lags = [x[j - i] for i in range(m)]           # current + m-1 past samples
            gaps = [t[j - i] - t[j - i - 1] for i in range(m - 1)]
            rows.append(np.concatenate(lags + [np.array(gaps, dtype=np.float32)]))
        all_obs.append(np.stack(rows))
        all_times.append(t[m - 1:])
    return torch.from_numpy(np.stack(all_obs)), torch.from_numpy(np.stack(all_times).astype(np.float32))


def make_splits(K=40, m=4, seed=0, path="data/mixoutALL_shifted.mat"):
    """Per-class 60/20/20 train/calib/test splits, globally standardized obs."""
    seqs, labels, keys = load_chartraj(path)
    obs, times = build_series(seqs, K=K, m=m, seed=seed)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "calib": [], "test": []}
    for c in range(len(keys)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        splits["train"].append((c, idx[:int(0.6 * n)]))
        splits["calib"].append((c, idx[int(0.6 * n):int(0.8 * n)]))
        splits["test"].append((c, idx[int(0.8 * n):]))

    train_idx = np.concatenate([i for _, i in splits["train"]])
    mean = obs[train_idx].reshape(-1, obs.shape[-1]).mean(0)
    std = obs[train_idx].reshape(-1, obs.shape[-1]).std(0).clamp_min(1e-6)
    obs = (obs - mean) / std

    out = {"keys": keys, "n_obs": obs.shape[-1]}
    for split, pairs in splits.items():
        out[split] = {c: {"obs": obs[i], "times": times[i]} for c, i in pairs}
    return out
