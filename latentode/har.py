"""UCI HAR: human activity recognition — real dynamical regimes.

Unlike character trajectories (same pen physics, different control programs),
activities ARE distinct dynamics laws of the body: walking, climbing stairs
and lying down differ in the vector field itself, which is the scope condition
for dynamics-consistency classification.

9 inertial channels (body acc, gyro, total acc) @ 50Hz, windows of 128
samples. Same treatment as chartraj: irregular subsampling + short Takens
delay embedding.
"""

from pathlib import Path

import numpy as np
import torch

CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"]
ACTIVITIES = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]
DT_NATIVE = 0.02  # 50 Hz


def load_har(root="data/UCI HAR Dataset", cache="data/har_cache.npz"):
    if Path(cache).exists():
        z = np.load(cache)
        return (z["Xtr"], z["ytr"]), (z["Xte"], z["yte"])
    out = {}
    for split in ("train", "test"):
        X = np.stack([np.loadtxt(f"{root}/{split}/Inertial Signals/{ch}_{split}.txt")
                      for ch in CHANNELS], axis=-1).astype(np.float32)  # [N, 128, 9]
        y = np.loadtxt(f"{root}/{split}/y_{split}.txt").astype(int) - 1
        out[split] = (X, y)
    np.savez_compressed(cache, Xtr=out["train"][0], ytr=out["train"][1],
                        Xte=out["test"][0], yte=out["test"][1])
    return out["train"], out["test"]


def build_series(X, K=40, m=3, seed=0):
    """Irregular subsample + delay embed: [N, K-m+1, 9m+(m-1)], times [N, K-m+1]."""
    rng = np.random.default_rng(seed)
    N, L, C = X.shape
    obs, times = [], []
    for i in range(N):
        idx = np.sort(rng.choice(L, size=K, replace=False))
        t = idx * DT_NATIVE
        x = X[i, idx]                                     # [K, 9]
        rows = []
        for j in range(m - 1, K):
            lags = [x[j - k] for k in range(m)]
            gaps = [t[j - k] - t[j - k - 1] for k in range(m - 1)]
            rows.append(np.concatenate(lags + [np.array(gaps, dtype=np.float32)]))
        obs.append(np.stack(rows))
        times.append(t[m - 1:])
    return (torch.from_numpy(np.stack(obs).astype(np.float32)),
            torch.from_numpy(np.stack(times).astype(np.float32)))


def make_splits(n_field_train=256, n_calib=128, K=40, m=3, seed=0):
    (Xtr, ytr), (Xte, yte) = load_har()
    obs_tr, times_tr = build_series(Xtr, K=K, m=m, seed=seed)
    obs_te, times_te = build_series(Xte, K=K, m=m, seed=seed + 1)

    rng = np.random.default_rng(seed)
    out = {"keys": ACTIVITIES, "n_obs": obs_tr.shape[-1],
           "train": {}, "calib": {}, "test": {}}
    train_pool = []
    for c in range(len(ACTIVITIES)):
        idx = np.where(ytr == c)[0]
        rng.shuffle(idx)
        out["train"][c] = {"obs": obs_tr[idx[:n_field_train]],
                           "times": times_tr[idx[:n_field_train]]}
        out["calib"][c] = {"obs": obs_tr[idx[n_field_train:n_field_train + n_calib]],
                           "times": times_tr[idx[n_field_train:n_field_train + n_calib]]}
        train_pool.append(idx[:n_field_train])
        te = np.where(yte == c)[0]
        out["test"][c] = {"obs": obs_te[te], "times": times_te[te]}

    pool = np.concatenate(train_pool)
    mean = obs_tr[pool].reshape(-1, obs_tr.shape[-1]).mean(0)
    std = obs_tr[pool].reshape(-1, obs_tr.shape[-1]).std(0).clamp_min(1e-6)
    for split in ("train", "calib", "test"):
        for c in out[split]:
            out[split][c]["obs"] = (out[split][c]["obs"] - mean) / std
    return out
