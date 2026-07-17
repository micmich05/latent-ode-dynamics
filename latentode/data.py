"""Synthetic dynamical systems, lifted to high-dimensional observations.

Ground-truth state is 2D; observations are a fixed random MLP lift to R^n
plus Gaussian noise. The model never sees the true state — it is kept only
for evaluation (probes, phase portraits).
"""

import numpy as np
import torch
import torch.nn as nn

FINE_SUBSTEPS = 20  # RK4 substeps per sampling interval for ground-truth integration


def damped_oscillator(s, omega=2.0, gamma=0.15):
    x, v = s[..., 0], s[..., 1]
    return np.stack([v, -(omega**2) * x - gamma * v], axis=-1)


def lotka_volterra(s, alpha=1.5, beta=1.0, delta=1.0, gamma=3.0):
    x, y = s[..., 0], s[..., 1]
    return np.stack([alpha * x - beta * x * y, delta * x * y - gamma * y], axis=-1)


SYSTEMS = {
    "oscillator": {
        "f": damped_oscillator,
        "init": lambda rng, n: rng.uniform(-2.0, 2.0, size=(n, 2)),
    },
    "lotka_volterra": {
        "f": lotka_volterra,
        "init": lambda rng, n: rng.uniform([1.0, 0.5], [5.0, 2.5], size=(n, 2)),
    },
}


def _rk4_step(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5 * dt[..., None] * k1)
    k3 = f(s + 0.5 * dt[..., None] * k2)
    k4 = f(s + dt[..., None] * k3)
    return s + dt[..., None] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_states(system, n_traj, T, dt_base, jitter=0.0, seed=0):
    """Integrate the true system. Returns states [n_traj, T, 2], times [n_traj, T].

    jitter s in [0, 1) controls sampling irregularity: dt ~ dt_base * U(1-s, 1+s).
    Mean dt stays dt_base, so total horizon is comparable across s.
    """
    rng = np.random.default_rng(seed)
    spec = SYSTEMS[system]
    s = spec["init"](rng, n_traj)
    if jitter > 0:
        dts = dt_base * rng.uniform(1 - jitter, 1 + jitter, size=(n_traj, T - 1))
    else:
        dts = np.full((n_traj, T - 1), dt_base)

    states = [s]
    for i in range(T - 1):
        h = dts[:, i] / FINE_SUBSTEPS
        for _ in range(FINE_SUBSTEPS):
            s = _rk4_step(spec["f"], s, h)
        states.append(s)
    states = np.stack(states, axis=1)
    times = np.concatenate([np.zeros((n_traj, 1)), np.cumsum(dts, axis=1)], axis=1)
    return states.astype(np.float32), times.astype(np.float32)


class RandomLift:
    """Fixed random MLP R^2 -> R^n_obs. Same weights for train/test (fixed seed)."""

    def __init__(self, n_obs=50, hidden=64, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.W1 = torch.randn(2, hidden, generator=g) * (1.0 / np.sqrt(2))
        self.b1 = torch.randn(hidden, generator=g) * 0.1
        self.W2 = torch.randn(hidden, n_obs, generator=g) * (1.0 / np.sqrt(hidden))
        self.b2 = torch.randn(n_obs, generator=g) * 0.1

    def __call__(self, states):
        h = torch.tanh(states @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def make_dataset(system="oscillator", n_train=256, n_test=64, T=60, dt_base=0.1,
                 n_obs=50, noise_std=0.02, jitter=0.0, seed=0):
    """Returns dict with obs/times/states tensors for train and test, plus norm stats."""
    lift = RandomLift(n_obs=n_obs)
    out = {}
    for split, n, sd in [("train", n_train, seed), ("test", n_test, seed + 1000)]:
        states, times = generate_states(system, n, T, dt_base, jitter=jitter, seed=sd)
        states_t = torch.from_numpy(states)
        clean = lift(states_t)
        obs = clean + noise_std * torch.randn(clean.shape, generator=torch.Generator().manual_seed(sd))
        out[split] = {"obs": obs, "obs_clean": clean,
                      "times": torch.from_numpy(times), "states": states_t}

    # Normalize with noisy-train stats; clean obs share the same coordinates so
    # "RMSE vs clean" is measured in the same space the models operate in.
    mean = out["train"]["obs"].reshape(-1, n_obs).mean(0)
    std = out["train"]["obs"].reshape(-1, n_obs).std(0).clamp_min(1e-6)
    for split in ("train", "test"):
        out[split]["obs"] = (out[split]["obs"] - mean) / std
        out[split]["obs_clean"] = (out[split]["obs_clean"] - mean) / std
    out["norm"] = {"mean": mean, "std": std}
    out["meta"] = {"system": system, "T": T, "dt_base": dt_base, "n_obs": n_obs,
                   "noise_std": noise_std, "jitter": jitter}
    return out
