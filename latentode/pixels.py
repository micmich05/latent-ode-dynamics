"""Phase 4: pixel observations (rendered pendulum) for both models.

Observation = stacked pair of consecutive 32x32 grayscale frames (a single
frame does not contain velocity). Regular sampling: H2 is orthogonal to
irregularity, which phase 1 already settled.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import generate_states
from .model import LatentODEJEPA, VectorField, integrate, mlp

SIZE = 32


def render_frames(x_state, theta_scale=0.55, R=14.0, pivot=(16.0, 8.0), sigma=1.5):
    """Render pendulum frames from the oscillator position channel.

    x_state: [n, T] -> frames [n, T, SIZE, SIZE] float32 in [0, 1].
    """
    theta = theta_scale * x_state
    px = pivot[0] + R * np.sin(theta)
    py = pivot[1] + R * np.cos(theta)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dx = xs[None, None] - px[..., None, None]
    dy = ys[None, None] - py[..., None, None]
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2)).astype(np.float32)


def make_pixel_dataset(n_train=128, n_test=48, T=40, dt_base=0.1, noise_std=0.05,
                       diff_channel=True, seed=0):
    """Damped-oscillator pendulum video. obs[t] = (frame_t, frame_t - frame_{t-1})
    (or the raw pair if diff_channel=False), so sequences have T-1 steps;
    states/times align with the CURRENT frame. Per-frame iid noise (each capture
    noisy once, shared across the two pairs that contain it).

    The diff channel is a fixed linear reparameterization of the raw pair (the
    first conv layer could compute it) that makes velocity visible as a signed
    dipole; without it the JEPA encoder learns position but fails to find
    velocity (R^2 ~ 0.26). Both models receive the same representation.
    """
    out = {}
    for split, n, sd in [("train", n_train, seed), ("test", n_test, seed + 1000)]:
        states, times = generate_states("oscillator", n, T, dt_base, jitter=0.0, seed=sd)
        frames = render_frames(states[..., 0])
        g = np.random.default_rng(sd + 7)
        noisy = frames + noise_std * g.standard_normal(frames.shape).astype(np.float32)

        def pairs(f):
            if diff_channel:
                return torch.from_numpy(np.stack([f[:, 1:], f[:, 1:] - f[:, :-1]], axis=2))
            return torch.from_numpy(np.stack([f[:, :-1], f[:, 1:]], axis=2))

        out[split] = {
            "obs": pairs(noisy), "obs_clean": pairs(frames),
            "times": torch.from_numpy(times[:, 1:]),
            "states": torch.from_numpy(states[:, 1:]),
        }
    noise_share = (((out["test"]["obs"] - out["test"]["obs_clean"]).var()
                    / out["test"]["obs"].var()).item())
    out["meta"] = {"T": T, "dt_base": dt_base, "noise_std": noise_std,
                   "noise_share": noise_share}
    return out


class CNNEncoder(nn.Module):
    """2-channel 32x32 -> out_dim, handling arbitrary leading batch dims."""

    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.SiLU(),   # 16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),  # 8x8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),  # 4x4
        )
        # Projector with normalization (standard in joint-embedding training):
        # without it the variance term never saturates and fights the
        # prediction loss to a bad equilibrium.
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        lead = x.shape[:-3]
        h = self.conv(x.reshape(-1, *x.shape[-3:]))
        return self.head(h.flatten(1)).reshape(*lead, -1)


class PixelDecoder(nn.Module):
    """d -> 2-channel 32x32 frame pair, arbitrary leading dims."""

    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.SiLU(),  # 8x8
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU(),  # 16x16
            nn.ConvTranspose2d(16, 2, 4, stride=2, padding=1),              # 32x32
        )

    def forward(self, z):
        lead = z.shape[:-1]
        h = self.fc(z.reshape(-1, z.shape[-1])).reshape(-1, 32, 4, 4)
        return self.deconv(h).reshape(*lead, 2, SIZE, SIZE)


class PixelJEPA(LatentODEJEPA):
    """Same JEPA machinery, CNN encoder instead of MLP."""

    def __init__(self, d=8, field_hidden=128, n_substeps=2, method="rk4", ema_decay=None):
        super().__init__(n_obs=1, d=d, field_hidden=field_hidden,
                         n_substeps=n_substeps, method=method, ema_decay=None)
        self.encoder = CNNEncoder(d)
        self.ema_decay = ema_decay
        if ema_decay is not None:
            import copy
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)


class PixelLatentODE(nn.Module):
    """Decoder-based baseline on pixels: CNN features -> GRU context encoder ->
    same vector field -> deconv decoder, pixel reconstruction loss."""

    def __init__(self, d=8, feat=64, hidden=128, n_substeps=2, method="rk4"):
        super().__init__()
        self.cnn = CNNEncoder(feat)
        self.gru = nn.GRU(feat + 1, hidden, batch_first=True)
        self.enc_out = nn.Linear(hidden, d)
        self.field = VectorField(d, hidden=hidden)
        self.decoder = PixelDecoder(d)
        self.n_substeps = n_substeps
        self.method = method

    def encode_context(self, obs_ctx, times_ctx):
        feats = self.cnn(obs_ctx)
        dts = times_ctx[:, 1:] - times_ctx[:, :-1]
        dts = torch.cat([torch.zeros(obs_ctx.shape[0], 1, device=dts.device), dts], dim=1)
        h, _ = self.gru(torch.cat([feats, dts[..., None]], dim=-1))
        return self.enc_out(h[:, -1])

    def rollout(self, z0, dts):
        preds, z = [], z0
        for i in range(dts.shape[1]):
            z = integrate(self.field, z, dts[:, i], self.n_substeps, self.method)
            preds.append(z)
        return torch.stack(preds, dim=1)

    def loss(self, obs, times, context):
        z_c = self.encode_context(obs[:, :context], times[:, :context])
        dts = times[:, context:] - times[:, context - 1:-1]
        z_roll = self.rollout(z_c, dts)
        return (F.mse_loss(self.decoder(z_roll), obs[:, context:])
                + F.mse_loss(self.decoder(z_c), obs[:, context - 1]))


def train_pixel_model(model, data, device, kind, context=10, epochs=200,
                      batch_size=64, lr=1e-3, rollout_horizon=8,
                      lambda_var=1.0, lambda_cov=0.1, seed=0, log_every=50, verbose=True):
    """One loop for both pixel models. kind: 'jepa' | 'lode'.
    Data stays on CPU; batches move to `device`."""
    from .losses import jepa_losses

    model.to(device)
    obs, times = data["train"]["obs"], data["train"]["times"]
    n_traj = obs.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        perm = torch.randperm(n_traj, generator=g)
        last = {}
        for i in range(0, n_traj, batch_size):
            idx = perm[i:i + batch_size]
            b_obs, b_times = obs[idx].to(device), times[idx].to(device)
            if kind == "jepa":
                losses = jepa_losses(model, b_obs, b_times, rollout_horizon)
                total = (losses["pred"] + losses["roll"]
                         + lambda_var * losses["var"] + lambda_cov * losses["cov"])
                last = {k: v.item() for k, v in losses.items() if k != "z"}
            else:
                total = model.loss(b_obs, b_times, context)
                last = {"recon": total.item()}
            opt.zero_grad()
            total.backward()
            opt.step()
            model.update_ema() if hasattr(model, "update_ema") and kind == "jepa" else None
        sched.step()
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"  [{kind}] epoch {epoch+1}: " +
                  " ".join(f"{k} {v:.4f}" for k, v in last.items()), flush=True)
    return model
