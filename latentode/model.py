"""Encoder + latent vector field + ODE integration with variable dt."""

import copy

import torch
import torch.nn as nn


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class VectorField(nn.Module):
    """g_phi: R^d -> R^d (autonomous by default; time appended if autonomous=False)."""

    def __init__(self, d, hidden=128, autonomous=True):
        super().__init__()
        self.autonomous = autonomous
        in_dim = d if autonomous else d + 1
        self.net = mlp([in_dim, hidden, hidden, d])

    def forward(self, z, t=None):
        if self.autonomous:
            return self.net(z)
        return self.net(torch.cat([z, t[..., None].expand(*z.shape[:-1], 1)], dim=-1))


def integrate(field, z0, dt, n_substeps=2, method="rk4"):
    """Integrate dz/ds = g(z) from z0 over interval dt (dt: scalar or [B] or [B,1]).

    Variable dt per sample is the whole point: this is what lets the same
    field handle irregular sampling.
    """
    if torch.is_tensor(dt) and dt.dim() == 1:
        dt = dt[:, None]
    h = dt / n_substeps
    z = z0
    for _ in range(n_substeps):
        if method == "euler":
            z = z + h * field(z)
        elif method == "rk4":
            k1 = field(z)
            k2 = field(z + 0.5 * h * k1)
            k3 = field(z + 0.5 * h * k2)
            k4 = field(z + h * k3)
            z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(method)
    return z


class LatentODEJEPA(nn.Module):
    def __init__(self, n_obs, d=8, enc_hidden=128, field_hidden=128,
                 n_substeps=2, method="rk4", ema_decay=None):
        super().__init__()
        self.encoder = mlp([n_obs, enc_hidden, enc_hidden, d])
        self.field = VectorField(d, hidden=field_hidden)
        self.n_substeps = n_substeps
        self.method = method
        self.ema_decay = ema_decay
        if ema_decay is not None:
            self.target_encoder = copy.deepcopy(self.encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)

    def encode(self, obs):
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs):
        if self.ema_decay is None:
            return self.encoder(obs)
        return self.target_encoder(obs)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_decay is None:
            return
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    def step(self, z, dt):
        """One observation-interval prediction: z(t) -> z(t + dt)."""
        return integrate(self.field, z, dt, self.n_substeps, self.method)

    def rollout(self, z0, dts):
        """Roll the field forward through a sequence of intervals dts [B, K].

        Returns predicted latents [B, K, d] (excludes z0).
        """
        preds = []
        z = z0
        for i in range(dts.shape[1]):
            z = self.step(z, dts[:, i])
            preds.append(z)
        return torch.stack(preds, dim=1)
