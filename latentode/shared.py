"""Shared-encoder dynamics classifier: ONE encoder + ONE decoder + K fields.

The fix identified in phases 9-10: with per-class models, out-of-class series
are off-distribution inputs to every other class's encoder, so their
residuals are noise rather than evidence. Here the encoder/decoder are
trained on ALL classes jointly — the latent space is common, residuals are
comparable by construction — and each class owns only a vector field
g_phi^(c). Per-class cost drops from a full model to one field.

This also makes the experiment a clean test of the scope condition: any
remaining failure is conceptual (classes don't differ in dynamics laws),
not an encoding artifact.
"""

import torch
import torch.nn.functional as F

from .losses import covariance_loss, variance_loss
from .model import VectorField, integrate, mlp


class SharedDynamicsClassifier(torch.nn.Module):
    def __init__(self, n_obs, n_classes, d=8, enc_hidden=128, field_hidden=128,
                 dec_hidden=128, n_substeps=2, method="rk4"):
        super().__init__()
        self.encoder = mlp([n_obs, enc_hidden, enc_hidden, d])
        self.decoder = mlp([d, dec_hidden, dec_hidden, n_obs])
        self.fields = torch.nn.ModuleList(
            [VectorField(d, hidden=field_hidden) for _ in range(n_classes)])
        self.n_substeps = n_substeps
        self.method = method

    def encode(self, obs):
        return self.encoder(obs)

    def step(self, z, dt, c):
        return integrate(self.fields[c], z, dt, self.n_substeps, self.method)

    def rollout(self, z0, dts, c):
        preds, z = [], z0
        for i in range(dts.shape[1]):
            z = self.step(z, dts[:, i], c)
            preds.append(z)
        return torch.stack(preds, dim=1)


def shared_losses(model, obs, times, c, rollout_horizon=8,
                  lambda_var=1.0, lambda_cov=0.1):
    """Unified-model losses for a batch of class-c series through field c."""
    z = model.encode(obs)
    z_tgt = z.detach()
    dts = times[:, 1:] - times[:, :-1]
    B, T, d = z.shape

    pred_next = model.step(z[:, :-1].reshape(-1, d), dts.reshape(-1), c).reshape(B, T - 1, d)
    loss = F.mse_loss(pred_next, z_tgt[:, 1:])

    H = min(rollout_horizon, T - 1)
    start = torch.randint(0, T - H, (1,)).item()
    z_roll = model.rollout(z[:, start], dts[:, start:start + H], c)
    loss = loss + F.mse_loss(z_roll, z_tgt[:, start + 1:start + 1 + H])

    loss = loss + F.mse_loss(model.decoder(pred_next), obs[:, 1:])
    loss = loss + F.mse_loss(model.decoder(z_roll), obs[:, start + 1:start + 1 + H])
    loss = loss + F.mse_loss(model.decoder(z), obs)
    loss = loss + lambda_var * variance_loss(z) + lambda_cov * covariance_loss(z)
    return loss


def train_shared(model, per_class_data, epochs=300, batch_size=64, lr=1e-3,
                 seed=0, verbose=False):
    """per_class_data: {class_idx: {"obs": [N,T,n], "times": [N,T]}}.
    Each epoch visits every class; encoder/decoder accumulate all classes'
    gradients, each field only its own."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    classes = list(per_class_data)
    for epoch in range(epochs):
        order = [classes[i] for i in torch.randperm(len(classes), generator=g)]
        for c in order:
            obs, times = per_class_data[c]["obs"], per_class_data[c]["times"]
            perm = torch.randperm(len(obs), generator=g)
            for i in range(0, len(obs), batch_size):
                idx = perm[i:i + batch_size]
                loss = shared_losses(model, obs[idx], times[idx], c)
                opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: last-class loss {loss.item():.4f}", flush=True)
    return model
