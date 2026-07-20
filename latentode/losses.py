"""Latent prediction loss, VICReg-style anti-collapse regularizers, collapse metrics."""

import torch
import torch.nn.functional as F


def variance_loss(z, target_std=1.0, eps=1e-4):
    """Hinge on per-dimension std over the batch: pushes each latent dim to stay spread."""
    z = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z):
    """Penalize off-diagonal covariance: dims should not be redundant copies."""
    z = z.reshape(-1, z.shape[-1])
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off**2).sum() / d


@torch.no_grad()
def effective_rank(z):
    """exp(entropy of normalized singular values). ~d if healthy, ~1 if collapsed."""
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(dim=0)
    s = torch.linalg.svdvals(z)
    p = s / s.sum().clamp_min(1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum()).item()


def jepa_losses(model, obs, times, rollout_horizon=8, detach_targets=True):
    """Compute prediction + rollout losses on a batch of trajectories.

    obs: [B, T, n], times: [B, T].
    Returns dict of loss tensors and metrics.
    """
    z = model.encode(obs)                       # [B, T, d]
    with torch.no_grad():
        z_tgt = model.encode_target(obs) if model.ema_decay is not None else z
    if detach_targets:
        z_tgt = z_tgt.detach()

    dts = times[:, 1:] - times[:, :-1]          # [B, T-1]

    # One-step: predict z_{t+1} from encoded z_t, for every consecutive pair.
    B, T, d = z.shape
    z_flat = z[:, :-1].reshape(B * (T - 1), d)
    dt_flat = dts.reshape(B * (T - 1))
    pred_next = model.step(z_flat, dt_flat).reshape(B, T - 1, d)
    loss_pred = F.mse_loss(pred_next, z_tgt[:, 1:])

    # Free rollout from a random start: same field integrated H intervals with no
    # re-encoding. This is what forces g to be a genuine vector field.
    H = min(rollout_horizon, T - 1)
    start = torch.randint(0, T - H, (1,)).item()
    z_roll = model.rollout(z[:, start], dts[:, start:start + H])
    loss_roll = F.mse_loss(z_roll, z_tgt[:, start + 1:start + 1 + H])

    return {
        "pred": loss_pred,
        "roll": loss_roll,
        "var": variance_loss(z),
        "cov": covariance_loss(z),
        "z": z,
        # integrated latents, exposed so a decoder head can supervise them in
        # observation space without recomputing (underscore = not a loss term)
        "_pred_next": pred_next,
        "_z_roll": z_roll,
        "_roll_start": start,
    }
