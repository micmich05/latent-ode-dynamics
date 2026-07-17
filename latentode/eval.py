"""Evaluation protocol: the encoder/field are frozen everywhere here.

- Linear probe latent -> true 2D state (R^2 per dim): did the latent capture the state?
- Decoder probe latent -> observations, then forecast RMSE: encode a context prefix,
  roll the field forward with NO access to future observations, decode, compare.
- Phase portraits: latent trajectories in PCA plane vs true state trajectories.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .model import mlp


@torch.no_grad()
def linear_probe_state(model, data, split="test"):
    """Closed-form ridge regression from latents to true states. Returns R^2 per state dim."""
    ztr = model.encode(data["train"]["obs"]).reshape(-1, model.encoder[-1].out_features)
    str_ = data["train"]["states"].reshape(-1, 2)
    zte = model.encode(data[split]["obs"]).reshape(-1, ztr.shape[1])
    ste = data[split]["states"].reshape(-1, 2)

    ztr1 = torch.cat([ztr, torch.ones(len(ztr), 1)], dim=1)
    zte1 = torch.cat([zte, torch.ones(len(zte), 1)], dim=1)
    reg = 1e-3 * torch.eye(ztr1.shape[1])
    W = torch.linalg.solve(ztr1.T @ ztr1 + reg, ztr1.T @ str_)
    pred = zte1 @ W
    ss_res = ((ste - pred) ** 2).sum(0)
    ss_tot = ((ste - ste.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).tolist()


def train_decoder_probe(model, data, hidden=128, epochs=200, lr=1e-3, seed=0):
    """Fit an MLP z -> obs on train latents (encoder frozen)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z = model.encode(data["train"]["obs"])
    obs = data["train"]["obs"]
    d, n_obs = z.shape[-1], obs.shape[-1]
    dec = mlp([d, hidden, hidden, n_obs])
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    zf, of = z.reshape(-1, d), obs.reshape(-1, n_obs)
    for _ in range(epochs):
        idx = torch.randint(0, len(zf), (1024,))
        loss = F.mse_loss(dec(zf[idx]), of[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return dec


@torch.no_grad()
def forecast_rmse(model, decoder, data, context=10, split="test", target="obs"):
    """Encode obs at t=context-1, roll the field to the end, decode, RMSE per horizon step.

    Also returns the reconstruction RMSE (decode the *encoded* future latents),
    which lower-bounds what forecasting through the field can achieve.
    target="obs_clean" measures error against the noise-free signal (H2): the
    model still only ever SEES noisy obs.
    """
    obs, times = data[split]["obs"], data[split]["times"]
    tgt = data[split][target]
    z_ctx = model.encode(obs[:, context - 1])
    dts = times[:, context:] - times[:, context - 1:-1]
    z_roll = model.rollout(z_ctx, dts)                    # [B, T-context, d]
    obs_pred = decoder(z_roll)
    err = ((obs_pred - tgt[:, context:]) ** 2).mean(dim=(0, 2)).sqrt()  # per-step RMSE

    z_enc = model.encode(obs[:, context:])
    recon = decoder(z_enc)
    recon_rmse = ((recon - tgt[:, context:]) ** 2).mean().sqrt().item()
    return {"per_step_rmse": err.tolist(), "mean_rmse": err.mean().item(),
            "recon_rmse": recon_rmse}


@torch.no_grad()
def phase_portrait(model, data, path, split="test", n_show=12):
    """Side by side: true 2D state trajectories vs latent trajectories in PCA plane."""
    obs = data[split]["obs"][:n_show]
    states = data[split]["states"][:n_show]
    z = model.encode(obs)                                  # [n, T, d]
    zf = z.reshape(-1, z.shape[-1])
    zc = zf - zf.mean(0)
    _, _, V = torch.linalg.svd(zc, full_matrices=False)
    z2 = (zc @ V[:2].T).reshape(z.shape[0], z.shape[1], 2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i in range(len(states)):
        axes[0].plot(states[i, :, 0], states[i, :, 1], alpha=0.7, lw=1)
        axes[0].plot(states[i, 0, 0], states[i, 0, 1], "k.", ms=4)
        axes[1].plot(z2[i, :, 0], z2[i, :, 1], alpha=0.7, lw=1)
        axes[1].plot(z2[i, 0, 0], z2[i, 0, 1], "k.", ms=4)
    axes[0].set_title("True state space")
    axes[1].set_title("Latent space (PCA, first 2 comps)")
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
