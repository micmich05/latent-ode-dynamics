"""Baselines for Phase 1.

- DiscreteJEPA: identical to ours except the predictor is an unconstrained MLP
  conditioned on dt (no integration). Isolates the ODE inductive bias.
- GRUForecaster: autoregressive next-observation prediction in obs space,
  dt as input feature. The standard discrete-time reference.
- LatentODEBaseline: deterministic Rubanova-style latent ODE — GRU encoder over
  the context, same vector field class, but trained through a DECODER with
  reconstruction loss. Isolates "where the loss lives" (obs space vs latent).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LatentODEJEPA, VectorField, mlp


class DiscreteJEPA(LatentODEJEPA):
    def __init__(self, n_obs, d=8, enc_hidden=128, pred_hidden=128, ema_decay=None):
        super().__init__(n_obs, d, enc_hidden, field_hidden=pred_hidden, ema_decay=ema_decay)
        del self.field  # replaced by the discrete predictor
        self.predictor = mlp([d + 1, pred_hidden, pred_hidden, d])

    def step(self, z, dt):
        if not torch.is_tensor(dt):
            dt = torch.full((z.shape[0],), float(dt))
        if dt.dim() == 1:
            dt = dt[:, None]
        return z + self.predictor(torch.cat([z, dt], dim=-1))


class GRUForecaster(nn.Module):
    def __init__(self, n_obs, hidden=128):
        super().__init__()
        self.cell = nn.GRUCell(n_obs + 1, hidden)
        self.head = mlp([hidden, hidden, n_obs])
        self.hidden = hidden

    def _teacher_forced(self, obs, times):
        """Predict x_{t+1} from (x_0..x_t, dts). Returns preds [B, T-1, n]."""
        B, T, _ = obs.shape
        dts = times[:, 1:] - times[:, :-1]
        h = torch.zeros(B, self.hidden)
        preds = []
        for t in range(T - 1):
            h = self.cell(torch.cat([obs[:, t], dts[:, t:t + 1]], dim=-1), h)
            preds.append(self.head(h))
        return torch.stack(preds, dim=1)

    def loss(self, obs, times):
        return F.mse_loss(self._teacher_forced(obs, times), obs[:, 1:])

    @torch.no_grad()
    def forecast(self, obs, times, context):
        """Consume context teacher-forced, then feed back own predictions."""
        B, T, _ = obs.shape
        dts = times[:, 1:] - times[:, :-1]
        h = torch.zeros(B, self.hidden)
        x = obs[:, 0]
        for t in range(context - 1):
            h = self.cell(torch.cat([obs[:, t], dts[:, t:t + 1]], dim=-1), h)
            x = self.head(h)
        preds = [x]  # prediction for t = context
        for t in range(context, T - 1):
            h = self.cell(torch.cat([x, dts[:, t:t + 1]], dim=-1), h)
            x = self.head(h)
            preds.append(x)
        return torch.stack(preds, dim=1)  # [B, T-context, n]


class LatentODEBaseline(nn.Module):
    """Deterministic Latent ODE: GRU context encoder -> z at context end ->
    integrate the field over future times -> decode -> reconstruction loss."""

    def __init__(self, n_obs, d=8, hidden=128, n_substeps=2, method="rk4"):
        super().__init__()
        self.enc_gru = nn.GRU(n_obs + 1, hidden, batch_first=True)
        self.enc_out = nn.Linear(hidden, d)
        self.field = VectorField(d, hidden=hidden)
        self.decoder = mlp([d, hidden, hidden, n_obs])
        self.n_substeps = n_substeps
        self.method = method

    def encode_context(self, obs_ctx, times_ctx):
        dts = times_ctx[:, 1:] - times_ctx[:, :-1]
        dts = torch.cat([torch.zeros(obs_ctx.shape[0], 1), dts], dim=1)
        h, _ = self.enc_gru(torch.cat([obs_ctx, dts[..., None]], dim=-1))
        return self.enc_out(h[:, -1])  # latent state at t_{context-1}

    def rollout(self, z0, dts):
        from .model import integrate
        preds, z = [], z0
        for i in range(dts.shape[1]):
            z = integrate(self.field, z, dts[:, i], self.n_substeps, self.method)
            preds.append(z)
        return torch.stack(preds, dim=1)

    def loss(self, obs, times, context):
        z_c = self.encode_context(obs[:, :context], times[:, :context])
        dts = times[:, context:] - times[:, context - 1:-1]
        z_roll = self.rollout(z_c, dts)
        recon_future = F.mse_loss(self.decoder(z_roll), obs[:, context:])
        recon_anchor = F.mse_loss(self.decoder(z_c), obs[:, context - 1])
        return recon_future + recon_anchor

    @torch.no_grad()
    def forecast(self, obs, times, context):
        z_c = self.encode_context(obs[:, :context], times[:, :context])
        dts = times[:, context:] - times[:, context - 1:-1]
        return self.decoder(self.rollout(z_c, dts))


def train_obs_space_model(model, data, context=10, epochs=200, batch_size=128,
                          lr=1e-3, seed=0, verbose=False):
    """Shared loop for GRUForecaster / LatentODEBaseline (loss lives in obs space)."""
    obs, times = data["train"]["obs"], data["train"]["times"]
    n_traj = obs.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(n_traj, generator=g)
        for i in range(0, n_traj, batch_size):
            idx = perm[i:i + batch_size]
            if isinstance(model, LatentODEBaseline):
                loss = model.loss(obs[idx], times[idx], context)
            else:
                loss = model.loss(obs[idx], times[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}: loss {loss.item():.4f}")
    return model


@torch.no_grad()
def obs_space_forecast_rmse(model, data, context=10, split="test", target="obs"):
    obs, times = data[split]["obs"], data[split]["times"]
    preds = model.forecast(obs, times, context)
    err = ((preds - data[split][target][:, context:]) ** 2).mean(dim=(0, 2)).sqrt()
    return {"per_step_rmse": err.tolist(), "mean_rmse": err.mean().item()}
