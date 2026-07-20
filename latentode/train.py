"""Training loop for the latent-ODE JEPA."""

import torch

from .losses import effective_rank, jepa_losses


def train(model, data, epochs=300, batch_size=128, lr=1e-3, weight_decay=1e-5,
          lambda_pred=1.0, lambda_roll=1.0, lambda_var=1.0, lambda_cov=0.1,
          rollout_horizon=8, log_every=25, seed=0, verbose=True,
          loss_fn=None, extra_weights=None):
    loss_fn = loss_fn or jepa_losses
    extra_weights = extra_weights or {}
    obs, times = data["train"]["obs"], data["train"]["times"]
    n_traj = obs.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(n_traj, generator=g)
        ep = {"pred": 0.0, "roll": 0.0, "var": 0.0, "cov": 0.0}
        n_batches = 0
        for i in range(0, n_traj, batch_size):
            idx = perm[i:i + batch_size]
            losses = loss_fn(model, obs[idx], times[idx], rollout_horizon)
            total = (lambda_pred * losses["pred"] + lambda_roll * losses["roll"]
                     + lambda_var * losses["var"] + lambda_cov * losses["cov"]
                     + sum(w * losses[k] for k, w in extra_weights.items()))
            opt.zero_grad()
            total.backward()
            opt.step()
            model.update_ema()
            for k in ep:
                ep[k] += losses[k].item()
            n_batches += 1
        sched.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            with torch.no_grad():
                z_all = model.encode(obs)
            rank = effective_rank(z_all)
            row = {k: v / n_batches for k, v in ep.items()}
            row.update(epoch=epoch + 1, eff_rank=rank)
            history.append(row)
            if verbose:
                print(f"epoch {epoch+1:4d} | pred {row['pred']:.4f} | roll {row['roll']:.4f} "
                      f"| var {row['var']:.4f} | cov {row['cov']:.4f} | eff_rank {rank:.2f}")
    return history
