"""The unified model: per-frame encoder + latent vector field + decoder.

Synthesis of the design study (phases 1-6):
  - continuous integration of g_phi        -> handles irregular sampling (phase 1)
  - decoder trained jointly                -> anchors the encoder, sharpens the
                                              dynamics (phases 2-4)
  - per-frame encoder + one-step loss over
    the whole manifold                     -> globally well-behaved field (E1)

Loss = latent one-step + latent rollout + VICReg (mild)
     + DECODED one-step and rollout vs the true FUTURE observations
       (this is where the decoder's gradient reaches g_phi)
     + reconstruction anchor decode(z_t) vs x_t (this one only trains f/D).
"""

import torch.nn.functional as F

from .losses import jepa_losses
from .model import LatentODEJEPA, mlp

# Weights for the decoder-side terms, passed to train(..., extra_weights=...).
UNIFIED_WEIGHTS = {"recon": 1.0, "dec_pred": 1.0, "dec_roll": 1.0}


class UnifiedLatentODE(LatentODEJEPA):
    def __init__(self, n_obs, d=8, enc_hidden=128, field_hidden=128,
                 dec_hidden=128, **kw):
        super().__init__(n_obs, d, enc_hidden, field_hidden, **kw)
        self.decoder = mlp([d, dec_hidden, dec_hidden, n_obs])

    def decode(self, z):
        return self.decoder(z)


def unified_losses(model, obs, times, rollout_horizon=8, detach_targets=True):
    L = jepa_losses(model, obs, times, rollout_horizon, detach_targets)
    # Anchor (k=0): decoder sees ENCODED latents; trains encoder+decoder only.
    L["recon"] = F.mse_loss(model.decoder(L["z"]), obs)
    # Prediction (k>=1): decoder sees INTEGRATED latents, targets are the true
    # future observations -- the gradient path decoder -> integrator -> g_phi.
    L["dec_pred"] = F.mse_loss(model.decoder(L["_pred_next"]), obs[:, 1:])
    s, H = L["_roll_start"], L["_z_roll"].shape[1]
    L["dec_roll"] = F.mse_loss(model.decoder(L["_z_roll"]), obs[:, s + 1:s + 1 + H])
    return L


def train_unified(model, data, **kw):
    from .train import train
    return train(model, data, loss_fn=unified_losses,
                 extra_weights=UNIFIED_WEIGHTS, **kw)
