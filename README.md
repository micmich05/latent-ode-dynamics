# latent-ode-dynamics

**Learning the derivative of a system in latent space — and using it to
classify time series by their dynamics.**

An independent research project. An encoder $`f_\theta`$ maps observations
$`x_t \in \mathbb{R}^n`$ to latents $`z_t \in \mathbb{R}^d`$ ($`d \ll n`$),
and a learned vector field $`g_\phi`$ defines continuous-time latent dynamics

```math
\frac{dz(s)}{ds} = g_\phi(z(s)),
```

integrated numerically (Euler/RK4) over the *actual* time gaps between
observations. The project runs a component-by-component design study of this
architecture on controlled synthetic systems, then applies it to anomaly
detection and open-set classification of irregularly-sampled time series —
including two real datasets. Every design decision is backed by its own
ablation; negative results are reported alongside positive ones.

![Forecast error vs sampling irregularity](assets/phase1_oscillator_gap.png)

## Key findings

**Capability — the learned field is a genuine dynamical object.**
1. **Continuous integration absorbs irregular sampling.** In an exact ablation
   (same model, integration replaced by a Δt-conditioned MLP), the discrete
   variant degrades 4.5× as sampling becomes irregular while the continuous
   model stays flat (0.19 ± 0.02 forecast RMSE across the whole sweep; 3
   seeds). *(Phase 1)*
2. **The physics is recoverable from the latent.** Eigenvalues of the learned
   field's Jacobian at its fixed point — invariant under smooth conjugacy —
   reproduce the true system's frequency within 2–5%, with ≈0 real part for
   the conservative system. The field can also be queried at time points that
   exist in no training data, matching an interpolator that peeks at the
   future to within 6%. *(Phase 5)*

![Eigenvalue recovery](assets/phase5_eigenvalues.png)

**Design anatomy — what each component buys.**
3. **The decoder is necessary.** Decoder-free (JEPA-style) training loses to
   joint reconstruction on dynamics accuracy, noise robustness and
   trainability, in two regimes (dense low-dim observations and pixels) —
   measured with a readout-free metric after showing that observation-space
   RMSE conflates dynamics quality with readout quality. *(Phases 2–4)*
4. **But the latent prediction loss buys global field geometry.** The
   JEPA-style field has its fixed point on the data manifold with clean
   Newton convergence; the decoder-only baseline's field is untrained off its
   rollout trajectories (Newton fails 2/6 runs). The unified model —
   per-frame encoder + field + decoder, jointly trained — inherits detection
   parity and is the only variant recovering the damping *sign* consistently.
   *(Phases 5, 7)*

**Application — dynamics-consistency classification.**
5. **Anomaly detection works, and latent-only scoring has a measurable blind
   spot.** Prediction-residual scores catch impulses and regime changes
   (AUROC 0.88–1.00), but a latent-only score is blind to observation-level
   sensor faults (AUROC 0.78) because the encoder learns to discard
   dynamics-irrelevant dimensions; adding a decoded observation-space stream
   closes it (0.98). *(Phase 6)*
6. **Open-set classification by dynamics: 100% vs 29%.** One field per
   regime, assignment by lowest residual: perfect closed-set accuracy on four
   synthetic regimes and **100% rejection of a never-seen regime** (Van der
   Pol), where max-softmax rejection of a supervised GRU reaches 29%.
   *(Phase 8)*
7. **Scope condition, from real data.** The method classifies *dynamics laws*,
   not *control programs*: it collapses on character trajectories (same pen
   physics for every letter; 39% vs GRU's 96%) but keeps a 5× open-set
   advantage on human-activity data (genuine dynamical regimes; 34% vs 7%
   rejection). Remaining gap diagnosed: per-class encoders receive
   off-distribution inputs — pointing to a shared-encoder architecture as the
   next step. *(Phases 9–10)*

## The unified architecture

```
x_t ──► encoder f_θ ──► z_t ──► dz/ds = g_φ(z)  (Euler/RK4 over real Δt)
                         │
                         └──► decoder D ──► x̂
```

Trained jointly with four loss terms (see `latentode/unified.py`):

```math
\mathcal{L} =
\underbrace{\|\tilde z_{t+1} - \mathrm{sg}(z_{t+1})\|^2 + \|\hat z_{t+k} - \mathrm{sg}(z_{t+k})\|^2}_{\text{latent prediction (one-step + free rollout)}}
+ \underbrace{\|D(\hat z_{t+k}) - x_{t+k}\|^2}_{\text{decoded prediction of the future}}
+ \underbrace{\|D(z_t) - x_t\|^2}_{\text{reconstruction anchor}}
```

where $`\hat z`$ denotes latents produced by *integrating the field* — the
decoded-prediction term is the gradient path through which the decoder
teaches $`g_\phi`$ — plus mild VICReg variance/covariance regularization.

## Repository map

| path | contents |
|---|---|
| `latentode/` | library: data generators, models (JEPA, baselines, unified), losses, training, eval, pixel & real-data pipelines |
| `experiments/` | one runnable script per phase (`phase0.py` … `phase10.py`) + figure scripts |
| `results/` | committed JSON results backing every number in the docs (figures regenerable) |
| `assets/` | key figures |
| `docs/LAB_NOTEBOOK.md` | phase-by-phase chronological log with all tables and caveats (Spanish) |
| `docs/NARRATIVA.md` | the findings organized into the project's narrative (Spanish) |

## Reproduce

Python ≥ 3.11.

```bash
pip install -r requirements.txt
bash scripts/download_data.sh   # only needed for phase 10 (UCI HAR)

python3 experiments/phase0.py --system oscillator   # sanity, ~2 min CPU
python3 experiments/phase1.py --seeds 0 1 2         # H1 sweep, ~35 min
python3 experiments/phase3.py                       # decoder study, ~35 min
python3 experiments/phase5_field.py                 # eigenvalue recovery, ~25 min
python3 experiments/phase6.py                       # anomaly detection, ~15 min
python3 experiments/phase8.py                       # open-set classifier, ~15 min
python3 experiments/phase9.py                       # CharacterTrajectories, ~5 min/seed
python3 experiments/phase10.py                      # UCI HAR, ~5 min/seed
```

Phase 4 (pixels) uses Apple MPS if available. Figure scripts
(`experiments/plot_phase*.py`) regenerate all PNGs from the committed JSONs.
All experiments are seeded; the committed `results/*.json` files contain the
exact numbers reported above.

## Data

- **CharacterTrajectories** (UCI ML Repository, CC BY 4.0) — included as
  `data/mixoutALL_shifted.mat`.
- **UCI HAR** (Anguita et al. 2013, UCI ML Repository) — downloaded by
  `scripts/download_data.sh`.
- All synthetic systems (damped oscillator, Lotka-Volterra, Van der Pol,
  rendered pendulum) are generated by `latentode/data.py` / `latentode/pixels.py`.

## Related work

Closest neighbors: latent ODEs ([Rubanova et al. 2019](https://arxiv.org/abs/1907.03907)),
JEPA + neural ODE state-space models ([2508.10489](https://arxiv.org/abs/2508.10489)),
Phys-JEPA ([2606.16076](https://arxiv.org/abs/2606.16076)), latent SDEs for
anomaly detection ([2606.18898](https://arxiv.org/abs/2606.18898)), neural
CDEs ([Kidger et al. 2020](https://arxiv.org/abs/2005.08926)). This project's
distinguishing angle: the exact continuous-vs-discrete ablation, the
readout-free evaluation protocol, the JEPA blind-spot finding, and open-set
classification by dynamics consistency.

## Limitations & next steps

Synthetic systems are low-dimensional; real-data classification still trails
a supervised GRU in closed-set accuracy (the open-set advantage is the
method's edge). Identified next step: shared encoder/decoder with per-class
fields only (fixes off-distribution encoding, cuts per-class cost). Also
open: damping recovery below seed resolution, a Neural CDE baseline,
PhysioNet, and a stochastic (SDE) field.

## License & citation

MIT — see [LICENSE](LICENSE). If you use this work, please cite via
[CITATION.cff](CITATION.cff).
