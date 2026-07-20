# Where the Loss Lives: an Anatomy of Continuous Latent Dynamics Models

*Draft v1 — target: ICLR Blogposts track / NeurIPS workshop / arXiv.*

## Abstract

Latent ODE models learn a vector field $`g_\phi`$ governing the evolution of
a low-dimensional latent state, $`dz/ds = g_\phi(z)`$. Their design space
mixes three ingredients — continuous-time integration, a reconstruction
decoder, and latent-space prediction losses — that the literature tends to
adopt or discard wholesale. We dissect them one at a time on controlled
synthetic systems, with an exact ablation for each ingredient and a
readout-free evaluation protocol, and find that each buys a *different and
localized* property: (i) continuous integration buys robustness to irregular
sampling — an identical model with the integrator replaced by a
Δt-conditioned MLP degrades 4.5× as sampling becomes irregular; (ii) the
decoder buys dynamics accuracy, noise robustness and trainability — dropping
it loses on all three, in dense and pixel observation regimes alike; and
(iii) the per-frame latent prediction loss buys something reconstruction
cannot: globally reliable field *structure*. Stress-testing (iii) across
three systems, three models and five seeds shows the latent-loss field's
fixed points are genuine (Newton converges in 15/15 runs, on the data
manifold, with linearization spectra matching the true systems' eigenvalues
to 2–5%) while the reconstruction-only field's are unreliable (Newton fails
in 40–60% of runs) — yet no loss recovers structure in regions the data
never visits, and short-horizon flow boundedness shows no difference. Our
summary is a locality principle: **where the loss lives determines where the
learned field is trustworthy**. We close by showing why this anatomy matters
downstream, where detector blind spots and classification scope conditions
follow directly from it.

## 1. Introduction

Suppose you want a model that *learns the derivative* of a system: an
encoder $`f_\theta`$ maps observations $`x_t \in \mathbb{R}^n`$ to latents
$`z_t \in \mathbb{R}^d`$ ($`d \ll n`$), and a neural vector field defines
continuous-time dynamics integrated over the actual gaps between
observations. The literature offers you two templates. Latent ODEs
[Rubanova et al., 2019] pair the field with a decoder and train on
reconstruction. Joint-embedding predictive architectures (JEPA) [Assran et
al., 2023; and recently combined with neural ODEs by Beltran-Velez et al.,
2025] drop the decoder and train purely by predicting future latents,
guarding against collapse with EMA targets or variance regularization.

Which template should you use, and why? The papers proposing each rarely
ablate the other's ingredients, and the answers turn out to be non-obvious:
one of the two templates fails outright on a property the other silently
guarantees, an intuitive robustness argument for decoder-free training turns
out to be an artifact of the evaluation metric, and the decoder-free model's
one genuine advantage lives somewhere neither template's authors measure.

This post dissects the design space with one experiment per ingredient. Our
testbed is deliberately small — 2D dynamical systems observed through a
fixed random lift to $`\mathbb{R}^{50}`$ with noise — because it buys three
things that scale cannot: the true state is known (enabling *readout-free*
evaluation of the learned dynamics), the true spectrum is known (enabling
falsifiable tests of what the field learned), and every claim can be run
with multiple seeds in minutes. Contributions:

1. **An exact continuous-vs-discrete ablation** (§4): the continuous model's
   robustness to irregular sampling is caused by the integration itself, not
   by capacity or the latent losses.
2. **A metric trap and its fix** (§5): observation-space error conflates
   dynamics quality with readout quality, and reversed our conclusion about
   noise robustness; we evaluate dynamics through a closed-form ridge probe
   from rolled-out latents to the true state instead.
3. **The locality principle** (§6): with 5 seeds across 3 systems, per-frame
   latent prediction produces fields whose fixed-point structure is
   categorically more reliable than reconstruction-only training — including
   physical constants recoverable from linearization spectra — while Van der
   Pol's unstable interior shows the limit: no loss buys structure where
   data never goes.
4. **Downstream consequences** (§7): two findings in applications follow
   directly from this anatomy — the blind spot of latent-only anomaly
   scores, and the scope condition of dynamics-based classification.

## 2. Background and related work

**Latent ODEs.** Rubanova et al. [2019] encode a sequence into an initial
latent, evolve it with a neural ODE [Chen et al., 2018], and decode each
point, training as a VAE. The decoder is essential to the objective: without
a reconstruction term there is nothing anchoring the latent to the data.
Neural CDEs [Kidger et al., 2020] handle irregular series discriminatively;
GRU-ODE-Bayes [De Brouwer et al., 2019] and latent SDEs add stochasticity.

**JEPA.** Predicting in representation space rather than pixel space, with
collapse prevented by asymmetry (EMA targets) or regularization (VICReg
[Bardes et al., 2022]), underlies I-JEPA and V-JEPA. Recent work brings
JEPA to dynamics: Beltran-Velez et al. [2025] train a JEPA + neural ODE
world model for control; Phys-JEPA [2026] adds physical supervision for
forecasting. TS-JEPA and LaT-PFN apply discrete-time JEPA to time series.

**What is missing** is the factorial view: these works adopt package deals.
We are not proposing a new architecture — we are measuring what each piece
of the existing ones buys, with the controls each claim needs.

## 3. Setup

**Model family.** All variants share the same skeleton: a per-frame MLP
encoder, a vector field $`g_\phi`$ (MLP, autonomous), and integration by
RK4 over the observed gaps. The variants differ only in the training
objective:

- **Latent-loss (JEPA-style)**: one-step latent prediction
  $`\|\tilde z_{t+1} - \mathrm{sg}(z_{t+1})\|^2`$ plus a free-rollout term
  (integrate $`H`$ intervals with no re-encoding — this is what makes
  $`g_\phi`$ a vector field rather than a residual block), plus VICReg
  variance/covariance anti-collapse.
- **Reconstruction-only (latent-ODE-style)**: GRU context encoder → same
  field class → decoder; loss is reconstruction of the decoded rollout
  against future observations.
- **Unified**: per-frame encoder + field + decoder trained jointly with
  latent prediction, decoded prediction of the future, and a reconstruction
  anchor.

**Testbed.** Damped oscillator ($`\omega = 2`$, $`\gamma = 0.15`$),
Lotka-Volterra, and Van der Pol, integrated finely; observations are a
fixed random MLP lift of the 2D state to $`\mathbb{R}^{50}`$ plus Gaussian
noise. Sampling irregularity is a dial: $`\Delta t \sim \Delta t_0 \cdot
U(1-s, 1+s)`$. Three to five seeds per cell.

**Readout-free evaluation.** Freeze everything; encode a context; roll the
field forward; fit a closed-form ridge regression from rolled-out latents to
the true state on training trajectories; report held-out state RMSE. No
trainable readout appears in the metric (see §5 for why this matters).

## 4. What continuous integration buys

The control is exact: replace
$`\tilde z_{t+1} = \mathrm{ODESolve}(g_\phi, z_t, \Delta t)`$ with
$`\tilde z_{t+1} = z_t + \mathrm{MLP}(z_t, \Delta t)`$ and change nothing
else — same encoder, same losses, Δt still an input.

![Forecast error vs sampling irregularity](../assets/phase1_oscillator_gap.png)

Flat versus a 4.5× degradation (0.105 → 0.471 forecast RMSE) as $`s`$ goes
from regular to a 19× gap range. Two honest notes: the discrete variant is
*better* on the regular grid (an unconstrained step is easier to fit when
Δt never varies), and a reconstruction-trained continuous model is equally
flat — the property comes from the integrator, not the objective. The
continuous bias is not free capability; it is an inductive bias that pays
exactly when sampling is irregular.

The same field can also be *queried at times that exist in no data*: on a
double-resolution test grid, integrating half an interval from the past
predicts held-out midpoint states within 6% of an interpolator that peeks
at the future observation. A discrete-time model cannot express the query.

## 5. What the decoder buys (and the metric trap)

The decoder-free argument is seductive: reconstruction forces the latent to
model everything, including noise; prediction in latent space should keep
only what the dynamics needs. Measured in observation space, our experiment
seemed to confirm it — under growing observation noise the decoder-free
model's error stayed flat while the reconstruction model degraded 2.3×
faster.

This was an artifact. The decoder-free model's observation-space error is
dominated by the constant error of its frozen post-hoc readout probe, which
flattens any curve. In the readout-free state-space metric, the verdict
reverses cleanly:

![State accuracy under noise](../assets/phase3_oscillator_state.png)

The reconstruction-trained model learns uniformly better dynamics at every
noise level (non-overlapping 3-seed bands), and both models degrade at the
same absolute rate. **Reconstruction does not pollute the latent; it
anchors it.** We repeated the experiment with pixel observations (rendered
pendulum, noise up to 91% of pixel variance) — the most decoder-hostile
setting we could construct — and the decoder model barely moved while the
decoder-free model sat 4× worse throughout, *after* requiring four fixes to
train at all (frame-difference channel, EMA targets, LayerNorm projector,
long-horizon rollout supervision). Training fragility is a real,
under-reported cost of decoder-free training at this scale.

## 6. What the latent loss buys: the locality principle

So is the latent prediction loss useless? No — its value is somewhere
neither accuracy metric looks.

**A coordinate-free test.** Latent coordinates are arbitrary, so we compare
what survives any smooth change of coordinates: eigenvalues of the field's
linearization at its fixed point. If the latent dynamics is the true
dynamics seen through a smooth map $`h`$, the chain rule makes the latent
Jacobian a similarity transform of the true one — same eigenvalues, with
the unknown $`Dh`$ cancelling. $`\mathrm{Im}(\lambda)`$ is the rotation
frequency; $`\mathrm{Re}(\lambda)`$ the contraction rate. The oscillator's
theoretical value is $`-0.075 \pm 1.999i`$; Lotka-Volterra's is
$`\pm 2.121i`$, purely imaginary because its orbits close.

**Procedure and sanity checks.** Damped Newton on $`g_\phi(z) = 0`$ from
the latent data mean; autograd Jacobian; take the complex pair with largest
imaginary part (the other six eigenvalues describe directions transverse to
the learned 2D manifold). Two checks guard against self-deception: the
Newton residual (is this a genuine zero?) and the distance from the fixed
point to the nearest encoded datum in latent scales (is it where data
lives?).

![Spectrum recovery across systems](../assets/phase12_spectra.png)

| 5 seeds × 3 systems | Newton finds a genuine zero | freq. error (osc / L-V) | fixed point on-manifold |
|---|---|---|---|
| latent loss | **100%** (15/15) | **3% / 3%** | **0.5–0.6 scales** |
| reconstruction only | 40–60% | 11% / 8% | 1.0–2.3 scales |
| unified | 80–100% | 4% / 18% | 0.8–2.6 scales |

The latent-loss field recovers the frequency to a few percent, puts ≈0 real
part on the conservative system, and its fixed points are real objects — on
the manifold, found by Newton every single run. The
reconstruction-only field, trained solely along rollout trajectories, fails
Newton in half its runs and places fixed points off the data manifold: off
its trajectories, that field is unconstrained extrapolation. The phase
portraits make the same point qualitatively — including Van der Pol's limit
cycle appearing in the latent as a diffeomorphic image:

![True vs learned phase portraits](../assets/portraits_true_vs_learned.png)

**The edge, and a null.** Van der Pol's *unstable* fixed point — inside the
limit cycle, where trajectories never linger — defeats every model: the
latent-loss field even gets its stability sign wrong. And a flow test
(perturb latents up to 4 latent scales off-manifold, integrate 5 time
units, measure the distance back) shows *no* difference between models:
all fields are similarly bounded at short horizons.

Assembling the three results gives the principle this post is named for:

> **Where the loss lives determines where the field is trustworthy.**
> Per-frame latent prediction supervises the field across the whole data
> manifold and buys reliable structure exactly there; reconstruction
> supervises only along trajectories and leaves even data-adjacent
> structure unreliable; and no loss buys structure in regions the data
> never visits.

## 7. Why the anatomy matters downstream

Two findings from applying this model family follow directly from the
anatomy — we state them briefly (full experiments in the repository).

**Detector blind spots are representation invariances.** Scoring anomalies
by latent prediction residual inherits the encoder's invariances: a sensor
fault that freezes observation dimensions the dynamics ignores is *invisible*
to a latent-only score (AUROC 0.78) and obvious to a decoded
observation-space stream (0.98 hybrid). The same property that §5 showed
makes the latent lean — discarding dynamics-irrelevant detail — is the
blind spot. Choose scores knowing what the training loss made the encoder
throw away.

**Classification by dynamics has a scope condition.** One field per regime,
assignment by residual typicality, gives perfect closed- and open-set
classification of synthetic regimes (100% rejection of never-seen dynamics,
vs 29% for max-softmax — rejection is architectural: a generative model has
a "none of the above" a softmax cannot express). On real data the method
works exactly where classes differ in their dynamics *law* (human
activities; the open-set advantage survives at 5×) and collapses where they
differ only in the control *program* over shared physics (handwritten
characters). The obvious architectural fix — a shared encoder with
per-class fields — was tested and refuted, sharpening the condition: the
laws must differ at the temporal scale the residual probes.

## 8. Limitations

Everything here is small: 2D systems, $`\mathbb{R}^{50}`$ or 32×32
observations, MLP/CNN models, minutes of CPU/MPS training. That is the
point (control, seeds, falsifiability) and the caveat: at V-JEPA scale the
decoder trade-off could flip, and our pixel experiment — though built to be
decoder-hostile — cannot rule that out. The damping constant sits below
seed resolution for most variants. Real-data classification trails
supervised baselines in closed-set accuracy; the open-set advantage is the
surviving applied claim. Multi-scale residual scoring is the open problem.

## 9. Conclusion

None of the three ingredients of continuous latent dynamics models is
redundant, and none does what the others do: integration buys the clock,
the decoder buys the anchor, the latent loss buys the geometry. Papers
proposing one template or the other are, mostly, choosing which property to
silently give up. The cheapest advice this study supports: keep all three
(the unified model matches the best variant on application metrics and is
the only one recovering the damping sign consistently) — and when you
evaluate a latent dynamics model, do it readout-free, check its fixed
points, and remember that nothing it learned extends past where its data
lived.

## References

- Rubanova, Chen, Duvenaud. *Latent ODEs for Irregularly-Sampled Time
  Series.* NeurIPS 2019. [arXiv:1907.03907](https://arxiv.org/abs/1907.03907)
- Chen, Rubanova, Bettencourt, Duvenaud. *Neural Ordinary Differential
  Equations.* NeurIPS 2018.
- Kidger, Morrill, Foster, Lyons. *Neural Controlled Differential Equations
  for Irregular Time Series.* NeurIPS 2020.
  [arXiv:2005.08926](https://arxiv.org/abs/2005.08926)
- Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance
  Regularization.* ICLR 2022.
- Beltran-Velez et al. *Learning State-Space Models of Dynamic Systems from
  Arbitrary Data using Joint Embedding Predictive Architectures.* 2025.
  [arXiv:2508.10489](https://arxiv.org/abs/2508.10489)
- *Phys-JEPA: Physics-Informed Latent World Models for Multivariate
  Time-Series Forecasting.* 2026.
  [arXiv:2606.16076](https://arxiv.org/abs/2606.16076)
- *Anomaly Detection for Sparse and Irregular Multivariate Time Series with
  Latent SDEs.* 2026. [arXiv:2606.18898](https://arxiv.org/abs/2606.18898)

---

*All experiments, seeds, result JSONs and figure scripts:
[github.com/micmich05/latent-ode-dynamics](https://github.com/micmich05/latent-ode-dynamics).*
