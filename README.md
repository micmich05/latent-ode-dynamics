# latent-ode-dynamics

Mini proyecto de research: **predicción puramente latente (estilo JEPA) con dinámica de
tiempo continuo (Neural ODE), sin decoder**, para series temporales.

## Formulación

Sea $\{x_t\}_{t=0}^T$ una secuencia de observaciones, $x_t \in \mathbb{R}^n$, y
$f_\theta : \mathbb{R}^n \to \mathbb{R}^d$ un encoder por frame con $d \ll n$,
$z_t = f_\theta(x_t)$. Un campo vectorial aprendido
$g_\phi : \mathbb{R}^d \to \mathbb{R}^d$ define la dinámica latente

$$\frac{dz(s)}{ds} = g_\phi(z(s)), \qquad \tilde z_{t+1} = z_t + \int_{t}^{t+1} g_\phi(z(s))\,ds$$

integrada numéricamente (Euler o RK4) sobre el $\Delta t$ real entre observaciones —
esto es lo que hace al modelo nativo para sampling irregular.

**Objetivo**: $\mathcal{L} = \underbrace{D(\tilde z_{t+1}, \mathrm{sg}(z_{t+1}))}_{\text{one-step}}
+ \lambda_r \underbrace{D(\hat z_{t+k}, \mathrm{sg}(z_{t+k}))}_{\text{rollout libre, } k \le H}
+ \lambda_v \mathcal{L}_{\text{var}} + \lambda_c \mathcal{L}_{\text{cov}}$

donde el rollout integra $g_\phi$ por $H$ intervalos sin re-encodear (fuerza a que $g$
sea un campo vectorial genuino y no un residual de un paso), y los términos de varianza
y covarianza (estilo VICReg, a nivel batch) previenen el colapso — el rol que en un
Latent ODE clásico cumple el decoder. Targets con stop-gradient; target encoder EMA
opcional.

## Pregunta de investigación

> ¿Puede un modelo que aprende la derivada en el espacio latente, entrenado puramente
> con predicción latente (sin decoder), igualar a los Latent ODEs clásicos
> (Rubanova et al. 2019) en series muestreadas irregularmente, a menor costo?

- **H1**: la ventaja sobre predictores discretos (GRU / JEPA discreta) crece con la
  irregularidad del sampling.
- **H2**: sin decoder, el latente es más robusto a ruido de observación que un
  Latent ODE con reconstrucción.
- **H3**: el campo $g_\phi$ recupera la topología del sistema real (espiral, ciclo
  límite) viendo solo observaciones liftadas.

Trabajo más cercano: [JEPA + Neural ODE para state-space models](https://arxiv.org/abs/2508.10489)
(control con acciones, péndulo, evaluación cualitativa) y
[Phys-JEPA](https://arxiv.org/abs/2606.16076). Ninguno caracteriza forecasting sin
acciones bajo sampling irregular.

## Protocolo de evaluación (encoder y campo siempre congelados)

| Criterio | Métrica |
|---|---|
| No colapso | rank efectivo de $z$ (exp-entropía de valores singulares) |
| Latente informativo | probe ridge $z \to$ estado verdadero, $R^2$ |
| Forecasting | encodear contexto, rollout de $g_\phi$, decodear con probe MLP post-hoc, RMSE vs piso de reconstrucción y vs persistencia |
| Dinámica real | retrato de fases del latente (PCA) vs estado verdadero |

## Estado

- **Fase 0 (hecha)** — sanity en sintéticos con sampling regular, obs $\mathbb{R}^{50}$
  (lift MLP random fijo + ruido), $d=8$:

  | | eff. rank | probe $R^2$ | forecast RMSE (50 pasos) | piso recon | persistencia |
  |---|---|---|---|---|---|
  | oscilador amortiguado | 3.8 / 8 | 0.99 | **0.17** | 0.15 | 1.46 |
  | Lotka-Volterra | 3.8 / 8 | 0.99 | **0.25** | 0.22 | 1.36 |

  Retratos de fase: espiral y ciclos cerrados recuperados (`results/*_portrait.png`).
  H3 pasa cualitativamente en ambos sistemas.

- **Fase 1 (siguiente)** — la tesis: sampling irregular con $\Delta t$ variable.
  Baselines: GRU discreta, JEPA discreta ($\Delta t$ como feature), Latent ODE con
  decoder. Barrer la varianza de $\Delta t$ y medir el gap (H1). Euler vs RK4.
- **Fase 2** — serie real irregular (PhysioNet) o forecasting estándar (ETT).

## Correr

```bash
python3 experiments/phase0.py --system oscillator
python3 experiments/phase0.py --system lotka_volterra
python3 experiments/phase0.py --system oscillator --irregular   # anticipo de fase 1
```

Requiere `torch`, `numpy`, `matplotlib`. Resultados (JSON + PNG) en `results/`.
