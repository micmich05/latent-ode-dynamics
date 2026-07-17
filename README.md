# latent-ode-dynamics

Mini proyecto de research: **predicción puramente latente (estilo JEPA) con dinámica de
tiempo continuo (Neural ODE), sin decoder**, para series temporales.

## Formulación

Sea $`\{x_t\}_{t=0}^T`$ una secuencia de observaciones, $`x_t \in \mathbb{R}^n`$, y
$`f_\theta : \mathbb{R}^n \to \mathbb{R}^d`$ un encoder por frame con $`d \ll n`$,
$`z_t = f_\theta(x_t)`$. Un campo vectorial aprendido
$`g_\phi : \mathbb{R}^d \to \mathbb{R}^d`$ define la dinámica latente

```math
\frac{dz(s)}{ds} = g_\phi(z(s)), \qquad \tilde z_{t+1} = z_t + \int_{t}^{t+1} g_\phi(z(s))\, ds
```

integrada numéricamente (Euler o RK4) sobre el $`\Delta t`$ real entre observaciones —
esto es lo que hace al modelo nativo para sampling irregular.

**Objetivo**:

```math
\mathcal{L} = \underbrace{D(\tilde z_{t+1}, \mathrm{sg}(z_{t+1}))}_{\text{one-step}}
+ \lambda_r \underbrace{D(\hat z_{t+k}, \mathrm{sg}(z_{t+k}))}_{\text{rollout libre, } k \le H}
+ \lambda_v \mathcal{L}_{\text{var}} + \lambda_c \mathcal{L}_{\text{cov}}
```

donde el rollout integra $`g_\phi`$ por $`H`$ intervalos sin re-encodear (fuerza a que
$`g_\phi`$ sea un campo vectorial genuino y no un residual de un paso), y los términos
de varianza y covarianza (estilo VICReg, a nivel batch) previenen el colapso — el rol
que en un Latent ODE clásico cumple el decoder. Targets con stop-gradient; target
encoder EMA opcional.

## Pregunta de investigación

> ¿Puede un modelo que aprende la derivada en el espacio latente, entrenado puramente
> con predicción latente (sin decoder), igualar a los Latent ODEs clásicos
> (Rubanova et al. 2019) en series muestreadas irregularmente, a menor costo?

- **H1**: la ventaja sobre predictores discretos (GRU / JEPA discreta) crece con la
  irregularidad del sampling.
- **H2**: sin decoder, el latente es más robusto a ruido de observación que un
  Latent ODE con reconstrucción.
- **H3**: el campo $`g_\phi`$ recupera la topología del sistema real (espiral, ciclo
  límite) viendo solo observaciones liftadas.

Trabajo más cercano: [JEPA + Neural ODE para state-space models](https://arxiv.org/abs/2508.10489)
(control con acciones, péndulo, evaluación cualitativa) y
[Phys-JEPA](https://arxiv.org/abs/2606.16076). Ninguno caracteriza forecasting sin
acciones bajo sampling irregular.

## Protocolo de evaluación (encoder y campo siempre congelados)

| Criterio | Métrica |
|---|---|
| No colapso | rank efectivo de $`z`$ (exp-entropía de valores singulares) |
| Latente informativo | probe ridge $`z \to`$ estado verdadero, $`R^2`$ |
| Forecasting | encodear contexto, rollout de $`g_\phi`$, decodear con probe MLP post-hoc, RMSE vs piso de reconstrucción y vs persistencia |
| Dinámica real | retrato de fases del latente (PCA) vs estado verdadero |

## Estado

- **Fase 0 (hecha)** — sanity en sintéticos con sampling regular, obs
  $`\mathbb{R}^{50}`$ (lift MLP random fijo + ruido), $`d=8`$:

  | | eff. rank | probe $`R^2`$ | forecast RMSE (50 pasos) | piso recon | persistencia |
  |---|---|---|---|---|---|
  | oscilador amortiguado | 3.8 / 8 | 0.99 | **0.17** | 0.15 | 1.46 |
  | Lotka-Volterra | 3.8 / 8 | 0.99 | **0.25** | 0.22 | 1.36 |

  Retratos de fase: espiral y ciclos cerrados recuperados — H3 pasa
  cualitativamente en ambos sistemas.

  ![retratos de fase](assets/phase0_oscillator_portrait.png)

- **Fase 1 (hecha)** — sweep de irregularidad $`s`$ ($`\Delta t \sim \Delta t_0 \cdot U(1-s, 1+s)`$),
  oscilador, forecast RMSE (contexto 10, 50 pasos):

  | $`s`$ | ours | JEPA discreta | GRU | Latent ODE + decoder |
  |---|---|---|---|---|
  | 0.0 | 0.169 | **0.110** | 0.365 | **0.088** |
  | 0.3 | 0.208 | 0.200 | 0.369 | 0.089 |
  | 0.6 | **0.180** | 0.332 | 0.428 | 0.090 |
  | 0.9 | **0.160** | 0.453 | 0.509 | 0.091 |

  ![gap H1](assets/phase1_oscillator_gap.png)

  **H1 confirmada contra los modelos discretos**: los dos modelos continuos son
  planos en $`s`$, mientras la JEPA discreta degrada 4× (0.11 → 0.45, cruce en
  $`s \approx 0.35`$) y la GRU también empeora. La integración del campo — no la
  loss latente — es lo que absorbe la irregularidad. **Matiz honesto**: el Latent
  ODE con decoder sigue ganando en RMSE absoluto; pero el piso de reconstrucción
  del probe (~0.15) indica que casi todo el error nuestro es del *readout*
  post-hoc, no de la dinámica aprendida. La comparación decisiva es H2 (ruido).

- **Fase 2 (hecha)** — H2: sweep de ruido de observación con $`s=0.9`$ fijo,
  RMSE medido contra la señal **limpia** (los modelos solo ven la ruidosa):

  | ruido (% varianza) | 0% | 7% | 33% | 60% | 78% |
  |---|---|---|---|---|---|
  | ours | 0.149 | 0.156 | 0.181 | 0.165 | 0.211 |
  | Latent ODE + decoder | **0.071** | **0.084** | **0.127** | **0.157** | **0.166** |

  ![robustez a ruido](assets/phase2_oscillator_noise.png)

  **H2 parcialmente soportada, sin cruce**: el mecanismo aparece — el Latent ODE
  con decoder degrada 2.3× más rápido (+133% vs +41% de RMSE) y en 60% de ruido
  quedan empatados — pero nuestro handicap constante de readout (probe post-hoc,
  piso ~0.15) impide que el orden se invierta en este rango. La comparación en
  espacio de observaciones conflata calidad de la *dinámica* con calidad del
  *readout*; el siguiente experimento debería comparar en espacio de estado
  (probe ridge $`z \to`$ estado verdadero bajo ruido), que es independiente del
  readout. Caveat: una sola seed (la curva nuestra es no-monótona por varianza
  de entrenamiento); un writeup serio necesita 3+ seeds con barras de error.

- **Fase 3** — H2 en espacio de estado + multi-seed; después serie real
  irregular (PhysioNet) o forecasting estándar (ETT).

## Correr

```bash
python3 experiments/phase0.py --system oscillator
python3 experiments/phase0.py --system lotka_volterra
python3 experiments/phase0.py --system oscillator --jitter 0.8  # sampling irregular
python3 experiments/phase1.py --system oscillator               # sweep H1 (baselines)
```

Requiere `torch`, `numpy`, `matplotlib`. Resultados (JSON + PNG) en `results/`.
