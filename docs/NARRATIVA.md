# Narrativa del proyecto — descubrimientos ordenados y plan

Documento maestro: qué encontramos, en qué orden contarlo, y con qué seguir.
El README tiene el detalle por fase; esto es la historia.

## La historia en una oración

> Se puede aprender la derivada de un sistema en un espacio latente — el campo
> aprendido es un objeto dinámico genuino — y usarla para clasificar series
> temporales por su dinámica; acá está la anatomía de qué hace falta para que
> funcione, cuándo aplica, y qué queda abierto.

## Los tres actos

### Acto 1 — La capacidad central: el campo aprendido es real

La premisa del proyecto (aprender `dz/ds = g_φ(z)` desde observaciones) quedó
validada por tres vías independientes:

| evidencia | resultado | fase |
|---|---|---|
| Ablación exacta continua vs discreta | discreto degrada 4.5× con sampling irregular; continuo plano (0.19 ± 0.02) | F1 (3 seeds) |
| Autovalores del campo en su punto fijo | frecuencia del sistema real recuperada al 2–5%; Re(λ)≈0 en el conservativo | F5-E1 |
| Consulta en tiempos que no existen en los datos | a 6% del interpolador que usa el futuro, usando solo el pasado | F5-E2 |

### Acto 2 — Anatomía del diseño: qué compra cada pieza

Cada componente de la arquitectura final tiene su ablación:

| pieza | qué compra | evidencia |
|---|---|---|
| Integración continua | robustez a Δt irregular | F1 |
| Decoder (entrenado junto) | precisión de dinámica, robustez a ruido, entrenabilidad | F2–F4 (dos regímenes: denso y pixels) |
| Loss de predicción latente | geometría global del campo (punto fijo sobre la variedad; el modelo solo-decoder falla Newton fuera de sus trayectorias) | F5-E1 |
| Rollout libre en la loss | fuerza campo genuino (vs bloque residual) | diseño F0 + F5-E2 |
| Modelo unificado (todo junto) | paridad en detección (AUROC ≥0.99), primer signo de amortiguamiento consistente | F7 |

Hallazgos metodológicos transversales:
- **El RMSE en espacio de observaciones engaña**: confunde calidad de dinámica
  con calidad de readout (la Fase 2 mostró una "robustez" que era artefacto).
  Métrica correcta: ridge readout-free de latentes rollouteados → estado (F3).
- **El decoder-free es frágil de entrenar** a esta escala: en pixels necesitó
  4 fixes (canal-diferencia, EMA, projector LN, horizonte largo) y aun así
  perdió — costo poco reportado en la literatura JEPA (F4).

### Acto 3 — La aplicación: clasificar por consistencia dinámica

| resultado | números | fase |
|---|---|---|
| Detección de anomalías semi-supervisada funciona | AUROC 0.88–1.00 según tipo | F6 |
| **Punto ciego JEPA**: el score latente no ve fallas que no afectan la dinámica; fix híbrido con stream observacional | 0.78 → 0.98 | F6 |
| Clasificador multi-clase por campos, sintéticos | 100% closed-set; **open-set 100% vs 29% del softmax** | F8 |
| Reality check en datos reales | ver abajo | F9–F10 |

El reality check produjo los hallazgos más finos:

1. **Condición de alcance** (el descubrimiento clave del acto): el método
   clasifica *leyes dinámicas*, no *programas de control*. CharacterTrajectories
   — misma física de lapicera, distinto programa por letra — colapsa a 39%;
   UCI HAR — actividades que sí son regímenes dinámicos — mantiene ventaja
   open-set 5× (34% vs 7%). Esto define el dominio de aplicación del método.
2. **El scoring importa y es no-trivial en datos reales**: rollout para
   detectar *desvíos dentro* de una serie (acumula evidencia), one-step para
   clasificar series *enteras* (evita deriva); el z-score con signo convierte
   clases de residual grande en atractores de series fáciles — el score de
   tipicidad `|z|` lo corrige pero con trade-offs.
3. **Causa raíz del gap restante, identificada**: encoders por clase reciben
   inputs off-distribution de las otras clases → residual = ruido. Fix
   arquitectural pendiente: encoder/decoder compartidos + un campo por clase.

## Los 10 descubrimientos, por importancia

1. La dinámica continua latente absorbe sampling irregular (ablación exacta, F1).
2. El decoder es necesario: ancla el encoder y afila la dinámica; decoder-free
   pierde en precisión, ruido y entrenabilidad en dos regímenes (F2–F4).
3. El campo aprendido contiene la física real: constantes recuperables por
   autovalores, consultable en tiempos arbitrarios (F5).
4. Disociación on/off-manifold, en su forma final (F5-E1 + F12, 5 seeds):
   **dónde vive la loss determina dónde es confiable el campo** — la
   predicción latente por frame compra estructura confiable sobre la variedad
   de datos (Newton 100% vs 40–60%, frecuencias al 3% vs 11%); ninguna loss
   compra estructura donde los datos no van (el punto fijo inestable de Van
   der Pol se le escapa a todos); la reconstrucción sola deja poco confiable
   incluso la estructura adyacente a los datos.
5. Punto ciego JEPA en detección de anomalías + fix híbrido (F6).
6. Clasificación open-set por dinámica: lo que un discriminativo no puede
   hacer por construcción (F8: 100% vs 29%).
7. Condición de alcance: leyes dinámicas sí, programas de control no (F9–F10).
8. La métrica obvia (obs-RMSE) engaña; evaluación readout-free (F2→F3).
9. El diseño del score depende de la tarea (rollout/one-step/tipicidad) (F6, F9–F10).
10. Fragilidad de entrenamiento del decoder-free como costo real (F4).

## Dos writeups posibles

**A. Corto (workshop / ICLR Blogposts / arXiv) — listo para escribir hoy.**
"Anatomía de la dinámica latente continua": Actos 1–2 completos (descubrimientos
1–4, 8, 10). Todo con seeds y figuras. Falta solo: related work y redacción.

**B. Largo (TMLR / venue aplicada).**
"Clasificación de series temporales por consistencia dinámica": Acto 3 con el
Acto 1–2 comprimido como estudio de diseño. La arquitectura de encoder
compartido ya se probó y refutó (F11) — el resultado negativo bien medido está;
bloqueantes restantes: (i) baseline Neural CDE, (ii) 3 seeds en F9–F11, y
(iii) decidir si el gap de closed-set se ataca (scoring multi-escala) o se
publica como problema abierto documentado.

## Con qué seguimos (en orden)

1. ~~Encoder compartido + campo por clase~~ — **probado y refutado** (F11):
   no cerró el gap real (HAR 40% vs 43–57%; caracteres 32% vs 39%). El gap no
   era (solo) encoding off-distribution. La condición de alcance se afina:
   la ley dinámica debe diferir *a la escala temporal que sondea el residual*
   — caminar y subir escaleras comparten la oscilación de marcha local.
   Problema abierto: scoring multi-escala.
2. **Escribir el writeup A** — todo cerrado, incluida la demostración robusta
   de F12 (5 seeds, Van der Pol como borde del insight, flow test nulo).
3. Para el writeup B: Neural CDE + seeds + decisión sobre el gap (atacar con
   scoring multi-escala o documentar como problema abierto).
4. Opcionales: PhysioNet, amortiguamiento con más training, campo SDE.
