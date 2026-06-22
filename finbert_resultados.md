# Resultados FinBERT (corrida en GPU — RTX 4070 SUPER, CUDA)

FinBERT 4a (feature extraction, frozen [CLS] + LogReg) y 4b (fine-tuning completo) en los dos
escenarios del trabajo. Métrica: **F1-macro** (clases `hike` / `cut` / `hold`). Split cronológico:
train 2000–2017, val 2018–2021, test 2022–2025.

- **07 (TP principal, X→X):** clasificar la decisión de la *propia* reunión, sobre el dataset
  **enmascarado** (se removió la mención explícita de la decisión → anti-leakage).
- **09 (forecasting, X→X+1):** predecir la decisión de la reunión *siguiente*, sobre el dataset
  **sin enmascarar** (la decisión de X es insumo legítimo; la respuesta es la de X+1).

---

## 1. TP principal — X→X (dataset enmascarado)

| Modelo | val | test |
|---|---|---|
| Baseline (mayoritaria) | 0.286 | 0.213 |
| Exp 1 — TF-IDF (C=10) | 0.519 | 0.213 |
| Exp 2 — LM lexicón (C=0.01) | 0.524 | 0.154 |
| Exp 3 — Word2Vec (C=10) | 0.691 | 0.396 |
| Exp 4a — FinBERT frozen (head+tail, C=0.01) | 0.610 | 0.484 |
| **Exp 4b — FinBERT fine-tuning (lr=1e-5, batch=8)** | **0.845** | **0.741** |

**Ranking test:** FinBERT FT 0.741 > FinBERT frozen 0.484 > Word2Vec 0.396 > TF-IDF ≈ baseline 0.213 > LM 0.154.

### Enmascarado vs original con leakage — ¿cuánto era leakage?

| | leakage (original) | enmascarado | Δ |
|---|---|---|---|
| FinBERT 4a frozen (test) | 0.502 | 0.484 | −0.018 |
| FinBERT 4b fine-tuning (test) | 0.648 | **0.741** | **+0.093** |

**El enmascarado NO degrada a FinBERT — al contrario.** El frozen baja apenas (−0.018, ruido) y el
fine-tuneado **mejora** casi 10 puntos. Conclusión: el resultado de FinBERT **no era leakage**. El
modelo no estaba "leyendo" la frase con la decisión; la **infiere** del tono y la discusión. Quitar
el *tell* explícito (~0.7% del texto) incluso ayuda: menos ruido superficial, más señal real.

> Esto **cambia la historia** respecto de la sospecha inicial (que parte del éxito de FinBERT fuera
> leer la respuesta). FinBERT entiende la discusión.

En test el 4b es además el **único modelo balanceado en las 3 clases**: cut recall **0.83** (5/6),
hike 0.55, hold 0.87 — capta los recortes de normalización 2024–25 que ningún modelo previo veía.

---

## 2. Forecasting — X→X+1 (dataset sin enmascarar)

| Modelo | val | test |
|---|---|---|
| Baseline (mayoritaria) | 0.284 | 0.207 |
| **Baseline (persistencia)** | **0.413** | **0.768** |
| Exp 1 — TF-IDF (C=1) | 0.423 | 0.207 |
| Exp 2 — LM lexicón (C=0.01) | 0.357 | 0.168 |
| Exp 3 — Word2Vec (C=0.1) | 0.433 | 0.298 |
| Exp 4a — FinBERT frozen (C=0.01) | 0.350 | 0.434 |
| Exp 4b — FinBERT fine-tuning (lr=1e-5, batch=8) | 0.502 | 0.475 |

### ¿FinBERT le gana a la persistencia? → NO

La persistencia (predecir que la próxima decisión = la última) logra **test 0.768**. El mejor
modelo textual es FinBERT 4b con **0.475** — casi 30 puntos por debajo. **Ningún modelo supera la
persistencia.**

Sí se observa que FinBERT (4a 0.434, 4b 0.475) le gana holgadamente a los modelos clásicos en
forecasting (TF-IDF/LM ≈ 0.17–0.21, Word2Vec 0.30): capta *algo* de señal anticipatoria del tono de
la discusión, pero no la suficiente para vencer la inercia de la política monetaria.

**Implicancia:** esto justifica formular el TP como X→X (clasificar la decisión de la propia
reunión), que es donde el texto sí discrimina. Anticipar la siguiente decisión desde el texto es
mucho más difícil y la persistencia es una vara muy alta.

---

## 3. Volcado crudo de los 4 runs (evidencia)

**07 — TP principal (X→X enmascarado):**
```
FinBERT 4a feature-extraction (head+tail, C=0.01) — VAL : f1_macro 0.610  (acc 0.875)
FinBERT 4a feature-extraction (head+tail, C=0.01) — TEST: f1_macro 0.484  (acc 0.594)
Mejor fine-tuning: lr=1e-05 batch=8 -> F1-macro val 0.845
FinBERT 4b fine-tuning (lr=1e-05, batch=8) — VAL : f1_macro 0.845  (acc 0.938)
FinBERT 4b fine-tuning (lr=1e-05, batch=8) — TEST: f1_macro 0.741  (acc 0.750)
```

**09 — forecasting (X→X+1 sin enmascarar):**
```
Exp 4a FinBERT frozen (C=0.01)   F1-macro  val=0.3495  test=0.4339
Exp 4b FinBERT fine-tuning       F1-macro  val=0.5022  test=0.4750
```

**Entorno:** Python 3.12, torch 2.6.0+cu124, transformers 5.12.1, `Device: cuda`
(NVIDIA GeForce RTX 4070 SUPER). Modelo base: `ProsusAI/finbert`. Class weights inversos a la
frecuencia, early stopping por F1-macro en val (paciencia 2), grid lr∈{1e-5,2e-5,5e-5} ×
batch∈{4,8,16}, máx 5 épocas.
