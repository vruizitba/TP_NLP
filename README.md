# Clasificación de decisiones de política monetaria de la FOMC

Trabajo Práctico de NLP (ITBA, 2do cuatrimestre 2026). Clasificación de texto
multiclase: predecir la decisión de tasa de la **FOMC** — `hike` / `cut` / `hold` —
a partir del texto de las *minutes* (actas) de sus reuniones.

## Objetivo

Dado el texto de un acta de la FOMC, predecir qué hizo el comité con la *federal
funds target rate* en esa reunión: subirla (`hike`), bajarla (`cut`) o mantenerla
(`hold`). La etiqueta se deriva del cambio de tasa **antes vs. después de cada
reunión** (no entre reuniones consecutivas, lo cual desfasaba la etiqueta un
período).

## Datos

- **Corpus:** minutas de la FOMC (inglés) de federalreserve.gov, período **2000–2025**.
- **Tasas:** *federal funds target rate* de FRED (St. Louis Fed): `DFEDTAR`
  (pre-2008) y `DFEDTARU` / `DFEDTARL` (régimen de banda post-2008).
- **209 documentos**, ~8000 palabras promedio cada uno.
- Dataset final: [`data/processed/fomc_dataset.csv`](data/processed/fomc_dataset.csv)
  con columnas `date, text, label, split`.

**Distribución de clases** (desbalanceada → la métrica principal es **F1-macro**):

| Clase  | Docs | %     |
|--------|------|-------|
| `hold` | 141  | 67.5% |
| `hike` | 40   | 19.1% |
| `cut`  | 28   | 13.4% |

**Split cronológico** (no aleatorio: hay *drift* lingüístico ~2005 y estructura de
régimen en las etiquetas; un split aleatorio filtraría información):

| Split | Período   | Docs |
|-------|-----------|------|
| Train | 2000–2017 | 145  |
| Val   | 2018–2021 | 32   |
| Test  | 2022–2025 | 32   |

## Experimentos y resultados

Cinco enfoques, de léxico a contexto+adaptación; cada paso corrige el límite del
anterior. Métrica: **F1-macro**.

| #   | Modelo                              | Val   | Test      |
|-----|-------------------------------------|-------|-----------|
| —   | Baseline (clase mayoritaria)        | 0.286 | 0.213     |
| 1   | TF-IDF + LogReg                     | 0.519 | 0.213     |
| 2   | Léxico Loughran-McDonald + LogReg   | 0.489 | 0.154     |
| 3   | Word2Vec (mean pooling) + LogReg    | 0.606 | 0.337     |
| 4a  | FinBERT `[CLS]` congelado + LogReg  | 0.643 | 0.502     |
| 4b  | **FinBERT fine-tuning**             | 0.845 | **0.648** |

**Ranking final (F1-macro en test):** FinBERT FT `0.648` > FinBERT congelado
`0.502` > Word2Vec `0.337` > TF-IDF `0.213` ≈ baseline > LM `0.154`.

Lecturas clave:
- TF-IDF sobreajusta el train (acc 1.0) y colapsa a `hold` en test.
- El léxico LM capta el *tono* (refleja el estado de la economía, no la decisión) → falla en test.
- Word2Vec es el primero en mejorar en test: la semántica vence al *drift* de vocabulario.
- FinBERT fine-tuned es el único balanceado en las 3 clases en test (recall de `cut` = 0.83);
  bajo *learning rate* + *class weights* + *early stopping* evitan el sobreajuste con sólo 145 docs.

## Estructura del repo

```
TP_NLP/
├── data/
│   ├── raw/{minutes,rates}/   # scraping FOMC + CSVs de FRED (contenido gitignored)
│   └── processed/             # dataset final etiquetado (versionado)
├── notebooks/
│   ├── 01_data_acquisition.ipynb   # scraping FOMC + FRED
│   ├── 02_preprocessing.ipynb      # limpieza + construcción de etiquetas
│   ├── 03_eda.ipynb                # análisis exploratorio
│   ├── 04_tfidf.ipynb              # Exp 1: TF-IDF + LogReg
│   ├── 05_lm_lexicon.ipynb         # Exp 2: léxico Loughran-McDonald + LogReg
│   ├── 06_word2vec.ipynb           # Exp 3: Word2Vec mean pooling + LogReg
│   ├── 07_finbert.ipynb            # Exp 4a (feature extraction) + 4b (fine-tuning)
│   └── 08_resultados.ipynb         # comparación de experimentos
├── reports/figures/                # figuras para informe / presentación
├── requirements.txt
└── CLAUDE.md
```

Estilo *notebooks-only* (sin paquete `src/`), siguiendo la convención de la cátedra:
todo el código vive en celdas de notebook.

## Reproducción

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name tp_nlp
```

Luego ejecutar los notebooks en orden (`01` → `08`). Notas:
- `data/raw/` está gitignored (las minutas son pesadas y re-derivables desde
  `01_data_acquisition.ipynb`); `data/processed/` sí está versionado, así que los
  notebooks de modelado (`04`–`08`) corren sin necesidad de re-scrapear.
- Word2Vec (Exp 3) requiere un venv con Python 3.12 (gensim no compila en 3.14).
- FinBERT (Exp 4) se corrió con GPU (CUDA / MPS); en CPU es lento pero funciona.
