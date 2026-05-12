# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Academic NLP project (ITBA, 2do cuatrimestre 2026). Group submission for the course's *Trabajo Práctico*. Three deliveries total; the first delivery (informe escrito) is already submitted.

**Goal:** predict FOMC monetary policy decisions — `hike` / `cut` / `hold` — from the text of FOMC meeting minutes (multiclass text classification).

**Corpus:**
- FOMC minutes (English) from federalreserve.gov — current calendar 2021–2027 + historical archive 1936–2020.
- Federal funds target rate from FRED (St. Louis Fed): series `DFEDTAR` (pre-2008), `DFEDTARU` / `DFEDTARL` (post-2008 range regime). Labels are derived from the rate change between consecutive meetings.
- **193 documents** (2000–2023), average ~8000 words each. Class distribution tracks known monetary regimes (`hold` dominates 2009–2015 and 2020–2021; `hike` in 2004–2006 and 2022–2023; `cut` in 2001, 2007–2008, 2019–2020).
- Known issue: stylistic drift around 2005 when FOMC began publishing more debate detail — controlling for epoch matters for generalization.
- 884 total minutes in `data/raw/minutes/` (1936–2023) but only 2000–2023 are used.

## Status of the work

**Current phase: preprocessing + EDA.** Minutes scraped (193 docs in `data/raw/minutes/`). FRED rates not yet fetched. Second delivery (sección 6 Experimentos) submitted 2026-05-11.

**Chronological split (fixed):**
- Train: 2000–2018 → 153 docs
- Val: 2019–2020 → 16 docs
- Test: 2021–2023 → 24 docs

**Experiment plan (locked for 2da entrega):**
1. TF-IDF + logistic regression (baseline interpretable)
2. Loughran-McDonald lexicon (6 tone features) + logistic regression
3. Word2Vec mean pooling (dim 300, GoogleNews-300) + logistic regression
4a. FinBERT feature extraction ([CLS] frozen, dim 768) + logistic regression
4b. FinBERT fine-tuning (AdamW lr=2e-5, early stopping on F1-macro, class weights)

**Next steps in order:**
1. `02_preprocessing.ipynb` — fetch FRED rates, align with minutes dates, build labels (hike/cut/hold), output `data/processed/fomc_dataset.csv`
2. `03_eda.ipynb` — class distribution, doc length, vocabulary per class, temporal drift, Loughran-McDonald descriptive analysis
3. Modelling notebooks (to be created: `04_tfidf.ipynb`, `05_lm_lexicon.ipynb`, `06_word2vec.ipynb`, `07_finbert.ipynb`)

**Delivery dates:**
- 2da entrega (sección 6): 2026-05-11 ✓
- Entrega final (presentación): 2026-06-22

## Repo layout

```
TP_NLP/
├── CLAUDE.md            # tracked — orienta a Claude para todo el grupo
├── context/             # gitignored — material académico local de cada integrante
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/{minutes,rates}/   # FOMC scraping output + FRED CSVs (gitignored content)
│   └── processed/             # final labelled dataset (tracked)
├── notebooks/
│   ├── 01_data_acquisition.ipynb   # scraping FOMC + FRED
│   ├── 02_preprocessing.ipynb      # cleaning + label construction
│   └── 03_eda.ipynb                # exploratory analysis
└── reports/figures/                # plots for informe / presentación
```

**Notebooks-only style** (no `src/` package). All code lives in `.ipynb` cells, matching cathedra convention. Scraping logic, preprocessing helpers, and modelling all go inside notebooks. If a function gets reused across 3+ notebooks, only then promote to a `src/` module.

**Dependency management:** `requirements.txt` + `venv` (estilo notebooks cátedra). Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name tp_nlp
```

**Data folder tracking:** `data/raw/` keeps its structure via `.gitkeep` but contents are gitignored (scraped minutes are bulky and re-derivable from `01_data_acquisition.ipynb`). `data/processed/` is tracked so the final labelled dataset is reproducible across team members without re-scraping.

The first delivery (`context/1er Entrega - NLP.md`) proposed three baselines:
1. TF-IDF + logistic regression (interpretable baseline).
2. Loughran-McDonald financial lexicon → tone scores → classifier.
3. Word2Vec (financial corpus) averaged embeddings + logistic regression.

**Important — direction change since the first delivery:** the planned modelling section was deliberately vague and did not include transformers. The current direction is to **add a transformer-based experiment** (BERT fine-tuning and/or feature extraction, encoder-only) on top of the three baselines above. Treat the first-delivery modelling proposal as a starting point only, not a constraint. The transformer work should follow the patterns shown in the cathedra notebooks (see below).

## Reference material (gitignored under `context/`)

The `context/` folder holds everything not for submission and is in `.gitignore`. Each team member populates their own `context/` locally with the shared course materials. `CLAUDE.md` itself **is tracked** so the whole group benefits from the same Claude orientation.

Suggested layout for each member's local `context/`:

- `context/1er Entrega - NLP.md` / `.pdf` — the submitted first-delivery report. Defines problem, data sources, EDA expectations, and the (now-superseded) baseline plan.
- `context/Consignas TP y Examen NLP (compartido).pdf` — official course rubric. Three deliveries (informe → experimentos → exposición final), notebook cuestionarios, and the final exam are graded separately.
- `context/Diapos/` — course slides (PDFs). Most relevant for this TP: `06-Transformers.pdf`, `07-BERT.pdf`, `08-TransferLearning.pdf`, `05-LMs.pdf`, `NLP - Clasificación con NNs.pdf`, `NLP - Clasificación y Evaluación.pdf`, `NLP - EDA + Clasificación.pdf`.
- `context/notebooks/` — consolidated cathedra notebooks (the canonical version). Use these as the **template for any modelling code we write**:
  - `01_EDA.ipynb`, `02_BoW_TfIdf_Logistic.ipynb`, `03_Word2vecClf.ipynb`, `04_Embeddings_PMI.ipynb` — baselines we mirror.
  - `06a_PytorchTutorial.ipynb`, `06b_HuggingFaceTutorial.ipynb` — framework setup.
  - `08a_BERTEmbeddings.ipynb`, `08b_BERTClfFineTuning.ipynb`, `08c_BERTClfFeatureExtraction.ipynb` — **directly applicable** to the transformer extension.
  - `07_CausalLMFinetuning.ipynb`, `09_TopicModeling.ipynb`, `10_LLMsAPIs.ipynb`, `11_MTEncoderDecoder.ipynb`, `12-RAG.ipynb` — likely out of scope but available.
- `context/materia_extras/` — older, more granular topic folders from the cathedra repo (`02_básicas`, `03_redes`, `05_embeddings`, `06_clasificación`, `07_TopicModelling`, `08_LanguageModels`, `09_Transformers`). Useful for narrower examples (BPE, padding, individual embedding methods, Naive Bayes, BERT variants, RAG, etc.) when the consolidated notebook is too dense.

When writing modelling code: **first** look at the matching consolidated notebook in `context/notebooks/` and follow its conventions (imports, tokenization, train/val/test split style, evaluation reporting). Fall back to `context/materia_extras/` for finer-grained references.

## Working conventions for this repo

- Language: Spanish for prose (informe, comments aimed at the team), English for code identifiers and docstrings.
- The dataset is 193 docs (153 train / 16 val / 24 test). Model complexity must stay justifiable — do not introduce architectures the data cannot support without explicit discussion.
- Time-aware splitting matters because of the linguistic drift around 2005 and the regime structure of the labels. Random splits will leak; chronological or grouped splits are the default.
- Stopwords plus domain-generic terms (`committee`, `meeting`, `board`, `federal`, `reserve`) are removed before TF-IDF / lexical analysis.
- Metric reporting should match what the cathedra notebooks use (typically accuracy + macro-F1 + per-class precision/recall/F1 and a confusion matrix); follow the notebook style rather than inventing new evaluation conventions.

## Notes for future Claude sessions

- Anything under `context/` is for the assistant's understanding only. Never reference `context/` paths in code, comments, commits, or anything that ends up in the submission.
- The user is the student; the assistant should ask before introducing tooling or libraries beyond what the cathedra notebooks demonstrate, since the grading rubric values "relevancia con respecto a la materia."
- There is no build/test/lint setup yet. When that is added, update this file.
