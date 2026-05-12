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

**Current phase: modelling.** Preprocessing and EDA are complete. `data/processed/fomc_dataset.csv` (193 docs) and `reports/figures/` (8 figures) are committed. Second delivery (sección 6 Experimentos) submitted 2026-05-11.

**Chronological split (fixed):**
- Train: 2000–2018 → 153 docs
- Val: 2019–2020 → 16 docs
- Test: 2021–2023 → 24 docs

**Key EDA findings (locked):**
- `hold` = 131 (67.88%), `hike` = 39, `cut` = 23 → F1-macro is the primary metric
- 100% of docs > 512 words → FinBERT requires explicit truncation strategy
- Jaccard drift pre/post-2005 = 0.538 → chronological split is mandatory
- LM Negative score: `cut`=0.0458 vs `hike`=0.0263 (74% higher) → Exp 2 has real signal

**Experiment plan (locked for 2da entrega):**
1. **TF-IDF + logistic regression** — `ngram_range=(1,2)`, `max_features=10000`, `sublinear_tf=True`; LogReg L2, C ∈ {0.01, 0.1, 1, 10} grid search on val.
2. **Loughran-McDonald lexicon + logistic regression** — 6 tone scores (Positive, Negative, Uncertainty, Litigious, Strong Modal, Weak Modal) as 6D features; same LogReg setup.
3. **Word2Vec mean pooling + logistic regression** — GoogleNews-300 pretrained vectors, mean pooling ignoring OOV tokens, dim=300; LogReg C tuned on val.
4a. **FinBERT feature extraction** — `ProsusAI/finbert`, [CLS] frozen, dim=768 → LogReg.
4b. **FinBERT fine-tuning** — `ProsusAI/finbert` + linear classification head; lr ∈ {1e-5, 2e-5, 5e-5}, batch ∈ {4, 8, 16}, max 5 epochs, early stopping on F1-macro, class weights inversely proportional to frequency.

**FinBERT truncation strategies** (evaluated empirically on val, choose best):
- *Head+tail*: first 256 + last 256 tokens per doc.
- *Chunking with mean pooling*: split into 512-token chunks (optional overlap), average [CLS] embeddings.

**Next steps in order:**
1. `04_tfidf.ipynb` — Exp 1: TF-IDF + LogReg
2. `05_lm_lexicon.ipynb` — Exp 2: LM scores + LogReg
3. `06_word2vec.ipynb` — Exp 3: Word2Vec mean pooling + LogReg
4. `07_finbert.ipynb` — Exp 4a + 4b: FinBERT feature extraction + fine-tuning

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

The combined first+second delivery (`context/Entrega_1_2_NLP .md` / `.pdf`) defines the full experiment plan including the FinBERT extension. Treat this document as the authoritative spec for all modelling decisions.

## Reference material (gitignored under `context/`)

The `context/` folder holds everything not for submission and is in `.gitignore`. Each team member populates their own `context/` locally with the shared course materials. `CLAUDE.md` itself **is tracked** so the whole group benefits from the same Claude orientation.

Suggested layout for each member's local `context/`:

- `context/Entrega_1_2_NLP .md` / `.pdf` — combined first + second delivery report. Defines problem, data sources, EDA results, and the full locked experiment plan (Exp 1–4b).
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
