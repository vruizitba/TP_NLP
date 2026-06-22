# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Academic NLP project (ITBA, 2do cuatrimestre 2026). Group submission for the course's *Trabajo Práctico*. Three deliveries total; the first delivery (informe escrito) is already submitted.

**Goal:** predict FOMC monetary policy decisions — `hike` / `cut` / `hold` — from the text of FOMC meeting minutes (multiclass text classification).

**Corpus:**
- FOMC minutes (English) from federalreserve.gov — current calendar 2021–2027 + historical archive 1936–2020.
- Federal funds target rate from FRED (St. Louis Fed): series `DFEDTAR` (pre-2008), `DFEDTARU` / `DFEDTARL` (post-2008 range regime). Labels are derived from the rate change **before vs after each meeting** (not between consecutive meetings — that lagged the label by one meeting; fixed in `02_preprocessing.ipynb`).
- **209 documents** (2000–2025), average ~8000 words each. Class distribution tracks known monetary regimes (`hold` dominates 2009–2015 and 2020–2021; `hike` in 2004–2006 and 2022–2023; `cut` in 2001, 2007–2008, 2019–2020, 2024–2025).
- Known issue: stylistic drift around 2005 when FOMC began publishing more debate detail — controlling for epoch matters for generalization.
- 884 total minutes in `data/raw/minutes/` (1936–2026) but only 2000–2025 are used.

## Status of the work

**Current phase: modelling.** Preprocessing and EDA complete. Corpus extended to 2000–2025 and labels corrected (see Corpus note). `data/processed/fomc_dataset.csv` (209 docs) and `reports/figures/` (8 figures) committed. Exp 1 (TF-IDF) and Exp 2 (LM lexicon) done; Exp 3–4 pending. Second delivery (sección 6 Experimentos) submitted 2026-05-11; final presentation 2026-06-22.

**Chronological split (fixed):** cuts chosen so all three classes appear in all three sets.
- Train: 2000–2017 → 145 docs
- Val: 2018–2021 → 32 docs
- Test: 2022–2025 → 32 docs

**Key EDA findings:**
- `hold` = 141 (67.5%), `hike` = 40, `cut` = 28 → F1-macro is the primary metric
- ~100% of docs > 512 words → FinBERT requires explicit truncation strategy
- Jaccard drift pre/post-2005 = 0.533 → chronological split is mandatory
- LM Negative score: `cut`=0.0407 vs `hike`=0.0250 (≈62% higher) → Exp 2 has real signal
- LM lexicon membership filter is `> 0` (negative year = word removed from category), not `!= 0`

**Anti-leakage masking (notebook 02):** the minutes state the decision explicitly ("the Committee
decided to raise/lower/maintain... to X percent"). For the X→X task that is leakage, so `mask_decision`
removes those phrases + decision rate values (~0.7% of text; explicit tell removed in 100% of docs;
not 100% leakage-free due to garbled 2-column PDF). 02 saves TWO datasets: `fomc_dataset.csv` (masked,
used by exp 1-4/notebooks 04-08) and `fomc_dataset_unmasked.csv` (used by forecasting notebook 09).
FinBERT (07) reads RAW minutes and masks in-flight (same function), since it needs natural text.

**Results — main TP, X→X on MASKED dataset (F1-macro):**
- Baseline (majority): val 0.286, test 0.213
- Exp 1 TF-IDF (C=10): val 0.519, test 0.213 (collapses to `hold` on test; drift)
- Exp 2 LM lexicon (C=0.01): val 0.524, test 0.154 (tone = economy state, not decision → fails on test)
- Exp 3 Word2Vec (C=10): val 0.691, test 0.396 — first to generalize (detects hikes); misses cuts. Py3.12 venv (gensim).
- Exp 4a FinBERT frozen [CLS] + LogReg (head+tail, C=0.01): val 0.610, test 0.484. RAW minutes text. Mac MPS / GPU.
- Exp 4b FinBERT fine-tuning (lr=1e-5, batch=8): val 0.845, test **0.741** — BEST. Only model balanced on 3 classes in test (cut recall 0.83, hike 0.55, hold 0.87). User's RTX 4070 (CUDA).

**Masking did NOT degrade FinBERT — 4b IMPROVED (0.648 leaky → 0.741 masked).** Proves its result was
NOT leakage: it infers from the discussion, doesn't read the stated answer. Raw FinBERT results in
`finbert_resultados.md` (repo root, gitignored).

**Final ranking (test F1-macro): FinBERT FT 0.741 > FinBERT frozen 0.484 > Word2Vec 0.396 > TF-IDF 0.213 ≈ baseline > LM 0.154.** Story: lexical → tone → semantics → context+adaptation; each step fixes the previous one's limit.

**Forecasting extension (notebook 09, X→X+1, UNMASKED dataset):** predict the NEXT meeting's decision.
Persistence baseline (repeat last decision) = test 0.768; NO model beats it (best FinBERT 4b 0.475;
classics 0.17–0.30). Monetary policy is highly autocorrelated → text doesn't beat inertia. This
justifies framing the TP as X→X. FinBERT > classics in forecasting too (some forward-guidance signal),
but not enough. Figure `reports/figures/09_forecasting_f1.png`. Kept as a presentation slide.

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
1. ✓ `04_tfidf.ipynb` — Exp 1: TF-IDF + LogReg (done)
2. ✓ `05_lm_lexicon.ipynb` — Exp 2: LM scores + LogReg (done)
3. ✓ `06_word2vec.ipynb` — Exp 3: Word2Vec mean pooling + LogReg (done; run in py3.12 venv)
4. ✓ `07_finbert.ipynb` — Exp 4a (Mac/MPS) + 4b (user's GPU). Bring executed notebook + 07a/07b figures from PC into repo.
5. `08_resultados.ipynb` — comparison table across experiments (next)
6. Final slides (Objetivos · Metodología · Resultados · Conclusiones · Limitaciones · Anexo)

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
