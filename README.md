# Head & Shoulders Consumer Switching Signal Detection Pipeline

A 3-layer Korean NLP × Trend × ML pipeline that detects and quantifies consumer switching signals for **Head & Shoulders (P&G Korea)** in the shampoo category. The pipeline integrates unstructured VoC data (Layer 1), structured search trend data (Layer 2), and a segment-level switching probability model (Layer 3) to answer a single strategic question:

> **"As consumers shift their scalp problem-solving frame from 'shampoo brands' toward 'derma/clinic solutions', can Head & Shoulders defend its position through Charcoal and mild-acid line extensions — or is the category itself being redefined?"**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
  - [Layer 1 — VoC Pipeline](#layer-1--voc-pipeline)
  - [Layer 2 — Trend Pipeline](#layer-2--trend-pipeline)
  - [Layer 3 — Switching Pipeline](#layer-3--switching-pipeline)
- [Methodological Decisions and Pivots](#methodological-decisions-and-pivots)
- [Key Findings](#key-findings)
- [Cross-Layer Analysis](#cross-layer-analysis)
- [Dashboard](#dashboard)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This pipeline was built as a portfolio project targeting the **P&G Korea S&M (Sales & Marketing)** internship role. Rather than producing a surface-level brand analysis, the project attempts to identify *causal* signals in consumer language and search behavior that explain *why* and *how fast* consumers are switching away from Head & Shoulders — and which specific consumer segments are at the highest structural risk.

The analysis was designed to answer:
- What language signals in Korean VoC indicate active switching vs. latent risk vs. satisfaction?
- Does Naver search trend data independently confirm the VoC signals, and at what velocity?
- Which consumer segments have the highest switching probability, and what intervention is appropriate for each?
- What does the cross-layer picture say about the *frame* shift in how consumers approach scalp problems — not just which brand they prefer?

---

## Business Context

The Korean shampoo market is undergoing a structural repositioning. Between 2020 and 2026, the following trends converged:

- **닥터그루트** (Doctor Groot), once the dominant functional shampoo brand, collapsed -87.8% in search volume from its 2020 peak
- **헤드앤숄더클리니컬스트렝스** (Head & Shoulders Clinical Strength), HNS's premium functional line, effectively disappeared from search by 2026 (35.9 avg in 2021 → 0.0 in 2026)
- **안티트로** (Antitro), a derma-channel brand by Curev, entered the market in December 2024 and *reversed* Head & Shoulders Core search volume within 6 months (by June 2025)
- The overall shampoo category rebounded +14.6% in 2026 — not from traditional brands, but driven by Antitro's rapid ascent

This is not a brand preference shift. It is a **category frame shift**: consumers who previously searched "비듬샴푸" (dandruff shampoo) are now searching "안티트로샴푸" — a brand-specific term that has become a category surrogate. The pipeline was designed to measure this shift in consumer language, quantify it in search data, and translate it into segment-level switching probabilities.

---

## Dataset

### Layer 1 — VoC Data

| Source | Collection Method | Records | Notes |
|--------|-------------------|---------|-------|
| Naver Blog | Naver Search API (official) | 653 | Long-form usage reviews, ingredient analysis |
| Naver Cafe | Naver Search API (official) | 675 | Community Q&A, comparison discussions |
| YouTube Comments | YouTube Data API v3 (official) | 1,386 | Purchase motivation, post-purchase reaction |

- **Total collected**: 2,714 documents
- **After HNS relevance filter**: 1,744 documents
- **Relevance filter keywords**: 비듬, 두피, 각질, 가려움, 지루성, 설페이트, 클리니컬, 프로페셔널, 차콜, 약산성, 안티트로, 두피염, 정수리
- **Collection note**: All data collected via official APIs only. No scraping of systems that prohibit automated access. Raw data not redistributed.

### Layer 2 — Naver DataLab Trend Data

| Source | Coverage | Features |
|--------|----------|----------|
| 분야통계_샴푸 (Category click volume) | 2020-01 ~ 2026-04 | 1 column |
| 쇼핑인사이트_헤드앤숄더 (Brand keyword search) | 2020-01 ~ 2026-04 | 5 columns |
| 쇼핑인사이트_증상카테고리 (Symptom keyword search) | 2020-01 ~ 2026-04 | 5 columns |
| 검색어트렌드_브랜드경쟁구도 (Brand competition) | 2020-01 ~ 2026-03 | 5 columns |

- **Total**: 76 months × 16 features

---

## Project Structure

```
pg-hns-consumer-signal-pipeline/
│
├── voc_pipeline/                        # Layer 1: Korean NLP VoC pipeline
│   ├── collector_naver.py               # Naver Blog/Cafe collection via Search API
│   ├── collector_youtube.py             # YouTube comment collection via Data API v3
│   ├── preprocessor.py                  # kiwipiepy morphological analysis + 4 modes
│   │                                    # user word: 안티트로 (NNP)
│   ├── LDA_pipeline.py                  # LDA topic modeling: per-source × per-mode
│   ├── causal_signal_detector.py        # Causal signal scoring + temporal analysis
│   └── data/
│       ├── raw/                         # Collected CSV files (gitignored)
│       └── processed/                   # Analysis outputs (gitignored)
│
├── trend_pipeline/                      # Layer 2: Trend & forecasting pipeline
│   ├── trend_loader.py                  # Multi-source DataLab CSV → unified feature table
│   ├── trend_analyzer.py                # Chronos forecast (Part A) + structural analysis (Part B)
│   └── data/
│       ├── raw/                         # Naver DataLab CSV + XLSX files (gitignored)
│       └── processed/                   # trend_features.csv, chronos_forecast.csv (gitignored)
│
├── switching_pipeline/                  # Layer 3: Consumer switching probability
│   ├── feature_builder.py               # Layer 1 + Layer 2 → unified feature table + segment labels
│   ├── segment_classifier_regression.py # Logistic regression approach (adopted)
│   ├── segment_classifier_clustering.py # KMeans clustering approach (tested, not adopted)
│   ├── switching_probability.py         # Brand risk score + intervention plan
│   └── data/                            # Output CSVs (gitignored)
│
├── notebooks/
│   └── hns_bertopic.ipynb               # BERTopic pipeline (Google Colab, T4 GPU)
│
├── dashboard.py                         # Streamlit 3-layer integrated dashboard
├── .env.example                         # API key template
├── .gitignore
└── requirements.txt
```

---

## Pipeline

```
Layer 1 — VoC Pipeline
─────────────────────────────────────────────────────────────────
Raw Data Collection
(Naver Blog/Cafe via Search API + YouTube via Data API v3)
        │
        ▼
collector_naver.py / collector_youtube.py
(2,714 docs total)
        │
        ▼
preprocessor.py
(kiwipiepy + 안티트로 user word + 4 preprocessing modes)
        │
        ▼
LDA_pipeline.py                    hns_bertopic.ipynb (Colab)
(per-source × per-mode)            (BERT + HDBSCAN, 17 topics)
        │                                      │
        ▼                                      ▼
hns_lda_results.csv         hns_bertopic_results.csv
        │                                      │
        └──────────────┬────────────────────────┘
                       ▼
            hns_lda_bertopic_consensus.csv
            (12 High Confidence signals)
                       │
                       ▼
        causal_signal_detector.py
        (churn risk 24.0% | competitor mention 101 docs)
                       │
                       ▼
        hns_causal_signals.csv / hns_temporal_signals.csv


Layer 2 — Trend Pipeline
─────────────────────────────────────────────────────────────────
Naver DataLab CSV / XLSX (22 files)
        │
        ▼
trend_loader.py
(76 months × 16 features → trend_features.csv)
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
Part A: Chronos Forecast                Part B: Structural Analysis
(Amazon Chronos zero-shot)              (antitro reversal, HNS lifecycle,
(12-month HNS forecast)                  competitor trajectory, category shift)
        │                                        │
        ▼                                        ▼
chronos_forecast.csv              trend_analysis_summary.csv


Layer 3 — Switching Pipeline
─────────────────────────────────────────────────────────────────
hns_causal_signals.csv + trend_features.csv
        │
        ▼
feature_builder.py
(VoC features + segment labels: Active Switcher / At-risk / Passive / Loyal)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
segment_classifier_regression.py  segment_classifier_clustering.py
(Logistic Regression, adopted)     (KMeans, tested — At-risk segment
                                    lost in clustering, not adopted)
        │
        ▼
switching_probability.py
(brand risk score + intervention plan)
        │
        ▼
switching_implications.csv / timeline_analysis.csv


All layers → dashboard.py (Streamlit 3-layer integrated dashboard)
```

---

## Methodology

### Layer 1 — VoC Pipeline

#### Preprocessing

Four preprocessing strategies were applied, enabling direct comparison of how token representation affects topic coherence:

| Mode | Description |
|------|-------------|
| `unigram` | Single noun tokens; stopwords + domain noise removed |
| `bigram` | Consecutive noun pairs (e.g. `클리니컬_스트렝스`, `정수리_냄새`) |
| `unibi_mix` | Union of unigram and bigram tokens |
| `adj_noun` | Adjective-noun pairs for sentiment-bearing phrases |

All modes use `kiwipiepy` for Korean morphological analysis. `안티트로` was registered as a user word (`NNP`) to prevent morpheme segmentation — confirmed necessary after the first-run LDA output showed `안티트` (truncated form) appearing in Topic 6.

**STOPWORDS design**: Finalized *after* examining raw collected data rather than before. Key decisions:
- `두피` (scalp) removed as standalone token — appeared in 100%+ of topics at 0.1+ weight, eliminating discriminative value. Compound forms survive through bigram extraction.
- Shopping platform noise (`최저가`, `적립`, `옵션`) added after first LDA run confirmed contamination.

#### LDA Topic Modeling

Gensim's `LdaModel` applied per combination of source × mode. Optimal k selected by maximizing c_v coherence across k = 2–7. `adj_noun` returned Insufficient data across all sources — Korean shampoo VoC is dominated by noun+verb structures rather than adjective-noun compounds.

**Coherence summary:**

| Scope | bigram | unibi_mix | unigram |
|-------|--------|-----------|---------|
| all | 0.5980 | 0.3524 | 0.3469 |
| blog | **0.6177** | 0.3136 | 0.3140 |
| cafearticle | 0.5951 | 0.3380 | 0.3414 |
| youtube | 0.6103 | 0.3865 | 0.3650 |

#### Causal Signal Detection

Keyword-based scoring across 6 churn signal categories and 3 positive signal categories. Documents flagged for competitor mention when `안티트로` or `니조랄` appear in text.

#### BERTopic

`paraphrase-multilingual-MiniLM-L12-v2` embeddings + HDBSCAN on Google Colab (T4 GPU). Initial `min_topic_size=15` produced 3 topics with Topic 0 absorbing 91% of documents. Reduced to `min_topic_size=8` to achieve 17 meaningful topics.

---

### Layer 2 — Trend Pipeline

#### trend_loader.py

Merges 22 Naver DataLab files into a unified monthly feature table (76 rows × 16 columns, 2020-01 to 2026-04).

#### trend_analyzer.py — Part A: Chronos Forecast

**Amazon Chronos** (2024, zero-shot time series foundation model) used for 12-month HNS Core forecast. Selected over ARIMA/Prophet (outdated methodology) and deep learning models (TFT, PatchTST) because with only 76 monthly observations, any fine-tuning-dependent model faces fundamental data constraint issues. Chronos bypasses this via zero-shot inference.

Forecast: HNS Core projected at 47–54 (median) for May 2026 – April 2027, 80% CI of 34–63. Flat trajectory — no sharp recovery, no further collapse.

#### trend_analyzer.py — Part B: Structural Analysis

ML classification for Part B was attempted and abandoned:

- **FT-Transformer**: All 12 risk-label months fall in the time series tail (2025-05 onward), leaving zero risk samples in training set under time-ordered splitting. Not a model performance issue — a structural property of the data.
- **ROCKET (sktime)**: Same structural limitation.
- **Change point detection**: Trivial — the change point (Antitro launch, 2024-12) is already known from the data.

The data's message is clear without a classifier. Structural descriptive analysis was adopted: antitro reversal timing, HNS line lifecycle, competitor trajectory, category dynamics, and symptom keyword language shift.

---

### Layer 3 — Switching Pipeline

#### Segment Definition

| Segment | Definition | n | % |
|---------|------------|---|---|
| Active Switcher | comparison_frame + competitor_mention + churn_signal | 83 | 4.8% |
| At-risk | (medical_frame OR ingredient_frame) + churn_signal, no direct comparison | 91 | 5.2% |
| Passive User | neutral signal, no strong frame | 1,224 | 70.2% |
| Loyal | positive_signal, no churn indicators | 346 | 19.8% |

#### Classifier Comparison

**KMeans Clustering**: Optimal k=7 (silhouette=0.3988). At-risk segment entirely absorbed into Passive User clusters — KMeans failed to distinguish the medical/ingredient frame signal. Not adopted.

**Logistic Regression**: Regularization grid search (C ∈ {0.01…10.0}), optimal C=1.0, CV accuracy 0.9300 ± 0.0135. Features exclude rule-defining columns to avoid data leakage. Adopted.

**Key coefficients (C=1.0):**

| Segment | Strongest positive signals |
|---------|---------------------------|
| Active Switcher | `churn_score` (+2.364), `is_blog` (+0.146) |
| At-risk | `is_ingredient_frame` (+0.938), `is_medical_frame` (+0.737) |
| Loyal | `positive_score` (+4.466), `churn_score` (-5.009) |

#### Switching Probability

```
trend_multiplier = 1.0 + w_antitro × antitro_pressure + w_hns × hns_decline
```

Weights optimized via grid search (maximize Active Switcher vs. Loyal probability gap): **w_antitro=0.50, w_hns=0.40** (multiplier=1.427).

| Segment | n | P(switch) |
|---------|---|-----------|
| Active Switcher | 83 | 0.405 |
| At-risk | 91 | 0.352 |
| Passive User | 1,224 | 0.043 |
| Loyal | 346 | 0.000 |

**Brand Risk Score: 0.068** — weighted average by segment size. The low overall score reflects that 90% of documents are Passive/Loyal. The signal is the *concentration* of switching probability in the 10% At-risk + Active Switcher segments.

---

## Methodological Decisions and Pivots

| Decision Point | Attempted | Outcome | Final Approach |
|----------------|-----------|---------|----------------|
| Layer 2 Part B | FT-Transformer classification | Train set: 0 risk samples | Abandoned |
| Layer 2 Part B | ROCKET classifier (sktime) | Same structural limitation | Abandoned |
| Layer 2 Part B | Rule-based change point detection | Trivial (launch date already known) | Abandoned |
| Layer 2 Part B | Structural descriptive analysis | Data message clear without classifier | Adopted |
| Layer 3 | KMeans clustering | At-risk segment lost in clustering | Not adopted |
| Layer 3 | Logistic Regression | At-risk preserved, CV 0.9300 | Adopted |
| BERTopic min_topic_size | 15 | 3 topics, 91% in Topic 0 | Reduced to 8 |
| Trend multiplier weights | Arbitrary (0.3, 0.2) | No data-driven basis | Grid search → (0.50, 0.40) |
| STOPWORDS | Pre-defined before data inspection | Risk of over/under-filtering | Finalized after examining raw data |

---

## Key Findings

### 1. Antitro Reversed Head & Shoulders Core in 6 Months

Antitro entered the Korean shampoo market in December 2024 with zero search volume. By June 2025 its Naver Shopping search volume exceeded Head & Shoulders Core — a reversal achieved in 6 months. By April 2026 the Antitro/HNS ratio stands at 1.327, with an acceleration peak of +40.2 in a single month (January 2026).

> **Business implication**: The critical intervention window is before the Antitro/HNS ratio exceeds 1.5. At current trajectory this threshold may be crossed within months. Beyond 1.5, Antitro risks becoming the default consumer reference point for scalp care — a position Head & Shoulders has held for decades.

---

### 2. The Category Is Being Redefined, Not Just Competed Against

Shampoo category click volume declined -20.6% in 2024 and -22.2% in 2025 — but rebounded +14.6% in 2026. This rebound is not from traditional brands recovering; it is driven by Antitro expanding the category by attracting consumers who now search for scalp solutions under a new brand-as-category term.

Symptom keyword language shift:
- 비듬샴푸 (Dandruff Shampoo): -21.3% from 2020 baseline
- 지루성두피샴푸 (Seborrheic Shampoo): -57.6%
- 안티트로샴푸 (Antitro Shampoo): 0 → 75.1 (now exceeds Dandruff Shampoo)

> **Business implication**: Competing on shampoo attributes (fragrance, foam, volume) does not address this frame shift. The consumer is no longer asking "which shampoo is best for dandruff?" — they are asking "is this a medical-grade scalp solution?"

---

### 3. HNS Product Line Lifecycle — Clinical Strength Collapse

| Line | 2021 avg | 2026 avg | Change |
|------|----------|----------|--------|
| Core | 46.0 | 63.3 | +37.6% |
| Clinical Strength | 35.9 | 0.0 | -100% |
| Professional | 0.0 | 3.4 | new |
| Charcoal | 0.0 | 0.9 | new |

Clinical Strength — HNS's premium functional line that most directly competed in the derma/medicated segment — has disappeared from consumer search. The new Charcoal (0.9) and Professional (3.4) lines show minimal consumer uptake vs. Antitro (75.1).

> **Business implication**: The line extension strategy has not generated meaningful demand in the segments being lost to Antitro. A repositioning strategy focused on clinical/dermatological credibility is indicated.

---

### 4. Churn Risk Is Concentrated in a Specific Signal Profile

Top churn keywords from 1,744 HNS-relevant documents:

```
가려움 (itchiness):        291  ← efficacy failure
자극 (irritation):          90  ← skin reaction
뾰루지 (pimples):           63  ← skin reaction
안티트로 (Antitro):          56  ← competitor switch
대신 (instead of):          52  ← replacement signal
니조랄 (Nizoral):            45  ← derma alternative
```

Itchiness (291 mentions) is 5× more frequent than any other churn signal. Consumers are not leaving because they dislike Head & Shoulders' brand image — they are leaving because it stopped solving their problem.

> **Business implication**: The primary churn driver is unresolved symptom efficacy, not brand equity. Marketing interventions alone will not address this. Product-level intervention — reinforcing the clinical efficacy of zinc pyrithione — is the appropriate response.

---

### 5. At-risk Segment: Pre-switch Consumers Identified by Frame, Not Action

The At-risk segment (91 documents, 5.2%) is defined by ingredient/medical frame adoption *without* direct competitor comparison — consumers who have started researching scalp problems through a clinical lens but have not yet identified Antitro as their alternative.

| Feature | At-risk | Passive User |
|---------|---------|--------------|
| is_medical_frame | 41.8% | 9.2% |
| is_ingredient_frame | 60.4% | 11.6% |
| is_competitor | 0.0% | 1.3% |
| P(switch) | 0.352 | 0.043 |

> **Business implication**: At-risk consumers are accessible before they become Active Switchers. Proactive content addressing ingredient transparency — explaining zinc pyrithione's antifungal mechanism — can intercept this segment before they discover Antitro's clinical positioning.

---

## Cross-Layer Analysis

### Finding A: The Frame Shift Is Channel-Stratified

Ingredient scrutiny (설페이트, 소듐라우레스설페이트, 계면활성제) appears exclusively in YouTube comments — blog and cafe show zero documents in this cluster (BERTopic Topics 14, 16). Medical frame language (피부과, 항진균, 질환) distributes uniformly across all channels.

YouTube users are at the leading edge of the ingredient/clinical frame shift. Blog and cafe consumers remain in result-focused exploration. The medical frame requires no ingredient literacy and is already mainstream across all channels.

**Cross-layer interpretation**: Layer 2 shows Antitro search growth accelerating from January 2026. Layer 1 shows ingredient scrutiny concentrated on YouTube. The likely sequence: YouTube-first ingredient scrutiny → Antitro discovery → search volume surge. Intercepting YouTube-channel consumers with ingredient transparency content is a higher-leverage intervention than broad channel campaigns.

---

### Finding B: Competitive Comparison Happens on YouTube, Not Blogs

BERTopic Topic 1 (303 documents, largest competitive cluster containing both 헤드엔숄더 and 안티트로): YouTube 237 documents, Cafe 48, Blog 18. Competitive head-to-head comparison is a YouTube-native phenomenon.

Blog churn rate (34.1%) is higher than YouTube (17.3%) — but blog churn documents reflect post-switch dissatisfaction expressed in long-form posts, not active comparison decisions. YouTube is where the comparison decision is being made.

**Cross-layer interpretation**: Layer 3 shows Active Switchers concentrated in blogs (47 vs. YouTube 19) — but this reflects post-switch documentation. The decision point is YouTube.

---

### Finding C: The Category Rebound Does Not Benefit HNS

Layer 2 shows category click volume +14.6% in 2026. A surface reading might suggest the category is recovering and HNS benefits. Cross-referencing with symptom keyword data: the rebound is entirely driven by 안티트로샴푸 (0 → 75.1), while 비듬샴푸 (-21.3%) and 지루성두피샴푸 (-57.6%) continue declining.

Head & Shoulders is losing search share in a category that is growing — Antitro is expanding the addressable market while simultaneously taking HNS's existing share. This is structurally more threatening than a declining category, because the growth signal masks the displacement.

---

## Dashboard

An interactive Streamlit dashboard visualizes all three pipeline layers:

```bash
streamlit run dashboard.py
```

**Tabs**: Overview | VoC Analysis | Trend Analysis | Switching Risk | BERTopic

---

## How to Run

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with Naver Search API and YouTube Data API v3 keys

# 3. Layer 1: Collect VoC data
python voc_pipeline/collector_naver.py
python voc_pipeline/collector_youtube.py

# 4. Layer 1: Preprocess
python voc_pipeline/preprocessor.py

# 5. Layer 1: LDA pipeline
python voc_pipeline/LDA_pipeline.py

# 6. Layer 1: Causal signal detection
python voc_pipeline/causal_signal_detector.py

# 7. Layer 1: BERTopic (Google Colab recommended)
# Upload voc_pipeline/data/processed/hns_processed.csv to Google Drive
# Run notebooks/hns_bertopic.ipynb on Google Colab (T4 GPU)
# Download 4 output files to voc_pipeline/data/processed/

# 8. Layer 2: Load and process trend data
python trend_pipeline/trend_loader.py
python trend_pipeline/trend_analyzer.py

# 9. Layer 3: Build features and run switching model
python switching_pipeline/feature_builder.py
python switching_pipeline/segment_classifier_regression.py
python switching_pipeline/switching_probability.py

# 10. Launch dashboard
streamlit run dashboard.py
```

---

## Dependencies

```
# Core NLP
kiwipiepy==0.23.1
gensim==4.4.0

# ML / modeling
scikit-learn
torch
chronos-forecasting
sktime

# Data
pandas
numpy
openpyxl

# Visualization / dashboard
streamlit
plotly

# Collection
requests
python-dotenv
google-api-python-client

# BERTopic (Colab environment)
bertopic
sentence-transformers
```

Install all dependencies:
```bash
pip install -r requirements.txt
```
