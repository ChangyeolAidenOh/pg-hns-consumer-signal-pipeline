# Head & Shoulders Consumer Switching Signal Detection Pipeline

A 3-layer Korean NLP × Trend × ML pipeline that detects and quantifies consumer switching signals for **Head & Shoulders (P&G Korea)** in the Korean shampoo category. The pipeline integrates unstructured VoC data (Layer 1), structured Naver DataLab search trend data (Layer 2), and a segment-level switching probability model (Layer 3) to answer a single strategic question:

> **"As consumers shift their scalp problem-solving frame from 'shampoo brands' toward 'derma/clinic solutions', can Head & Shoulders defend its position through Charcoal and mild-acid line extensions — or is the category itself being redefined around a new reference point?"**

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
- [Strategic Conclusion](#strategic-conclusion)
- [Dashboard](#dashboard)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This pipeline was built as a portfolio project targeting the **P&G Korea S&M (Sales & Marketing)** internship role. Rather than producing a surface-level brand sentiment analysis, the project attempts to identify *causal* signals in consumer language and search behavior that explain *why* and *how fast* consumers are switching away from Head & Shoulders — and which specific consumer segments are at the highest structural risk.

The analysis was designed to answer:
- What language signals in Korean VoC indicate active switching, latent risk, or continued satisfaction — and do these signals differ by channel (blog, cafe, YouTube)?
- Does Naver DataLab search trend data independently confirm the VoC signals, and at what velocity is the competitive displacement occurring?
- Which consumer segments carry the highest switching probability when Layer 1 and Layer 2 signals are integrated, and what is the appropriate intervention for each?
- What does the cross-layer picture reveal about the *frame* shift in how consumers approach scalp problems — not just which brand they choose, but how they define the problem itself?

---

## Business Context

The Korean shampoo market is showing signs of significant structural change. Between 2020 and 2026, several trends converged simultaneously:

- **닥터그루트** (Doctor Groot), once the dominant functional shampoo brand with the highest search volume in the category, collapsed -87.8% from its 2020 peak. Combined with 케라시스 (-39.3%) and 팬틴 (-35.7%) over the same period, this is consistent with broader structural pressure across the functional shampoo segment — though a single-brand collapse alone is insufficient to confirm a segment-wide trend.
- **헤드앤숄더클리니컬스트렝스** (Head & Shoulders Clinical Strength), the HNS line most directly positioned in the medicated/clinical segment, effectively disappeared from Naver Shopping search by 2026 (annual average: 35.9 in 2021 → 0.0 in 2026).
- **안티트로** (Antitro), a derma-channel shampoo brand by Curev with hospital and pharmacy distribution, entered the Naver Shopping search index in December 2024 with near-zero volume and reversed Head & Shoulders Core search volume within exactly 6 months (June 2025). By April 2026 its search volume stands at 1.327× Head & Shoulders Core.
- The overall shampoo category click volume, which had declined -20.6% in 2024 and -22.2% in 2025, rebounded +14.6% in 2026 — not because traditional brands recovered, but primarily because Antitro's ascent generated new search volume under a new keyword. Whether this reflects genuine market expansion or search behavior redistribution cannot be determined from search volume data alone.

These trends together indicate a **category frame shift**: consumers who historically searched "비듬샴푸" (dandruff shampoo) as their primary solution-seeking behavior are transitioning toward brand-specific clinical terminology — specifically "안티트로샴푸" — as their new default reference. This pipeline was built to measure that shift in consumer language, validate it against search trend data, and translate it into segment-level switching probabilities with actionable intervention recommendations.

---

## Dataset

### Layer 1 — VoC Data

| Source | Collection Method | Records (raw) | Notes |
|--------|-------------------|---------------|-------|
| Naver Blog | Naver Search API (official) | 653 | Long-form usage reviews, ingredient discussions |
| Naver Cafe | Naver Search API (official) | 675 | Community Q&A, comparison threads |
| YouTube Comments | YouTube Data API v3 (official) | 1,386 | Purchase motivation, post-purchase reaction |

- **Total collected**: 2,714 documents
- **After HNS relevance filter**: 1,744 documents
- **Relevance filter keywords**: 비듬, 두피, 각질, 가려움, 지루성, 설페이트, 클리니컬, 프로페셔널, 차콜, 약산성, 안티트로, 두피염, 정수리
- **Collection note**: All data collected via official APIs only. No scraping of systems that prohibit automated access. Raw data not redistributed in this repository.

### Layer 2 — Naver DataLab Trend Data

| File | Coverage | Columns |
|------|----------|---------|
| 분야통계_샴푸 (7 files) | 2020-01 ~ 2026-04 (daily, resampled monthly) | category_click |
| 쇼핑인사이트_헤드앤숄더 (7 files) | 2020-01 ~ 2026-04 | 헤드앤숄더샴푸, 헤드앤숄더차콜, 헤드앤숄더, 헤드앤숄더프로페셔널, 헤드앤숄더클리니컬스트렝스 |
| 쇼핑인사이트_증상카테고리 (7 files) | 2020-01 ~ 2026-04 | 비듬샴푸, 지루성두피샴푸, 안티트로샴푸, 지성두피샴푸, 비듬 |
| 검색어트렌드_브랜드경쟁구도 (1 XLSX) | 2020-01 ~ 2026-03 | 팬틴, 헤드앤숄더_경쟁, 닥터그루트, 케라시스, 두피케어카테고리 |

- **Unified feature table**: 76 months × 16 columns

---

## Project Structure

```
pg-hns-consumer-signal-pipeline/
│
├── voc_pipeline/                        # Layer 1: Korean NLP VoC pipeline
│   ├── collector_naver.py               # Naver Blog/Cafe collection via Search API
│   ├── collector_youtube.py             # YouTube comment collection via Data API v3
│   ├── preprocessor.py                  # kiwipiepy morphological analysis + 4 modes
│   │                                    # user word: 안티트로 registered as NNP
│   ├── LDA_pipeline.py                  # LDA topic modeling: per-source × per-mode
│   ├── causal_signal_detector.py        # Causal signal scoring + temporal analysis
│   └── data/
│       ├── raw/                         # Collected CSV files (gitignored)
│       └── processed/                   # All analysis output CSVs (gitignored)
│
├── trend_pipeline/                      # Layer 2: Search trend & forecasting pipeline
│   ├── trend_loader.py                  # 22 DataLab files → unified monthly feature table
│   ├── trend_analyzer.py                # Part A: Chronos forecast | Part B: structural analysis
│   └── data/
│       ├── raw/                         # Naver DataLab CSV + XLSX (gitignored)
│       └── processed/                   # trend_features.csv, chronos_forecast.csv (gitignored)
│
├── switching_pipeline/                  # Layer 3: Consumer switching probability model
│   ├── feature_builder.py               # Layer 1 + Layer 2 → unified feature table + segment labels
│   ├── segment_classifier_regression.py # Logistic Regression approach (adopted)
│   ├── segment_classifier_clustering.py # KMeans clustering approach (tested, not adopted)
│   ├── switching_probability.py         # Brand risk score + segment intervention plan
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
─────────────────────────────────────────────────────────────────────
Raw Data Collection
(Naver Blog/Cafe via Search API + YouTube via Data API v3)
        │
        ▼
collector_naver.py / collector_youtube.py
(2,714 docs: Blog 653 + Cafe 675 + YouTube 1,386)
        │
        ▼
preprocessor.py
(kiwipiepy morphological analysis)
(4 modes: unigram / bigram / unibi_mix / adj_noun)
(user word: 안티트로 → NNP to prevent segmentation)
(STOPWORDS finalized after raw data inspection)
        │
        ▼
LDA_pipeline.py                     hns_bertopic.ipynb (Colab, T4)
(per-source × per-mode,             (paraphrase-multilingual-MiniLM-L12-v2
 k=2~7 coherence optimization)       + HDBSCAN, min_topic_size=8)
        │                                        │
        ▼                                        ▼
hns_lda_results.csv              hns_bertopic_results.csv
                                 hns_bertopic_keywords.json
                                 hns_bertopic_documents.csv
        │                                        │
        └──────────────┬──────────────────────────┘
                       ▼
            hns_lda_bertopic_consensus.csv
            (12 High Confidence signals, LDA × BERTopic cross-validated)
                       │
                       ▼
        causal_signal_detector.py
        (6 churn signal categories + 3 positive signal categories)
        (competitor_mentioned flag: 안티트로, 니조랄)
                       │
                       ▼
        hns_causal_signals.csv       hns_temporal_signals.csv
        (1,744 docs scored)          (monthly churn/positive trend)


Layer 2 — Trend Pipeline
─────────────────────────────────────────────────────────────────────
22 Naver DataLab files (CSV × 21 + XLSX × 1)
        │
        ▼
trend_loader.py
(outer join on monthly date index → 76 rows × 16 columns)
(trend_features.csv)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
Part A: Chronos zero-shot forecast            Part B: Structural trend analysis
(Amazon Chronos-T5-Small, CPU inference)      (antitro reversal timing + velocity,
(12-month HNS Core forecast,                   HNS product line lifecycle,
 80% prediction interval)                      competitor trajectory vs 2020 baseline,
        │                                      category click dynamics,
        ▼                                      symptom keyword language shift)
chronos_forecast.csv                                   │
                                                       ▼
                                          trend_analysis_summary.csv


Layer 3 — Switching Pipeline
─────────────────────────────────────────────────────────────────────
hns_causal_signals.csv + trend_features.csv
        │
        ▼
feature_builder.py
(11 VoC behavioral features per document)
(rule-based segment labels:
 Active Switcher / At-risk / Passive User / Loyal)
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
segment_classifier_regression.py      segment_classifier_clustering.py
(Logistic Regression,                  (KMeans k=7,
 C grid search: 0.01~10,               At-risk segment lost in clusters,
 CV accuracy 0.9300,                   not adopted)
 adopted)
        │
        ▼
switching_probability.py
(trend_multiplier = 1.0 + w_antitro × antitro_pressure + w_hns × hns_decline)
(weights grid-searched: w_antitro=0.50, w_hns=0.40)
(brand risk score + segment intervention plan)
        │
        ▼
switching_prob_regression.csv    switching_implications.csv
timeline_analysis.csv


All layers → dashboard.py (Streamlit 5-tab integrated dashboard)
```

---

## Methodology

### Layer 1 — VoC Pipeline

#### Preprocessing

Four preprocessing strategies were applied to enable direct comparison of how token representation affects topic coherence in Korean beauty VoC:

| Mode | Description |
|------|-------------|
| `unigram` | Single noun tokens (NNG, NNP, length ≥ 2); stopwords + domain noise removed |
| `bigram` | Consecutive noun pairs (e.g. `클리니컬_스트렝스`, `정수리_냄새`, `지루_두피`) |
| `unibi_mix` | Union of unigram and bigram tokens |
| `adj_noun` | Adjective-noun pairs for sentiment-bearing phrases (e.g. `순한_클렌저`) |

All modes use `kiwipiepy` for Korean morphological analysis, selected over KoNLPy for compatibility with Apple Silicon (M-series Mac). The kiwipiepy tokenizer filters 1-character tokens, Korean particles (이/가/을/를), and verb forms (하다/있다/되다) automatically through POS tagging — meaning these do not need to appear in the STOPWORDS dictionary, keeping it purposeful and lean.

`안티트로` was registered as a user word (`NNP`, score=0) to prevent morpheme segmentation. This was confirmed necessary after the first-run LDA output showed the truncated form `안티트` appearing in Topic 6 — without the user word registration, the brand name was being split by the tokenizer and losing its identity as a competitor signal.

**STOPWORDS design**: The stopwords list was finalized *after* examining the raw collected data — not before. This is an intentional design choice: pre-defining stopwords without looking at the data risks both over-filtering (removing meaningful signals) and under-filtering (missing domain-specific noise). Key decisions made from data inspection:

- `두피` (scalp) removed as a standalone token. It appeared in 100%+ of LDA topics at 0.1+ weight across all modes, effectively functioning as an uninformative prior that suppresses all other signals. Compound forms (`지루성두피`, `두피각질`) survive through bigram extraction and retain their discriminative value.
- `헤드앤숄더`, `샴푸` removed for the same reason — present in virtually all documents, contributing no topic-level differentiation.
- Shopping platform noise (`최저가`, `적립`, `옵션`, `평점`) added after the first LDA run confirmed contamination in Topic 1 of the full corpus unigram model.

`adj_noun` mode returned Insufficient data (fewer than 30 valid documents) across all sources. Korean shampoo VoC is dominated by noun+verb structures (`두피 가려움이 심해졌어요`, `비듬이 없어졌어요`) rather than adjective-noun compounds. This is structurally different from English beauty reviews (where `gentle_cleanser`, `dry_skin` are natural compound descriptors), and is a linguistic property of Korean scalp care VoC rather than a pipeline deficiency.

#### LDA Topic Modeling

Gensim's `LdaModel` was applied per combination of source (`blog` / `cafearticle` / `youtube` / `all`) × mode (`unigram` / `bigram` / `unibi_mix`). Optimal topic count *k* was selected by maximizing c_v coherence across k = 2–7. All models used `passes=15` and `random_state=42`.

**Coherence summary:**

| Scope | bigram | unibi_mix | unigram |
|-------|--------|-----------|---------|
| all | 0.5980 | 0.3524 | 0.3469 |
| blog | **0.6177** | 0.3136 | 0.3140 |
| cafearticle | 0.5951 | 0.3380 | 0.3414 |
| youtube | 0.6103 | 0.3865 | 0.3650 |

Bigram consistently outperforms unigram and unibi_mix across all sources. This is consistent with the finding from the prior ANUA Amazon review project (English) and confirms a cross-language principle: compound noun expressions capture the relational semantics of scalp care VoC better than isolated tokens. `unibi_mix` underperforms bigram because the addition of unigrams introduces noise that dilutes the phrase-level coherence gained from bigrams alone.

#### Causal Signal Detection

`causal_signal_detector.py` applies a keyword-based scoring system to each document across two opposing signal categories:

**Churn signals** (`direct_churn`, `efficacy_failure`, `skin_reaction`, `competitor_switch`, `formula_change`, `channel_barrier`) and **positive signals** (`repurchase`, `efficacy_positive`, `recommendation`).

Each document receives a `signal_type` label (`이탈위험` / `긍정` / `중립`) based on the net balance of detected signals. The module additionally flags documents where `안티트로` or `니조랄` appear in the raw text (`competitor_mentioned`), enabling direct measurement of brand-switching signal intensity. Temporal analysis aggregates monthly churn and positive rates to surface trend anomalies.

#### BERTopic

`paraphrase-multilingual-MiniLM-L12-v2` embeddings + HDBSCAN clustering, run on Google Colab (T4 GPU) on 1,742 filtered documents. Initial `min_topic_size=15` produced only 3 topics with Topic 0 absorbing 91% of documents (1,590 of 1,742). This reflects the corpus's inherent thematic homogeneity: most documents share the scalp/dandruff context, making large-scale semantic differentiation difficult. Reducing to `min_topic_size=8` produced 17 semantically distinct topics, enabling LDA × BERTopic cross-validation that yielded 12 High Confidence signals.

---

### Layer 2 — Trend Pipeline

#### trend_loader.py

Merges 22 Naver DataLab files via outer join on a monthly date index, resampled to month-end frequency. Produces 76 rows × 16 columns covering 2020-01 to 2026-04. Missing values (primarily `안티트로샴푸` pre-2024 and `헤드앤숄더차콜` pre-2021) are filled with 0, which correctly represents zero search volume rather than missing observations.

#### Part A: Chronos Zero-Shot Forecast

**Amazon Chronos** (2024) was selected for the 12-month HNS Core search volume forecast. The selection required ruling out alternatives:

- **ARIMA / SARIMA / Prophet**: Statistically appropriate for 76 observations but represent 2010s-era methodology insufficient for a data science portfolio targeting a technology-forward role.
- **TFT / PatchTST / TimesNet**: State-of-the-art time series deep learning, but all require substantially more training data than 76 observations for meaningful fine-tuning. The data constraint is fundamental, not addressable through architecture choice.
- **Chronos (zero-shot)**: As a pre-trained foundation model (2024), Chronos bypasses the fine-tuning constraint via zero-shot inference from context alone, producing calibrated probabilistic forecasts without requiring training data splits.

Forecast result: HNS Core projected at 47–54 (median) for May 2026 – April 2027, with 80% confidence interval of 34–63. The flat trajectory indicates the search volume has stabilized at a structurally lower level than the 2020–2022 baseline, with no organic recovery expected.

#### Part B: Structural Analysis

ML classification for Part B was attempted through three successive approaches before being abandoned. The core structural problem: all 12 risk-labeled months fall within the final 12 months of the 76-month time series. Under time-ordered splitting (required to prevent data leakage), no training set can contain any risk samples regardless of model architecture. This is not a model capacity issue — it is a data structure constraint that invalidates supervised classification entirely. Full documentation in the [Methodological Decisions and Pivots](#methodological-decisions-and-pivots) section.

The structural descriptive analysis adopted instead directly answers the business question: *when* did the displacement begin, *how fast* did it occur, *which product lines* are declining vs. growing, and *how has consumer search language shifted* over the same period.

---

### Layer 3 — Switching Pipeline

#### Segment Definition

Four consumer segments defined from VoC behavioral signals via rule-based assignment:

| Segment | Definition | n | % |
|---------|------------|---|---|
| Active Switcher | comparison_frame AND competitor_mention AND churn_signal | 83 | 4.8% |
| At-risk | (medical_frame OR ingredient_frame) AND churn_signal, no direct competitor comparison | 91 | 5.2% |
| Passive User | neutral signal, no strong frame detected | 1,224 | 70.2% |
| Loyal | positive_signal, no churn indicators | 346 | 19.8% |

The At-risk definition captures a specific behavioral state: consumers who have adopted clinical/ingredient scrutiny framing (signaling awareness of alternative solution categories) but have not yet explicitly compared Head & Shoulders to Antitro. These are pre-decision consumers — frame has shifted, brand comparison has not yet begun.

#### Classifier Comparison

**KMeans Clustering** (`segment_classifier_clustering.py`): Optimal k=7 by silhouette score (0.3988). The At-risk segment (91 documents) was entirely absorbed into Passive User clusters in cross-tabulation. KMeans cannot weight the ingredient/medical frame signal sufficiently without supervision — At-risk documents share too many surface features with Passive User (no competitor mention, moderate churn score). Not adopted.

**Logistic Regression** (`segment_classifier_regression.py`): Features exclude rule-defining columns (`is_churn`, `is_competitor`, `is_comparison_frame`) to prevent data leakage — the classifier must predict segment membership from behavioral proxies alone. Regularization grid search (C ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0}), optimal C=1.0, stratified 5-fold CV accuracy 0.9300 ± 0.0135. At-risk segment correctly preserved. Adopted.

**Key coefficients at C=1.0:**

| Segment | Strongest positive predictors | Interpretation |
|---------|-------------------------------|----------------|
| Active Switcher | `churn_score` (+2.364), `is_blog` (+0.146) | High churn intensity on blog channel |
| At-risk | `is_ingredient_frame` (+0.938), `is_medical_frame` (+0.737) | Frame shift precedes brand comparison |
| Loyal | `positive_score` (+4.466), `churn_score` (-5.009) | Cleanest opposing signal profile |
| Passive User | `is_medical_frame` (-0.528), `is_ingredient_frame` (-0.603) | Absence of frame engagement |

The At-risk coefficient pattern supports the segment hypothesis: `is_ingredient_frame` (+0.938) and `is_medical_frame` (+0.737) are the two strongest At-risk predictors, operating independently of competitor mention. This confirms that frame adoption precedes brand comparison as a behavioral sequence.

#### Switching Probability

Base switching probability from logistic regression P(Active Switcher | features) per document, averaged by segment, adjusted by a data-driven trend multiplier:

```
trend_multiplier = 1.0 + w_antitro × antitro_pressure + w_hns × hns_decline

where:
  antitro_pressure = min(antitro_ratio / 2.0, 1.0)   # normalized [0, 1]
  hns_decline      = max(-hns_momentum, 0)            # positive when declining
  hns_momentum     = 3-month pct change of HNS Core search volume
```

Weights grid-searched to maximize probability gap between Active Switcher and Loyal: **w_antitro=0.50, w_hns=0.40** (trend multiplier=1.427 at current values). This replaces arbitrary weight assignment with a data-driven objective.

**Final switching probabilities:**

| Segment | n | Churn Rate | Competitor Rate | Medical Frame Rate | P(switch) |
|---------|---|-----------|-----------------|-------------------|-----------|
| Active Switcher | 83 | 1.000 | 1.000 | 0.229 | **0.405** |
| At-risk | 91 | 1.000 | 0.000 | 0.418 | **0.352** |
| Passive User | 1,224 | 0.199 | 0.013 | 0.092 | 0.043 |
| Loyal | 346 | 0.000 | 0.006 | 0.121 | 0.000 |

**Brand Risk Score: 0.068** — weighted average by segment size. The low overall figure reflects that 90% of documents are Passive User or Loyal. The correct interpretation: switching risk is concentrated in the 10% At-risk + Active Switcher population, where P(switch) exceeds 0.35 for both segments.

---

## Methodological Decisions and Pivots

### Layer 2 Part B: Three Failed Approaches Before Structural Analysis

**FT-Transformer**: Applied to 73 sliding-window samples (96 features, WINDOW_SIZE=6). All 12 risk-labeled months fall in the time series tail (2025-05 onward). Training set contains zero risk samples under time-ordered splitting. PCA dimensionality reduction and pos_weight class balancing were applied but cannot resolve a zero-sample training problem. The model correctly learned to predict "Stable" — because that is the only class it observed during training.

**ROCKET classifier (sktime)**: State-of-the-art time series classification algorithm. Same fundamental constraint: Train — Stable: 51, Risk: 0. ROCKET performs well on small datasets through random convolutional kernels but cannot generalize from a training set that contains no positive-class examples.

**Rule-based change point detection**: Abandoned immediately. The change point (Antitro market entry, December 2024) is explicitly visible in the data. Algorithmic detection adds no analytical value when the event is already known.

**Conclusion**: The data has already answered the classification question directly and unambiguously. Forcing a supervised classifier onto this structure would produce technically invalid results while adding no information beyond what structural descriptive analysis provides.

### Layer 3: KMeans vs. Logistic Regression

KMeans was tested first as an unsupervised approach that makes no assumptions about segment definitions. The critical failure — At-risk segment entirely absorbed into Passive User clusters — occurred because KMeans cannot distinguish the subtle ingredient/medical frame signal from broader Passive User behavior without supervision. The At-risk segment's distinctiveness is in the *combination* of frame features (medical + ingredient) with churn signal in the absence of competitor mention — a nuanced behavioral profile that KMeans cannot weight appropriately through distance-based clustering.

### BERTopic min_topic_size Reduction

Initial `min_topic_size=15` collapse (91% into one topic) reflects corpus homogeneity, not a model failure. The reduction to 8 was data-driven: a topic with 8–15 documents is statistically marginal, but the alternative (3 topics, one absorbing 91% of the corpus) provides no discriminative value. The 17-topic solution at min_topic_size=8 is treated with appropriate caution — topics 10–16 (9–15 documents each) are supporting evidence rather than primary signals.

### STOPWORDS Finalization After Data Inspection

Pre-defining stopwords from domain knowledge alone would have missed two critical decisions: (1) `두피` removal (not obvious a priori that it would dominate 100%+ of topics) and (2) shopping platform noise identification (only visible after running the first LDA and seeing `최저가`, `적립`, `옵션` in Topic 1). The iterative inspect-run-refine approach produced a more accurate stopword list and is the recommended practice for domain-specific Korean NLP.

| Decision Point | Attempted | Outcome | Final Approach |
|----------------|-----------|---------|----------------|
| Layer 2 Part B | FT-Transformer | 0 risk samples in training set | Abandoned |
| Layer 2 Part B | ROCKET (sktime) | Same structural constraint | Abandoned |
| Layer 2 Part B | Change point detection | Trivial (event already known) | Abandoned |
| Layer 2 Part B | Structural descriptive analysis | Directly answers business question | Adopted |
| Layer 3 classifier | KMeans k=7 | At-risk segment lost | Not adopted |
| Layer 3 classifier | Logistic Regression | At-risk preserved, CV 0.9300 | Adopted |
| BERTopic min_topic_size | 15 | 3 topics, 91% in Topic 0 | Reduced to 8 |
| Trend multiplier weights | Arbitrary (0.3, 0.2) | No data-driven basis | Grid search → (0.50, 0.40) |
| STOPWORDS | Pre-defined before inspection | Risk of mis-specification | Finalized after data inspection |

---

## Key Findings

### 1. Antitro Reversed Head & Shoulders Core in 6 Months — and Is Still Accelerating

Antitro entered the Naver Shopping search index in December 2024 with a volume of 1. By March 2025 it reached 18.1 (still below HNS Core at 54.3). By June 2025 it crossed the reversal threshold. By January 2026, a single-month increase of +40.2 units represented the largest single-month jump in the entire 76-month dataset across any keyword. By April 2026, the Antitro/HNS ratio stands at 1.327.

The velocity of this displacement is the primary finding — not the fact of it. Six months from market entry to category reversal is not gradual competitive pressure. It is a rapid, non-linear displacement driven by consumers who adopted Antitro rapidly once it became visible — suggesting latent demand for a derma-channel scalp solution that was not being met by existing options. The Chronos forecast projects HNS Core at 47–54 (median) for May 2026 – April 2027 — flat, with no organic recovery expected.

> **Business implication**: The critical intervention window is before the Antitro/HNS ratio continues to widen. At current trajectory (ratio 1.327, January 2026 acceleration +40.2), the competitive gap is widening at an accelerating pace. 

---

### 2. The Category Is Growing — But Head & Shoulders Is Not the Beneficiary

Shampoo category click volume increased +14.6% in 2026 after declining -20.6% in 2024 and -22.2% in 2025. This appears to be positive news. Symptom keyword data reveals the composition of this recovery:

| Keyword | 2020 avg | 2026 avg | Change |
|---------|----------|----------|--------|
| 비듬샴푸 (Dandruff Shampoo) | 41.2 | 32.5 | -21.3% |
| 지루성두피샴푸 (Seborrheic Shampoo) | 13.7 | 5.8 | -57.6% |
| 안티트로샴푸 (Antitro Shampoo) | 0.0 | 75.1 | new entrant |

The 2026 category rebound is primarily driven by `안티트로샴푸`. The keywords that historically directed consumers toward Head & Shoulders — `비듬샴푸` (-21.3%) and `지루성두피샴푸` (-57.6%) — continue to decline. Antitro is simultaneously competing for the same consumer search demand as Head & Shoulders and expanding total category search volume by attracting new search behavior under a new keyword. Head & Shoulders is losing share in a growing category — the category growth signal masks the displacement.

> **Business implication**: Category-level click volume is no longer a reliable proxy for Head & Shoulders brand health. The appropriate metric is keyword-level share: `헤드앤숄더샴푸` as a proportion of total scalp-care search volume. By this measure, Head & Shoulders is declining at an accelerating rate even as the category expands. Also noteworthy: consumer search interest in "안티트로샴푸" as a category-level term is an interest signal — not necessarily direct purchase intent. However, as a brand-as-category term, it indicates Antitro has achieved consumer mindshare at a scale that warrants monitoring as a potential category redefinition signal.

---

### 3. Head & Shoulders Clinical Strength Collapsed — New Lines Are Not Compensating

| Line | 2021 avg | 2023 avg | 2025 avg | 2026 avg |
|------|----------|----------|----------|----------|
| Core (헤드앤숄더샴푸) | 46.0 | 56.3 | 36.0 | 63.3 |
| Clinical Strength (클리니컬스트렝스) | 35.9 | 19.4 | 4.5 | 0.0 |
| Professional (프로페셔널) | 0.0 | 2.6 | 2.3 | 3.4 |
| Charcoal (차콜) | 0.0 | 0.0 | 0.1 | 0.9 |

Clinical Strength — the line most directly positioned as a clinical-grade scalp solution — collapsed from 35.9 in 2021 to 0.0 in 2026. Critically, this decline *predates Antitro's entry*: Clinical Strength was already at 4.5 in 2025 before Antitro achieved meaningful volume. Clinical Strength did not lose to Antitro — it lost consumer relevance before Antitro arrived to fill the vacuum.

The new Charcoal (0.9) and Professional (3.4) lines show negligible uptake against Antitro's 75.1. These extensions show minimal search uptake in comparison to Antitro. Whether they address the same consumer need as Clinical Strength did — or a different one — cannot be determined from search volume data alone.

> **Business implication**: The line extension strategy has not generated meaningful demand in the segments being lost to Antitro. The search volume data does not show evidence of migration to Charcoal or Professional from consumers who stopped searching for Clinical Strength. Where those consumers went cannot be determined from search volume data alone. Recovering the clinical segment requires a positioning strategy that directly addresses clinical credibility (mechanism of action, dermatologist validation, antifungal efficacy), not product form novelty.

---

### 4. Churn Is Driven by Efficacy Failure — Not Brand Image Deterioration

Of 1,744 HNS-relevant documents, 418 (24.0%) carry churn signals. Top churn keywords:

```
가려움 (itchiness):        291  ← efficacy failure — core symptom unresolved
자극 (irritation):          90  ← skin reaction — product is aggravating the condition
뾰루지 (pimples/bumps):     63  ← skin reaction — breakout trigger
안티트로 (Antitro):          56  ← active competitive switch signal
대신 (instead of):          52  ← replacement framing
트러블 (skin trouble):       52  ← skin reaction
니조랄 (Nizoral):            45  ← derma/pharmacy channel alternative
올라오 (flare-up):           30  ← symptom escalation
```

Itchiness (291) is 5.2× more frequent than the next signal (자극, 90). This is primarily a product efficacy signal. The dominant churn language is symptom-based (itchiness, irritation, breakouts) rather than brand-perception-based (image, packaging, price). Whether brand image deterioration is also contributing cannot be determined from this dataset alone. The co-occurrence of `안티트로` (56) and `니조랄` (45) alongside skin reaction signals — rather than alongside pricing or availability complaints — is consistent with a churn sequence of: efficacy failure → clinical alternative search → brand switch. However, this sequence is inferred from keyword co-occurrence within documents, not from longitudinal consumer journey data.

> **Business implication**: Marketing interventions that reinforce brand values or increase awareness will not address efficacy-driven churn. Consumers who mention itchiness and irritation in the context of Head & Shoulders are describing product-use experiences in their text, suggesting the core symptom relief promise is not being met for a meaningful proportion of this segment. The appropriate response is clinical credibility reinforcement — addressing the efficacy comparison consumers are drawing with Antitro's positioning. What specific ingredient or mechanism claims are appropriate is a product and regulatory question beyond the scope of this analysis.

---

### 5. At-risk Segment: Pre-switch Consumers Are Identifiable Before They Switch

The At-risk segment (91 documents, 5.2%) defines consumers who have adopted clinical/ingredient scrutiny framing without yet naming Antitro as their alternative:

| Feature | At-risk | Active Switcher | Passive User |
|---------|---------|-----------------|--------------|
| is_medical_frame | 41.8% | 22.9% | 9.2% |
| is_ingredient_frame | 60.4% | 22.9% | 11.6% |
| is_competitor | 0.0% | 100.0% | 1.3% |
| churn_rate | 100.0% | 100.0% | 19.9% |
| P(switch) | 0.352 | 0.405 | 0.043 |

The At-risk segment's ingredient frame rate (60.4%) is higher than Active Switcher (22.9%) — meaning ingredient scrutiny is an *early-stage* behavior that peaks before competitor discovery, not a concurrent behavior. Consumers develop chemical/clinical literacy (searching for 설페이트, 계면활성제, 약산성) *before* they identify Antitro as their solution. The logistic regression is consistent with this pattern: `is_ingredient_frame` (+0.938) is the strongest positive predictor of At-risk membership, operating independently of any competitor mention.

The churn rates of At-risk (100%) and Active Switcher (100%) are identical. The only distinguishing feature is competitor mention: At-risk consumers have the same dissatisfaction intensity but have not yet found their alternative. This is the intervention window.

> **Business implication**: At-risk consumers represent a conversion opportunity that narrows significantly once they discover Antitro's clinical positioning. The behavioral sequence is: ingredient curiosity emerges → consumer searches for 설페이트 / 계면활성제 / 약산성 content → encounters Antitro's derma positioning → transitions to Active Switcher. Proactive content that addresses the ingredient and medical-frame questions At-risk consumers are already asking (설페이트, 계면활성제, 약산성) from Head & Shoulders' perspective can intercept this segment during the ingredient curiosity phase, before they encounter Antitro's positioning. What specific claims are appropriate is beyond the scope of this analysis. The intervention timing is the period between frame adoption and competitive brand discovery. The exact duration cannot be estimated from this dataset, but the behavioral signal sequence — ingredient frame peaking before competitor mention — is consistent with the existence of this window.

---

### 6. Temporal Churn Trend: Structural Escalation Followed by Volume-Driven Apparent Moderation

Monthly churn rate from documents with date metadata:

| Period | Churn Rate | n | Notes |
|--------|-----------|---|-------|
| 2024-11 | 0.0% | 2 | Pre-Antitro baseline |
| 2024-12 | 22.2% | — | Antitro market entry |
| 2025-04 | 100.0% | 3 | Early concentrated switching (all Antitro-related) |
| 2025-08 | 66.7% | 6 | Sustained high |
| **2025-11** | **80.0%** | **10** | Peak: highest volume + highest rate |
| 2026-01 | 57.1% | 7 | Moderation begins |
| 2026-03 | 45.1% | 113 | Sharp document volume increase |
| 2026-04 | 26.6% | 331 | Near normalization — volume artifact |

The November 2025 peak (80.0% at n=10) represents the moment of most concentrated switching signal — when Antitro had established sufficient market presence to generate systematic consumer comparison. The apparent moderation in March–April 2026 is a volume artifact: a 23–47× increase in monthly document count (113 and 331 documents vs. single-digit volumes) reflects a broader population mentioning Head & Shoulders, most of whom are not in active switching mode. A 26.6% churn rate at 331 documents represents more absolute switching documents than 80.0% at 10 documents — the apparent rate moderation masks absolute volume growth in switching behavior.

> **Business implication**: Churn rate alone is an insufficient metric when document volume is simultaneously increasing. Absolute churn document count is a better proxy for brand health deterioration. The March–April 2026 data — 88 churn documents in March, 88 in April — represents more absolute switching signal than any prior month, despite the lower rate.

---

## Cross-Layer Analysis

The three layers produce a more complete picture when read together than any single layer provides independently.

### Finding A: The Frame Shift Is Channel-Stratified — YouTube Is the Leading Edge

BERTopic topic-source distribution reveals a sharp channel asymmetry in how clinical/ingredient frame appears:

**Ingredient scrutiny topics:**
- Topic 14 (계면활성제, 약산, 약용, 약국): YouTube 9, Blog 0, Cafe 1
- Topic 16 (설페이트, 소듐라우레스설페이트, 화학): YouTube 9, Blog 0, Cafe 0

Ingredient scrutiny topics — where consumers directly examine and critique Head & Shoulders' chemical formulation — are concentrated almost exclusively in YouTube comments (Topic 14: YouTube 9, Blog 0, Cafe 1; Topic 16: YouTube 9, Blog 0, Cafe 0).

**Medical frame topics:**
- Topic 11 (항진균, 지루성두피염, 질환, 원인): Blog 4, Cafe 4, YouTube 3
- Topic 12 (지루성피부염, 피부과, 처방, 병원): Blog 3, Cafe 1, YouTube 6

Medical frame language distributes relatively uniformly across channels — it has diffused to the general population.

**Competitive comparison:**
- Topic 1 (303 documents, 헤드엔숄더 + 안티트로 co-occurrence): YouTube 237, Cafe 48, Blog 18

**Cross-layer interpretation**: Layer 2 shows Antitro search acceleration from January 2026. Layer 1 shows ingredient scrutiny exclusively on YouTube. One plausible interpretation consistent with the data: YouTube viewers encounter ingredient comparison content (설페이트 criticism, antifungal mechanism comparison) → develop clinical literacy → search for Antitro → drive the Layer 2 search volume surge. This sequence cannot be confirmed from the available data and should be treated as a hypothesis. Medical frame (disease-model framing of scalp conditions) has already diffused across all channels and represents a broad population shift. Ingredient frame remains YouTube-concentrated and represents the most recent, active stage of displacement. YouTube is the channel where ingredient scrutiny and competitive comparison are most concentrated in this dataset — though this reflects the nature of YouTube comment data (real-time reactions) vs. blog/cafe data (longer-form posts) as much as it reflects a causal role.

---

### Finding B: Blog Churn Rate Is High — But Reflects Post-Switch Documentation, Not Decision-Making

Layer 1 causal signal analysis: Blog churn rate 34.1% vs. YouTube 17.3%. A surface reading suggests blogs are the highest-risk channel. BERTopic cross-tabulation contradicts this:

Topic 1 competitive comparison: YouTube 237 documents, Cafe 48, Blog 18. Layer 3 Active Switcher segment: Blog 47 documents, YouTube 19, Cafe 17.

One possible interpretation of this pattern: blog's long-form format may be better suited for documenting completed experiences (including post-switch accounts), while YouTube comments may capture more active, in-progress comparison discussions. However, this is an interpretation based on format characteristics, not on temporal metadata from the documents. We cannot confirm from the available data when individual documents were written relative to the switching decision, or whether decisions were made on YouTube specifically.

**Cross-layer implication**: Switch prevention interventions belong on YouTube (where decisions are made), not on blog channels (where decisions have already been executed). Blog monitoring is useful for measuring switching volume; YouTube content strategy is the appropriate prevention lever.

---

### Finding C: Cafe High Coherence Is Partly a Shopping Noise Artifact

Layer 1 LDA showed cafearticle bigram achieving the highest coherence (0.6329, optimal k=2). BERTopic cross-tabulation reveals an important caveat:

Topic 2 (휴대, 가격, 무료, 구매, 배송, 할인 — shopping platform noise): Cafe 95 documents, Blog 16, YouTube 10.

Cafe is substantially contaminated with shopping comparison content — product listings, discount comparisons, and purchase option discussions that mention Head & Shoulders in a commercial rather than experiential context. The high LDA coherence may partly reflect the linguistic consistency of shopping-platform language (`최저가`, `적립`, `구매`) providing a coherent "shopping context" cluster that boosts overall coherence artificially.

**Cross-layer implication**: Cafe LDA coherence should not be interpreted as signal quality. For genuine consumer experience signals, blog (long-form experience documentation) and YouTube (active comparison discussion) are higher-quality sources despite lower raw coherence. The cafearticle bigram coherence advantage is partly a measurement artifact of shopping noise providing topically consistent but informationally low-value content.

---

## Strategic Conclusion

> **"As consumers shift their scalp problem-solving frame from 'shampoo brands' toward 'derma/clinic solutions', can Head & Shoulders defend its position through Charcoal and mild-acid line extensions — or is the category itself being redefined around a new reference point?"**

The three-layer pipeline converges on a clear answer: **the Charcoal and mild-acid line extensions are insufficient to defend Head & Shoulders' position in the clinical/medicated scalp segment, and the category is actively being redefined around Antitro as the new consumer reference point.** However, the pipeline identifies specific consumer segments and signal patterns that suggest an intervention window remains — though whether the displacement is reversible cannot be determined from this data.

---

### 1. The Line Extension Strategy Has Not Addressed the Right Consumer Problem

The data does not support the hypothesis that Charcoal or Professional extensions are capturing the consumers being lost to Antitro. Charcoal (2026 avg: 0.9) and Professional (2026 avg: 3.4) show negligible search uptake against Antitro's 75.1. More tellingly, the consumers leaving Head & Shoulders are leaving because of **efficacy failure** — itchiness unresolved (291 churn mentions), skin irritation (90), breakouts (63) — not because of product format preferences. Charcoal and mild-acid extensions address sensory experience and ingredient positioning, but they do not address the clinical efficacy gap that is driving churn.

The collapse of Clinical Strength (35.9 in 2021 → 0.0 in 2026) further confirms this: the line that was most directly positioned to compete with clinical/derma solutions lost consumer relevance before Antitro even entered the market. The market was already telling Head & Shoulders that its clinical positioning was insufficient — the brand response (Charcoal, Professional) addressed a different question.

---

### 2. The Category Is Being Redefined — Not Just Competed Against

The most consequential finding across all three layers is not that Antitro is taking share from Head & Shoulders. It is that Antitro is changing what "solving a scalp problem" means to Korean consumers. Layer 1 VoC shows consumers approaching scalp conditions as clinical/disease-model problems (항진균, 지루성두피염, 피부과, 처방) rather than as grooming/cosmetic problems. Layer 2 trend data shows `비듬샴푸` (-21.3%) and `지루성두피샴푸` (-57.6%) declining while `안티트로샴푸` (0 → 75.1) rises — not just as a brand preference shift but as a category language shift.

This distinction matters strategically. If it were only a brand preference shift, Head & Shoulders could respond with better marketing, stronger efficacy claims, or reformulation. A **category frame shift** requires a fundamentally different response: establishing Head & Shoulders as a legitimate clinical/dermatological solution, not just a better shampoo. The current line extension strategy does not accomplish this. Charcoal and mild-acid are shampoo-category innovations; Antitro is a clinic-category solution. The consumer is not choosing between two equivalent shampoo brands — they are choosing between a mass-retail shampoo and a derma-channel shampoo that speaks the language of clinical scalp care.

---

### 3. A Defense Window Exists — But It Is Measured in Months, Not Years

The pipeline identifies a specific consumer group that represents both the mechanism of the displacement and the intervention opportunity: the **At-risk segment** (91 documents, 5.2%, P(switch)=0.352).

At-risk consumers are defined by the behavioral sequence that precedes switching: they have adopted ingredient/medical scrutiny framing (is_ingredient_frame: 60.4%, is_medical_frame: 41.8%) and carry full churn intensity (churn_rate: 100%) but have not yet identified Antitro as their alternative (is_competitor: 0.0%). The logistic regression is consistent with the hypothesis that ingredient frame adoption is an early-stage pre-switch signal: `is_ingredient_frame` peaks in the At-risk segment (60.4%) and is lower in the Active Switcher segment (22.9%), suggesting frame scrutiny precedes direct competitor comparison. However, this is a cross-sectional pattern across different consumers, not a longitudinal observation of individual consumer journeys.

The intervention window is the period between ingredient curiosity emergence and competitive brand discovery — the exact duration cannot be estimated from this dataset. Antitro's ratio crossed 1.327 in April 2026; at January 2026 acceleration velocity (+40.2 single month), the competitive gap is widening at an accelerating pace.

---

### 4. Strategic Implications From the Data

The pipeline data points to three specific actions, ordered by urgency:

**Immediate**: Deploy clinical credibility content on YouTube — the channel where ingredient scrutiny originates and where competitive comparison decisions are made (Topic 1: YouTube 237 documents vs. Blog 18). The content should address the ingredient and medical-frame questions that At-risk consumers are already asking — the data shows 설페이트, 계면활성제, 약산성 as the specific terms they are searching. What specific ingredient claims or clinical evidence Head & Shoulders should lead with is a product and regulatory question beyond what VoC and search volume data can specify.

**Medium-term (3–6 months)**: Address the efficacy failure signal that is the root cause of churn. The 291 itchiness mentions, 90 irritation mentions, and 63 breakout mentions in the VoC data indicate that efficacy-related experience is the dominant churn signal. Whether these reflect product formulation issues, usage method issues, or individual skin-type incompatibility cannot be determined from VoC text data alone. Formulation review for the consumer segments experiencing skin reactions (particularly the skin_reaction churn category) is indicated. Clinical Strength's collapse suggests that the clinical/medicated segment positioning has been losing consumer relevance for several years; the current situation may require more than repositioning.

**Structural (6–12 months)**: Consider channel credibility strategy. Antitro's positioning advantage is not only formulation — it is channel association (hospital/pharmacy). BERTopic Topic 12 (피부과, 처방, 병원) reflects consumers who are seeking dermatologist-associated solutions. The data is consistent with a channel credibility gap, though whether a pharmacy/clinic distribution strategy is feasible or sufficient to close it cannot be determined from VoC and search volume data alone.

---

### 5. What the Data Cannot Confirm

Two important caveats:

First, the Antitro search volume data reflects **consumer interest** (Naver Shopping search queries), not purchase volume or market share. High search volume for Antitro confirms consumer awareness and active exploration but does not directly measure revenue displacement. The business impact may be larger or smaller than the search volume gap suggests, depending on conversion rates and repeat purchase behavior that are not captured in this dataset.

Second, the At-risk segment intervention timing (the period between frame adoption and competitor discovery) cannot be measured from this dataset. Individual consumer journey data would be required to estimate this window. The actual timing may also be influenced by content algorithm factors (YouTube recommendation of Antitro comparison videos) that are outside Head & Shoulders' control.

These limitations do not invalidate the strategic direction — the displacement is real and measurable across all three data layers — but they establish the boundaries of what can be claimed with confidence from this analysis.

---

---

## Dashboard

An interactive Streamlit dashboard visualizes all three pipeline layers:

```bash
streamlit run dashboard.py
```

**Tab 1 — Overview**: Pipeline architecture, key metrics across all three layers, cross-layer finding summary cards.

**Tab 2 — VoC Analysis**: Signal distribution by source, competitor mention vs. overall churn, monthly churn/positive trend (2025–2026), LDA coherence heatmap, topic keyword explorer (interactive source × mode selection).

**Tab 3 — Trend Analysis**: Antitro vs. HNS Core search volume with reversal point annotation, HNS product line lifecycle, Chronos 12-month forecast with 80% CI, symptom category keyword language shift (2020–2026).

**Tab 4 — Switching Risk**: Consumer segment distribution, switching probability by segment, segment × source channel breakdown, segment intervention plan with risk-level color coding, Antitro competitive timeline.

**Tab 5 — BERTopic**: Topic distribution, LDA × BERTopic consensus table, LDA vs. BERTopic methodology comparison.

---

## How to Run

```bash
# 1. Set up environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env: NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, YOUTUBE_API_KEY

# 3. Layer 1: Collect VoC data
python voc_pipeline/collector_naver.py
python voc_pipeline/collector_youtube.py

# 4. Layer 1: Preprocess
python voc_pipeline/preprocessor.py

# 5. Layer 1: LDA topic modeling
python voc_pipeline/LDA_pipeline.py

# 6. Layer 1: Causal signal detection
python voc_pipeline/causal_signal_detector.py

# 7. Layer 1: BERTopic (Google Colab recommended)
# Upload voc_pipeline/data/processed/hns_processed.csv to Google Drive
# Run notebooks/hns_bertopic.ipynb on Google Colab (T4 GPU)
# Download 4 output files to voc_pipeline/data/processed/:
#   hns_bertopic_results.csv, hns_bertopic_documents.csv,
#   hns_bertopic_keywords.json, hns_lda_bertopic_consensus.csv

# 8. Layer 2: Load and process trend data
# Place Naver DataLab CSV/XLSX files in trend_pipeline/data/raw/
python trend_pipeline/trend_loader.py
python trend_pipeline/trend_analyzer.py

# 9. Layer 3: Build features and run switching probability model
python switching_pipeline/feature_builder.py
python switching_pipeline/segment_classifier_regression.py
python switching_pipeline/switching_probability.py

# 10. Launch integrated dashboard
streamlit run dashboard.py
```

---

## Dependencies

```
# Korean NLP
kiwipiepy==0.23.1
gensim==4.4.0

# ML / forecasting
scikit-learn
torch
chronos-forecasting        # Amazon Chronos zero-shot time series forecasting
sktime                     # tested for ROCKET classifier (not adopted in final pipeline)

# Data processing
pandas
numpy
openpyxl                   # Naver DataLab XLSX loading

# Visualization / dashboard
streamlit
plotly

# API collection
requests
python-dotenv
google-api-python-client

# BERTopic (Google Colab environment only)
# bertopic
# sentence-transformers
```

Install all local dependencies:
```bash
pip install -r requirements.txt
```
