# Prior Art Search System

A multi-stage patent prior art search pipeline that progressively filters a patent corpus using classification codes, keyword scoring, citation network analysis, and semantic similarity — culminating in an AI-generated patent viability report.

Built by **Savio David and Atharva Chavan**.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Classification Search](#stage-1--classification-search)
  - [Stage 2 — Keyword Search](#stage-2--keyword-search)
  - [Stage 3 — Citation Search](#stage-3--citation-search)
  - [Stage 4 — Semantic Search](#stage-4--semantic-search)
  - [Final Analysis — AI Report](#final-analysis--ai-report)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Input Format](#input-format)
- [Output Files](#output-files)
- [Sample Datasets](#sample-datasets)
- [Technical Design Decisions](#technical-design-decisions)

---

## Overview

Given a description of a new invention (title, abstract, IPC/CPC classification codes, and keywords), this system searches a corpus of existing patents to find the most similar prior art. It then generates a structured PDF report — powered by Llama 3.3 70B via Groq — that provides a patent viability verdict, similarity breakdown, novelty gaps, risk areas, future scope, and recommended next steps.

**Key capabilities:**

- Hierarchical IPC/CPC classification filtering for high recall
- Weighted keyword scoring with title/abstract differentiation
- Bidirectional citation graph traversal with PageRank-based influence scoring
- Two-stage semantic search: SPECTER bi-encoder retrieval + cross-encoder reranking
- Mahalanobis distance retrieval on large corpora (≥500 patents), cosine fallback on small corpora
- Citation score fusion with semantic scores for final ranking
- Professionally formatted PDF report with AI analysis

---

## System Architecture

```
input.csv  +  all_scraped_patents.csv
        │
        ▼
┌───────────────────────┐
│  Stage 1              │  IPC/CPC prefix matching
│  Classification Search│  High recall — cast a wide net
└──────────┬────────────┘
           │  classified_patents.csv
           ▼
┌───────────────────────┐
│  Stage 2              │  Weighted keyword scoring
│  Keyword Search       │  Title ×3, Abstract ×1, top-N filter
└──────────┬────────────┘
           │  keyword_filtered_patents.csv
           ▼
┌───────────────────────┐
│  Stage 3              │  Bidirectional BFS expansion
│  Citation Search      │  PageRank influence scoring
└──────────┬────────────┘
           │  citation_expanded_patents.csv
           ▼
┌───────────────────────┐
│  Stage 4              │  SPECTER bi-encoder + Mahalanobis/cosine
│  Semantic Search      │  Cross-encoder reranking → citation fusion
└──────────┬────────────┘
           │  final_results.csv
           ▼
┌───────────────────────┐
│  Final Analysis       │  Groq API (Llama 3.3 70B)
│  AI Report            │  PDF + TXT report generation
└───────────────────────┘
           │
           ▼
    Output/patent_analysis_report.pdf
```

---

## Pipeline Stages

### Stage 1 — Classification Search

**File:** `classification_search.py`

Filters the full patent corpus by IPC and CPC classification codes using prefix matching. Input codes from `input.csv` are matched hierarchically — for example, an input code of `G06F` matches all patents whose codes begin with `G06F`, covering the entire computing and data processing hierarchy.

- **Goal:** Maximum recall — retain all potentially relevant patents
- **Method:** Prefix matching on IPC and CPC code columns
- **Output:** `classified_patents.csv`

### Stage 2 — Keyword Search

**File:** `keyword_search.py`

Scores each classified patent against the input keywords using weighted term frequency. Title matches are weighted 3× higher than abstract matches, reflecting the higher information density of patent titles. Patents below a minimum score threshold are eliminated, and results are truncated to the top N.

- **Goal:** Precision enhancement — filter to keyword-relevant patents
- **Scoring:** `score = title_matches × 3 + abstract_matches × 1`
- **Parameters:** `min_score=3`, `top_n=20` (configurable)
- **Output:** `keyword_filtered_patents.csv`

### Stage 3 — Citation Search

**File:** `citation_search.py`

Expands the keyword-filtered seed patents using bidirectional BFS traversal of the citation graph, then ranks all discovered patents using a composite score that incorporates PageRank.

**Graph construction:** Both backward citations (patents a given patent cites) and forward citations (patents that cite a given patent) are stored explicitly, enabling traversal in both directions.

**PageRank:** Standard PageRank (damping=0.85, 50 iterations) is computed over the full corpus. This ensures that being cited by a highly-cited patent contributes more to a patent's score than being cited by an obscure one — a key advantage over raw citation counts.

**Composite score components:**

| Component | Weight | Rationale |
|---|---|---|
| PageRank | 0.50 | Network influence in full citation graph |
| Forward citations | 0.20 | Direct popularity signal |
| Backward citations | 0.15 | Breadth of prior art grounding |
| Seed bonus | 0.10 | Keyword stage relevance preserved |
| Hop penalty `1/(1+d)` | 0.05 | Proximity to query patent |

- **BFS depth:** 2 hops (configurable)
- **Output:** `citation_expanded_patents.csv` with per-patent score breakdown

### Stage 4 — Semantic Search

**File:** `semantic_search.py`

Two-stage semantic pipeline that converts patent text to dense vector embeddings and computes conceptual similarity — independent of exact vocabulary overlap.

**Stage 4a — Bi-encoder retrieval:**

Text representations are built by concatenating fields:
```
[TITLE] {title} [TITLE] {title} [ABSTRACT] {abstract} [IPC] {ipc_codes}
```
Title is repeated twice to upweight it without a separate late-fusion step.

Embeddings are generated using **SPECTER** (`sentence-transformers/allenai-specter`), trained by AllenAI specifically on scientific and technical documents using citation graphs as a training signal — directly aligned with this pipeline's design.

Retrieval metric selection:
- **Corpus ≥ 500 patents:** Mahalanobis distance, which accounts for the anisotropic structure of transformer embedding spaces by re-weighting dimensions using the inverse covariance matrix. Regularisation (`λ=1e-5`) ensures positive definiteness.
- **Corpus < 500 patents:** Cosine similarity fallback. Insufficient data for stable covariance estimation — logged explicitly.

**Stage 4b — Cross-encoder reranking:**

The top-K shortlist from Stage 4a is reranked using **`cross-encoder/ms-marco-MiniLM-L6-v2`**. Unlike bi-encoders which compress each text independently, cross-encoders process `(query, candidate)` pairs jointly with full attention between both texts — recovering token-level interaction signal lost during compression.

**Score fusion:**

Citation scores from Stage 3 are fused with semantic scores before final ranking:
```
final_score = α × norm(rerank_score) + (1 - α) × norm(citation_score)
```
Both scores are min-max normalised to `[0, 1]` before fusion. Default `α = 0.7`.

**Fallback backends** (used if `sentence-transformers` is not installed):
- Bi-encoder: TF-IDF + Truncated SVD (LSA, 128 components)
- Reranker: BM25 token overlap scorer

- **Output:** `final_results.csv` with all score components

### Final Analysis — AI Report

**Function:** `run_analysis()` in `run_pipeline.py`

The top-5 results and the original invention description are passed to **Llama 3.3 70B** via the Groq API. The model produces a structured patent analysis report with six sections:

1. **Patent Viability Verdict** — RECOMMENDED / CONDITIONAL / NOT RECOMMENDED
2. **Similarity Breakdown** — per prior art patent: what overlaps, what differs
3. **Novelty Gaps** — aspects not present in any prior art (strongest patent claims)
4. **Risk Areas** — aspects most closely overlapping with prior art (weakest claims)
5. **Future Scope** — concrete technical directions to strengthen the application
6. **Recommended Next Steps** — practical actions for the researcher

The report is saved as both `.txt` and a formatted `.pdf` (via ReportLab).

---

## Project Structure

```
Prior Art Search/
│
├── run_pipeline.py            # Main entry point — runs all 4 stages + analysis
│
├── classification_search.py   # Stage 1: IPC/CPC prefix matching
├── keyword_search.py          # Stage 2: Weighted keyword scoring
├── citation_search.py         # Stage 3: PageRank + BFS citation expansion
├── semantic_search.py         # Stage 4: SPECTER + cross-encoder reranking
├── pdf_report.py              # PDF report generator (ReportLab)
│
├── input.csv                  # Query patent — YOUR INVENTION GOES HERE (sample provided)
├── all_scraped_patents.csv    # Patent corpus to search (sample provided)
│
├── .env                       # API keys and environment config — DO NOT COMMIT
├── .env.example               # Template for .env (safe to commit)
├── .gitignore
├── requirements.txt
│
└── Output/                    # Created automatically on first run
    ├── classified_patents.csv
    ├── keyword_filtered_patents.csv
    ├── citation_expanded_patents.csv
    ├── final_results.csv
    ├── patent_analysis_report.txt
    └── patent_analysis_report.pdf
```

---

## Requirements

- **Python 3.9+**
- **~1.5 GB disk space** for model downloads (one-time):
  - `sentence-transformers/allenai-specter` — ~440 MB
  - `cross-encoder/ms-marco-MiniLM-L6-v2` — ~85 MB
- **Groq API key** (free) — [console.groq.com](https://console.groq.com)
  - Note: Groq console is geo-restricted in some regions. If access is blocked, use a VPN connected to the Netherlands or another supported region for the one-time sign-up. The API itself (`api.groq.com`) works globally without a VPN.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Gmax-13/prior-art-search.git
cd prior-art-search
```

**2. Create and activate a virtual environment** (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the ML models** (one-time, ~525 MB)

```bash
python model_downloader.py
```

Models are cached to the path set in `HF_HOME` (see Configuration). Subsequent runs load from cache with no network calls.

---

## Configuration

**1. Create your `.env` file** by copying the example:

```bash
cp .env_example.txt .env
```

**2. Edit `.env` and fill in your values:**

```env
# Required — get from https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# Path where ML models are cached after download
# Change this to any directory with ~1.5 GB free space
HF_HOME=D:/huggingface_cache

# Leave these as-is
HF_HUB_DISABLE_SYMLINKS_WARNING=1
HF_HUB_OFFLINE=1
```

**3. (Windows only) Enable Developer Mode** to fix HuggingFace symlink warnings:

Settings → System → For Developers → Developer Mode → **On**

This is optional — models work correctly without it, just with slightly less efficient caching.

---

## Usage

**Run the full pipeline:**

```bash
python run_pipeline.py
```

The pipeline will:
1. Run all 4 search stages sequentially
2. Call the Groq API for AI analysis
3. Save all outputs to the `Output/` folder
4. Print a ranked results table and the full analysis to the console

**Expected runtime:**
- Stages 1–3: < 1 second each (pure Python/pandas)
- Stage 4: 5–30 seconds on CPU depending on corpus size and hardware
- Groq API call: 5–15 seconds

**Tunable parameters** — edit at the top of `run_pipeline.py`:

```python
KEYWORD_MIN_SCORE  = 3     # minimum score for Stage 2 filter
KEYWORD_TOP_N      = 20    # max patents passed from Stage 2 to Stage 3
CITATION_MAX_DEPTH = 2     # BFS hops in citation graph
CITATION_TOP_N     = 40    # max patents passed from Stage 3 to Stage 4
SEMANTIC_TOP_K     = 20    # shortlist size for cross-encoder reranking
SEMANTIC_TOP_N     = 5     # final results returned
SEMANTIC_ALPHA     = 0.7   # semantic weight in score fusion (1-α = citation weight)
```

---

## Input Format

### `input.csv` — Your Invention

This file describes the invention you want to search prior art for. It must contain exactly one row.

| Column | Description | Example |
|---|---|---|
| `title` | Short title of the invention | `Semantic patent search system` |
| `abstract` | Description of the invention | `A system for identifying similar patent documents...` |
| `ipc_codes` | IPC classification code(s), comma-separated | `G06F 40/30` |
| `cpc_codes` | CPC classification code(s), comma-separated | `G06F 40/30` |
| `keywords` | Search keywords, comma-separated | `patent search,semantic similarity,nlp` |
| `citations` | Known related patent IDs (optional) | `P001,P005` |
| `publication_year` | Year (informational only) | `2024` |

**To search for your own invention:** replace the contents of `input.csv` with your invention's details. IPC/CPC codes can be looked up at [ipcpub.wipo.int](https://ipcpub.wipo.int).

### `all_scraped_patents.csv` — Patent Corpus

The database of patents to search against.

| Column | Description |
|---|---|
| `patent_id` | Unique identifier (e.g. `P001`, or a real patent number) |
| `title` | Patent title |
| `abstract` | Patent abstract |
| `ipc_codes` | IPC classification code(s) |
| `cpc_codes` | CPC classification code(s) |
| `citations` | Comma-separated IDs of patents this patent cites |
| `publication_year` | Year of publication |

---

## Output Files

All outputs are written to the `Output/` directory, which is created automatically.

| File | Stage | Description |
|---|---|---|
| `classified_patents.csv` | Stage 1 | Patents matching input IPC/CPC codes |
| `keyword_filtered_patents.csv` | Stage 2 | Top patents by keyword score, with `keyword_score` column added |
| `citation_expanded_patents.csv` | Stage 3 | BFS-expanded patents with `citation_score`, `pagerank_score`, `forward_citations`, `backward_citations`, `hop_distance` columns |
| `final_results.csv` | Stage 4 | Top-N ranked patents with `final_score`, `rerank_score`, `retrieval_score`, `embed_backend`, `retrieval_metric`, `rerank_backend` columns |
| `patent_analysis_report.txt` | Analysis | Plain text version of the AI report |
| `patent_analysis_report.pdf` | Analysis | Formatted PDF report with cover page, ranked table, patent detail cards, and full AI analysis |

---

## Sample Datasets

> **Note:** `input.csv` and `all_scraped_patents.csv` are **sample datasets for demonstration and testing only.**

`input.csv` contains a single dummy invention — a *Semantic Patent Search System* — used to verify that the pipeline runs correctly end-to-end.

`all_scraped_patents.csv` contains 30 dummy patents across 6 technology domains (NLP/patent search, image recognition, keyword search, legal documents, speech recognition, autonomous vehicles). These were created to test all pipeline stages, including cross-domain filtering in Stage 1 and multi-hop citation traversal in Stage 3.

**To use the system with real patents:**

1. Replace `all_scraped_patents.csv` with a real patent dataset. Real patents can be scraped or downloaded from:
   - [Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_data/patents) (BigQuery)
   - [USPTO PatentsView](https://patentsview.org/download/data-download-tables)
   - [EPO Open Patent Services API](https://developers.epo.org)
   - [Lens.org](https://lens.org) (bulk export available)

2. Replace `input.csv` with your invention details (one row, same column schema).

3. For corpora of 500+ patents, the pipeline automatically switches from cosine similarity to Mahalanobis distance in Stage 4 for improved retrieval accuracy.

---