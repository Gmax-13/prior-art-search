"""
Prior Art Search — Full Pipeline Runner
========================================
Runs all 4 stages in sequence, then calls the Groq API (Llama 3.3 70B)
for a final patent analysis report.

Usage:
    python run_pipeline.py

Requirements:
    pip install pandas numpy scikit-learn sentence-transformers groq

API Key:
    Get a free Groq API key from: https://console.groq.com
    Set it in the GROQ_API_KEY constant below.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # loads .env from the project directory

# HuggingFace settings — read from .env, fall back to sensible defaults
os.environ.setdefault("HF_HOME",                         os.getenv("HF_HOME", "D:/huggingface_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"))
os.environ.setdefault("HF_HUB_OFFLINE",                  os.getenv("HF_HUB_OFFLINE", "1"))

import sys
import time
import textwrap
import pandas as pd

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Import the four search modules (must be in the same directory)
# ---------------------------------------------------------------------------
from classification_search import classification_search
from keyword_search        import keyword_search
from citation_search       import citation_search
from semantic_search       import semantic_search
from pdf_report            import generate_pdf

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")  # set in .env
GROQ_MODEL    = "llama-3.3-70b-versatile"   # free, 131K context window

INPUT_CSV     = "input.csv"
PATENTS_CSV   = "all_scraped_patents.csv"

# All outputs go here — folder is created automatically on first run
OUTPUT_DIR    = "Output"

# Intermediate files (written between stages)
CLASSIFIED_CSV = f"{OUTPUT_DIR}/classified_patents.csv"
KEYWORD_CSV    = f"{OUTPUT_DIR}/keyword_filtered_patents.csv"
CITATION_CSV   = f"{OUTPUT_DIR}/citation_expanded_patents.csv"
FINAL_CSV      = f"{OUTPUT_DIR}/final_results.csv"
PDF_REPORT     = f"{OUTPUT_DIR}/patent_analysis_report.pdf"

# Stage parameters
KEYWORD_MIN_SCORE  = 3
KEYWORD_TOP_N      = 20
CITATION_MAX_DEPTH = 2
CITATION_TOP_N     = 40
SEMANTIC_TOP_K     = 20
SEMANTIC_TOP_N     = 5
SEMANTIC_ALPHA     = 0.7   # 0.7 semantic, 0.3 citation fusion weight


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

DIVIDER = "=" * 65

def header(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def stage_banner(n, name):
    print(f"\n{'─' * 65}")
    print(f"  STAGE {n} — {name}")
    print(f"{'─' * 65}")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_classification():
    stage_banner(1, "Classification Search")
    t = time.time()
    classification_search(
        input_csv   = INPUT_CSV,
        patents_csv = PATENTS_CSV,
        output_csv  = CLASSIFIED_CSV,
    )
    print(f"  ✓ Completed in {time.time()-t:.1f}s  →  {CLASSIFIED_CSV}")


def run_keyword():
    stage_banner(2, "Keyword Search")
    t = time.time()
    keyword_search(
        input_csv      = INPUT_CSV,
        classified_csv = CLASSIFIED_CSV,
        output_csv     = KEYWORD_CSV,
        min_score      = KEYWORD_MIN_SCORE,
        top_n          = KEYWORD_TOP_N,
    )
    print(f"  ✓ Completed in {time.time()-t:.1f}s  →  {KEYWORD_CSV}")


def run_citation():
    stage_banner(3, "Citation Search  (PageRank + BFS)")
    t = time.time()
    citation_search(
        keyword_csv  = KEYWORD_CSV,
        patents_csv  = PATENTS_CSV,
        output_csv   = CITATION_CSV,
        max_depth    = CITATION_MAX_DEPTH,
        top_n        = CITATION_TOP_N,
    )
    print(f"  ✓ Completed in {time.time()-t:.1f}s  →  {CITATION_CSV}")


def run_semantic():
    stage_banner(4, "Semantic Search  (Bi-encoder → Cross-encoder)")
    t = time.time()
    semantic_search(
        citation_csv    = CITATION_CSV,
        input_csv       = INPUT_CSV,
        all_patents_csv = PATENTS_CSV,
        output_csv      = FINAL_CSV,
        top_k_retrieval = SEMANTIC_TOP_K,
        top_n_final     = SEMANTIC_TOP_N,
        alpha           = SEMANTIC_ALPHA,
    )
    print(f"  ✓ Completed in {time.time()-t:.1f}s  →  {FINAL_CSV}")


# ---------------------------------------------------------------------------
# Final analysis — Groq API
# ---------------------------------------------------------------------------

def build_analysis_prompt(query_row, top_patents):
    query_title    = query_row.get("title",    "N/A")
    query_abstract = query_row.get("abstract", "N/A")
    query_keywords = query_row.get("keywords", "N/A")
    query_ipc      = query_row.get("ipc_codes","N/A")

    patents_block = ""
    for i, row in top_patents.iterrows():
        rank = i + 1
        patents_block += (
            f"\n  [{rank}] {row['patent_id']} — {row['title']}"
            f"  (Year: {row.get('publication_year','N/A')})\n"
            f"       Similarity Score : {row['final_score']:.4f}\n"
            f"       Abstract         : {row['abstract']}\n"
            f"       IPC Codes        : {row.get('ipc_codes','N/A')}\n"
        )

    prompt = (
        "You are a senior patent analyst with expertise in prior art evaluation.\n\n"
        "A researcher wants to know whether their invention is novel enough to apply for a patent.\n"
        f"Below is their invention description, followed by the top {len(top_patents)} most similar\n"
        "existing patents found by a multi-stage prior art search system (classification -> keyword\n"
        "-> citation network -> semantic similarity).\n\n"
        "RESEARCHER'S INVENTION\n"
        f"Title      : {query_title}\n"
        f"Abstract   : {query_abstract}\n"
        f"Keywords   : {query_keywords}\n"
        f"IPC Codes  : {query_ipc}\n\n"
        f"TOP {len(top_patents)} MOST SIMILAR PRIOR ART PATENTS\n"
        f"{patents_block}\n\n"
        "YOUR ANALYSIS TASK\n"
        "Provide a structured patent analysis report with the following sections.\n"
        "Be specific, technical, and direct. Do not use vague language.\n\n"
        "## 1. PATENT VIABILITY VERDICT\n"
        "State clearly: RECOMMENDED / CONDITIONAL / NOT RECOMMENDED\n"
        "Give a one-paragraph justification citing specific overlaps or gaps.\n\n"
        "## 2. SIMILARITY BREAKDOWN\n"
        f"For each of the top {len(top_patents)} prior art patents, write 2-3 sentences on:\n"
        "- What specifically overlaps with the researcher's invention\n"
        "- What is meaningfully different\n\n"
        "## 3. NOVELTY GAPS\n"
        "List the specific technical aspects of the researcher's invention that do NOT\n"
        "appear in any of the prior art patents. These are the strongest claims to\n"
        "build a patent application around.\n\n"
        "## 4. RISK AREAS\n"
        "List the specific technical aspects where prior art most closely overlaps.\n"
        "These would be the weakest claims and most likely to face rejection.\n\n"
        "## 5. FUTURE SCOPE\n"
        "Suggest 3-5 concrete technical directions the researcher could develop further\n"
        "that would clearly differentiate from existing prior art and strengthen\n"
        "the patent application or a continuation patent.\n\n"
        "## 6. RECOMMENDED NEXT STEPS\n"
        "Practical steps the researcher should take (e.g. consult a patent attorney,\n"
        "file a provisional, narrow claims to X aspect, etc.)"
    )

    return prompt.strip()


def call_groq(prompt):
    """
    Call Groq API with Llama 3.3 70B.
    Free tier: 14,400 requests/day, 131,072 token context window.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return (
            "[ANALYSIS SKIPPED]\n\n"
            "Groq API key not set. Get a free key from:\n"
            "https://console.groq.com\n\n"
            "Then set GROQ_API_KEY at the top of run_pipeline.py and re-run."
        )

    try:
        from groq import Groq
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 2048,
        )
        return response.choices[0].message.content

    except ImportError:
        return (
            "[API ERROR] groq package not installed.\n"
            "Run: pip install groq"
        )
    except Exception as e:
        return f"[API ERROR] {type(e).__name__}: {e}"


def run_analysis():
    stage_banner("✦", "Final Analysis")

    query_df  = pd.read_csv(INPUT_CSV)
    final_df  = pd.read_csv(FINAL_CSV)
    query_row = query_df.iloc[0]

    # ── Ranked results table ──────────────────────────────────────────────
    print("\n  TOP SIMILAR PATENTS\n")
    print(f"  {'Rank':<5} {'Patent ID':<10} {'Final Score':<14} {'Rerank':<10} "
          f"{'Retrieval':<12} {'Year':<6}  Title")
    print(f"  {'─'*4} {'─'*9} {'─'*13} {'─'*9} {'─'*11} {'─'*5}  {'─'*35}")

    for i, row in final_df.iterrows():
        rank  = i + 1
        title = str(row["title"])[:45]
        print(
            f"  {rank:<5} {str(row['patent_id']):<10} "
            f"{row['final_score']:<14.4f} {row['rerank_score']:<10.3f} "
            f"{row['retrieval_score']:<12.4f} {str(row.get('publication_year','N/A')):<6}  {title}"
        )

    # ── Score metadata ────────────────────────────────────────────────────
    meta = final_df.iloc[0]
    print(f"\n  Embedding backend : {meta.get('embed_backend','N/A')}")
    print(f"  Retrieval metric  : {meta.get('retrieval_metric','N/A')}")
    print(f"  Reranker backend  : {meta.get('rerank_backend','N/A')}")
    print(f"  Citation fusion α : {SEMANTIC_ALPHA}")

    # ── Groq analysis ─────────────────────────────────────────────────────
    print(f"\n  Calling Groq ({GROQ_MODEL}) for patent analysis report...")
    t       = time.time()
    prompt  = build_analysis_prompt(query_row, final_df.reset_index(drop=True))
    report  = call_groq(prompt)
    elapsed = time.time() - t

    print(f"  ✓ Analysis received in {elapsed:.1f}s\n")

    # ── Print report ──────────────────────────────────────────────────────
    header("PATENT ANALYSIS REPORT")
    print()
    for line in report.splitlines():
        if len(line) > 100:
            print(textwrap.fill(line, width=100, subsequent_indent="    "))
        else:
            print(line)

    # ── Save text report ──────────────────────────────────────────────────
    report_path = f"{OUTPUT_DIR}/patent_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PRIOR ART SEARCH — PATENT ANALYSIS REPORT\n")
        f.write(f"{'='*65}\n\n")
        f.write(f"Query Patent : {query_row.get('title','N/A')}\n\n")
        f.write(f"{'='*65}\n\n")
        f.write(report)
        f.write(f"\n\n{'='*65}\n")
        f.write("Generated by Prior Art Search Pipeline\n")

    print(f"\n  Text report saved  →  {report_path}")

    # ── Generate PDF report ───────────────────────────────────────────────
    print(f"  Generating PDF report...")
    generate_pdf(
        query_csv   = INPUT_CSV,
        final_csv   = FINAL_CSV,
        analysis    = report,
        output_path = PDF_REPORT,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header("PRIOR ART SEARCH PIPELINE")
    print(f"  Input patent : {pd.read_csv(INPUT_CSV).iloc[0].get('title','N/A')}")
    print(f"  Corpus       : {PATENTS_CSV}  ({len(pd.read_csv(PATENTS_CSV))} patents)")
    print(f"  LLM          : Groq / {GROQ_MODEL}")
    print(f"  Output dir   : {OUTPUT_DIR}/")

    total_start = time.time()

    try:
        run_classification()
        run_keyword()
        run_citation()
        run_semantic()
        run_analysis()
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing file: {e}")
        print("Make sure input.csv and all_scraped_patents.csv are in the same directory.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Pipeline failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    header(f"PIPELINE COMPLETE  ({time.time()-total_start:.1f}s total)")
    print(f"  Classified patents : {CLASSIFIED_CSV}")
    print(f"  Keyword filtered   : {KEYWORD_CSV}")
    print(f"  Citation expanded  : {CITATION_CSV}")
    print(f"  Final results      : {FINAL_CSV}")
    print(f"  Text report        : {OUTPUT_DIR}/patent_analysis_report.txt")
    print(f"  PDF report         : {PDF_REPORT}")
    print()


if __name__ == "__main__":
    run_pipeline()