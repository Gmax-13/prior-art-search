"""
Flask API Backend for Prior Art Search Pipeline
=================================================
Wraps the existing CLI pipeline as a REST API with real-time
stage-by-stage progress tracking.
"""

import os
import sys
import json
import time
import threading
import csv
import io
import traceback

# Add parent directory to path so we can import the pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "D:/huggingface_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"))
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV      = os.path.join(BASE_DIR, "input.csv")
PATENTS_CSV    = os.path.join(BASE_DIR, "all_scrapped_patents.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "Output")
CLASSIFIED_CSV = os.path.join(OUTPUT_DIR, "classified_patents.csv")
KEYWORD_CSV    = os.path.join(OUTPUT_DIR, "keyword_filtered_patents.csv")
CITATION_CSV   = os.path.join(OUTPUT_DIR, "citation_expanded_patents.csv")
FINAL_CSV      = os.path.join(OUTPUT_DIR, "output.csv")
REPORT_TXT     = os.path.join(OUTPUT_DIR, "patent_analysis_report.txt")
REPORT_PDF     = os.path.join(OUTPUT_DIR, "patent_analysis_report.pdf")

# Pipeline parameters
KEYWORD_MIN_SCORE  = 3
KEYWORD_TOP_N      = 20
CITATION_MAX_DEPTH = 2
CITATION_TOP_N     = 40
SEMANTIC_TOP_K     = 20
SEMANTIC_TOP_N     = 5
SEMANTIC_ALPHA     = 0.7

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
pipeline_state = {
    "status": "idle",       # idle | running | completed | error
    "current_stage": None,  # 1, 2, 3, 4, "analysis", None
    "stage_name": "",
    "stages_completed": [],
    "error": None,
    "started_at": None,
    "completed_at": None,
    "report": None,
}
pipeline_lock = threading.Lock()


def reset_state():
    pipeline_state.update({
        "status": "idle",
        "current_stage": None,
        "stage_name": "",
        "stages_completed": [],
        "error": None,
        "started_at": None,
        "completed_at": None,
        "report": None,
    })


def set_stage(stage_num, name):
    pipeline_state["current_stage"] = stage_num
    pipeline_state["stage_name"] = name


def complete_stage(stage_num):
    if stage_num not in pipeline_state["stages_completed"]:
        pipeline_state["stages_completed"].append(stage_num)


# ---------------------------------------------------------------------------
# Pipeline runner (runs in background thread)
# ---------------------------------------------------------------------------
def run_pipeline_thread():
    try:
        from classification_search import classification_search
        from keyword_search import keyword_search
        from citation_search import citation_search
        from semantic_search import semantic_search
        from pdf_report import generate_pdf

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pipeline_state["status"] = "running"
        pipeline_state["started_at"] = time.time()

        # Stage 1
        set_stage(1, "Classification Search")
        classification_search(
            input_csv=INPUT_CSV,
            patents_csv=PATENTS_CSV,
            output_csv=CLASSIFIED_CSV,
        )
        complete_stage(1)

        # Stage 2
        set_stage(2, "Keyword Search")
        keyword_search(
            input_csv=INPUT_CSV,
            classified_csv=CLASSIFIED_CSV,
            output_csv=KEYWORD_CSV,
            min_score=KEYWORD_MIN_SCORE,
            top_n=KEYWORD_TOP_N,
        )
        complete_stage(2)

        # Stage 3
        set_stage(3, "Citation Search (PageRank + BFS)")
        citation_search(
            keyword_csv=KEYWORD_CSV,
            patents_csv=PATENTS_CSV,
            output_csv=CITATION_CSV,
            max_depth=CITATION_MAX_DEPTH,
            top_n=CITATION_TOP_N,
        )
        complete_stage(3)

        # Stage 4
        set_stage(4, "Semantic Search (Bi-encoder → Cross-encoder)")
        semantic_search(
            citation_csv=CITATION_CSV,
            input_csv=INPUT_CSV,
            all_patents_csv=PATENTS_CSV,
            output_csv=FINAL_CSV,
            top_k_retrieval=SEMANTIC_TOP_K,
            top_n_final=SEMANTIC_TOP_N,
            alpha=SEMANTIC_ALPHA,
        )
        complete_stage(4)

        # Stage 5: Analysis
        set_stage(5, "AI Analysis (Groq / Llama 3.3 70B)")

        import textwrap
        query_df = pd.read_csv(INPUT_CSV)
        final_df = pd.read_csv(FINAL_CSV)
        query_row = query_df.iloc[0]

        # Build prompt and call Groq
        from run_pipeline import build_analysis_prompt, call_groq
        prompt = build_analysis_prompt(query_row, final_df.reset_index(drop=True))
        report = call_groq(prompt)

        # Save text report
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write("PRIOR ART SEARCH — PATENT ANALYSIS REPORT\n")
            f.write(f"{'=' * 65}\n\n")
            f.write(f"Query Patent : {query_row.get('title', 'N/A')}\n\n")
            f.write(f"{'=' * 65}\n\n")
            f.write(report)
            f.write(f"\n\n{'=' * 65}\n")
            f.write("Generated by Prior Art Search Pipeline\n")

        # Generate PDF
        generate_pdf(
            query_csv=INPUT_CSV,
            final_csv=FINAL_CSV,
            analysis=report,
            output_path=REPORT_PDF,
        )

        complete_stage(5)
        pipeline_state["status"] = "completed"
        pipeline_state["completed_at"] = time.time()
        pipeline_state["report"] = report

    except Exception as e:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = f"{type(e).__name__}: {e}"
        pipeline_state["completed_at"] = time.time()
        traceback.print_exc()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def upload_input():
    """Accept input CSV and trigger pipeline."""
    with pipeline_lock:
        if pipeline_state["status"] == "running":
            return jsonify({"error": "Pipeline is already running"}), 409

    data = request.json
    if not data or "csv_content" not in data:
        return jsonify({"error": "Missing csv_content in request body"}), 400

    csv_content = data["csv_content"].strip()
    if not csv_content:
        return jsonify({"error": "CSV content is empty"}), 400

    # Save to input.csv
    with open(INPUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(csv_content)
        if not csv_content.endswith("\n"):
            f.write("\n")

    # Reset and start pipeline
    reset_state()
    thread = threading.Thread(target=run_pipeline_thread, daemon=True)
    thread.start()

    return jsonify({"message": "Pipeline started", "status": "running"})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return current pipeline status with stage-by-stage progress."""
    elapsed = None
    if pipeline_state["started_at"]:
        end = pipeline_state["completed_at"] or time.time()
        elapsed = round(end - pipeline_state["started_at"], 1)

    return jsonify({
        "status": pipeline_state["status"],
        "current_stage": pipeline_state["current_stage"],
        "stage_name": pipeline_state["stage_name"],
        "stages_completed": pipeline_state["stages_completed"],
        "error": pipeline_state["error"],
        "elapsed_seconds": elapsed,
    })


@app.route("/api/report", methods=["GET"])
def get_report():
    """Return the text report content."""
    if not os.path.exists(REPORT_TXT):
        return jsonify({"error": "Report not yet generated"}), 404

    with open(REPORT_TXT, "r", encoding="utf-8") as f:
        content = f.read()

    return jsonify({"report": content})


@app.route("/api/report/pdf", methods=["GET"])
def get_report_pdf():
    """Serve the PDF report for download."""
    if not os.path.exists(REPORT_PDF):
        return jsonify({"error": "PDF report not yet generated"}), 404

    return send_file(
        REPORT_PDF,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="patent_analysis_report.pdf",
    )


@app.route("/api/outputs", methods=["GET"])
def list_outputs():
    """List all available output CSV files."""
    files = []
    output_map = {
        "classified_patents.csv": {"stage": 1, "name": "Classified Patents"},
        "keyword_filtered_patents.csv": {"stage": 2, "name": "Keyword Filtered Patents"},
        "citation_expanded_patents.csv": {"stage": 3, "name": "Citation Expanded Patents"},
        "output.csv": {"stage": 4, "name": "Final Results"},
    }

    for filename, meta in output_map.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                files.append({
                    "filename": filename,
                    "stage": meta["stage"],
                    "name": meta["name"],
                    "rows": len(df),
                    "columns": list(df.columns),
                })
            except Exception:
                pass

    return jsonify({"files": files})


@app.route("/api/outputs/<filename>", methods=["GET"])
def get_output(filename):
    """Return parsed CSV as JSON."""
    allowed = [
        "classified_patents.csv",
        "keyword_filtered_patents.csv",
        "citation_expanded_patents.csv",
        "output.csv",
    ]
    if filename not in allowed:
        return jsonify({"error": "File not found"}), 404

    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not yet generated"}), 404

    try:
        df = pd.read_csv(filepath)
        # Limit columns for large files
        display_cols = [c for c in df.columns if c not in ["cites", "cited_by", "cpc_codes"]]
        df_display = df[display_cols].copy()
        # Truncate abstract for display
        if "abstract" in df_display.columns:
            df_display["abstract"] = df_display["abstract"].astype(str).str[:200] + "..."

        return jsonify({
            "filename": filename,
            "total_rows": len(df),
            "columns": list(df_display.columns),
            "data": df_display.fillna("").to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/insights", methods=["GET"])
def get_insights():
    """Return comprehensive chart data from all output files."""
    insights = {}

    # ── Final results data ────────────────────────────────────────────
    if os.path.exists(FINAL_CSV):
        try:
            df = pd.read_csv(FINAL_CSV)
            insights["final_scores"] = [
                {
                    "patent_id": str(row["patent_id"])[:15],
                    "patent_id_full": str(row["patent_id"]),
                    "title": str(row.get("title", ""))[:50],
                    "final_score": round(row["final_score"], 4),
                    "rerank_score": round(row["rerank_score"], 3),
                    "retrieval_score": round(row["retrieval_score"], 4),
                    "citation_score": round(row.get("citation_score", 0), 4),
                    "pagerank_score": round(row.get("pagerank_score", 0), 6),
                    "forward_citations": int(row.get("forward_citations", 0)),
                    "backward_citations": int(row.get("backward_citations", 0)),
                    "hop_distance": int(row.get("hop_distance", 0)),
                    "is_seed": bool(row.get("is_seed", False)),
                    "publication_year": int(row["publication_year"]) if pd.notna(row.get("publication_year")) else None,
                }
                for _, row in df.iterrows()
            ]

            # Pipeline metadata
            meta = df.iloc[0]
            insights["pipeline_meta"] = {
                "embed_backend": str(meta.get("embed_backend", "N/A")),
                "retrieval_metric": str(meta.get("retrieval_metric", "N/A")),
                "rerank_backend": str(meta.get("rerank_backend", "N/A")),
                "alpha": SEMANTIC_ALPHA,
                "top_n": len(df),
            }
        except Exception:
            pass

    # ── Pipeline funnel ───────────────────────────────────────────────
    funnel = []
    stage_files = [
        (PATENTS_CSV, "Corpus"),
        (CLASSIFIED_CSV, "Classified"),
        (KEYWORD_CSV, "Keyword Filtered"),
        (CITATION_CSV, "Citation Expanded"),
        (FINAL_CSV, "Final Results"),
    ]
    for fpath, label in stage_files:
        if os.path.exists(fpath):
            try:
                funnel.append({"stage": label, "count": len(pd.read_csv(fpath))})
            except Exception:
                pass
    insights["funnel"] = funnel

    # ── Citation stage data ───────────────────────────────────────────
    if os.path.exists(CITATION_CSV):
        try:
            cdf = pd.read_csv(CITATION_CSV)

            # Hop distance distribution
            if "hop_distance" in cdf.columns:
                hop_counts = cdf["hop_distance"].value_counts().sort_index()
                insights["hop_distribution"] = [
                    {"hop": f"Hop {int(h)}" if h > 0 else "Seed (Hop 0)", "count": int(c)}
                    for h, c in hop_counts.items()
                ]

            # Seed vs discovered
            if "is_seed" in cdf.columns:
                seed_count = int(cdf["is_seed"].sum())
                discovered_count = len(cdf) - seed_count
                insights["seed_vs_discovered"] = [
                    {"type": "Seed Patents", "count": seed_count},
                    {"type": "BFS Discovered", "count": discovered_count},
                ]

            # Forward vs Backward citations scatter data
            if "forward_citations" in cdf.columns and "backward_citations" in cdf.columns:
                insights["citation_scatter"] = [
                    {
                        "patent_id": str(row.get("patent_id", ""))[:15],
                        "forward": int(row.get("forward_citations", 0)),
                        "backward": int(row.get("backward_citations", 0)),
                        "citation_score": round(row.get("citation_score", 0), 4),
                    }
                    for _, row in cdf.head(30).iterrows()
                ]
        except Exception:
            pass

    # ── Keyword stage data ────────────────────────────────────────────
    if os.path.exists(KEYWORD_CSV):
        try:
            kdf = pd.read_csv(KEYWORD_CSV)
            if "keyword_score" in kdf.columns:
                insights["keyword_scores"] = [
                    {
                        "patent_id": str(row.get("patent_id", ""))[:15],
                        "title": str(row.get("title", ""))[:40],
                        "keyword_score": int(row["keyword_score"]),
                    }
                    for _, row in kdf.sort_values("keyword_score", ascending=False).iterrows()
                ]
        except Exception:
            pass

    # ── Classified stage — IPC code distribution ──────────────────────
    if os.path.exists(CLASSIFIED_CSV):
        try:
            clf = pd.read_csv(CLASSIFIED_CSV)
            if "ipc_codes" in clf.columns:
                # Extract top-level IPC sections (first 4 chars like "G06F", "H01L")
                all_codes = []
                for codes_str in clf["ipc_codes"].dropna():
                    for code in str(codes_str).split(";"):
                        code = code.strip()
                        if len(code) >= 4:
                            all_codes.append(code[:4])
                if all_codes:
                    code_series = pd.Series(all_codes)
                    code_counts = code_series.value_counts().head(15)
                    insights["ipc_distribution"] = [
                        {"code": str(code), "count": int(cnt)}
                        for code, cnt in code_counts.items()
                    ]
        except Exception:
            pass

    return jsonify(insights)


@app.route("/api/input", methods=["GET"])
def get_input():
    """Return the current input.csv content."""
    if not os.path.exists(INPUT_CSV):
        return jsonify({"error": "No input file found"}), 404

    try:
        df = pd.read_csv(INPUT_CSV)
        row = df.iloc[0]
        return jsonify({
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
            "ipc_codes": str(row.get("ipc_codes", "")),
            "cpc_codes": str(row.get("cpc_codes", "")),
            "keywords": str(row.get("keywords", "")),
            "citations": str(row.get("citations", "")),
            "publication_year": str(row.get("publication_year", "")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
