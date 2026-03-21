"""
pdf_report.py — Prior Art Search PDF Report Generator
======================================================
Generates a professionally formatted PDF report from the pipeline results
and the Gemini analysis text.

Called by run_pipeline.py — do not run directly.

Requires:
    pip install reportlab
"""

import re
from datetime import datetime

import pandas as pd
from reportlab.lib              import colors
from reportlab.lib.enums        import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes    import A4
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units        import mm
from reportlab.platypus         import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY        = colors.HexColor("#0D1B2A")
ACCENT      = colors.HexColor("#1A6B9A")
ACCENT_LITE = colors.HexColor("#D6EAF8")
GOLD        = colors.HexColor("#C8990A")
LIGHT_GREY  = colors.HexColor("#F4F6F8")
MID_GREY    = colors.HexColor("#BDC3C7")
DARK_GREY   = colors.HexColor("#4A4A4A")
WHITE       = colors.white
GREEN       = colors.HexColor("#1E8449")
AMBER       = colors.HexColor("#B7770D")
RED         = colors.HexColor("#922B21")

PAGE_W, PAGE_H = A4
MARGIN         = 18 * mm


# ---------------------------------------------------------------------------
# Style sheet
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": S("cover_title",
            fontName="Helvetica-Bold", fontSize=26,
            textColor=WHITE, alignment=TA_CENTER, leading=32,
            spaceAfter=6),

        "cover_sub": S("cover_sub",
            fontName="Helvetica", fontSize=12,
            textColor=ACCENT_LITE, alignment=TA_CENTER, leading=16),

        "cover_meta": S("cover_meta",
            fontName="Helvetica", fontSize=10,
            textColor=MID_GREY, alignment=TA_CENTER, leading=14),

        "h1": S("h1",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=WHITE, alignment=TA_LEFT,
            spaceBefore=14, spaceAfter=6, leading=16),

        "h2": S("h2",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=ACCENT, alignment=TA_LEFT,
            spaceBefore=10, spaceAfter=4, leading=14),

        "h3": S("h3",
            fontName="Helvetica-BoldOblique", fontSize=10,
            textColor=DARK_GREY, alignment=TA_LEFT,
            spaceBefore=7, spaceAfter=3, leading=13),

        "body": S("body",
            fontName="Helvetica", fontSize=9.5,
            textColor=DARK_GREY, alignment=TA_JUSTIFY,
            leading=14, spaceAfter=4),

        "bullet": S("bullet",
            fontName="Helvetica", fontSize=9.5,
            textColor=DARK_GREY, alignment=TA_LEFT,
            leading=14, leftIndent=14, spaceAfter=3,
            bulletIndent=4),

        "caption": S("caption",
            fontName="Helvetica-Oblique", fontSize=8.5,
            textColor=MID_GREY, alignment=TA_CENTER,
            spaceAfter=6),

        "verdict_rec": S("verdict_rec",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=GREEN, alignment=TA_CENTER, leading=18),

        "verdict_cond": S("verdict_cond",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=AMBER, alignment=TA_CENTER, leading=18),

        "verdict_no": S("verdict_no",
            fontName="Helvetica-Bold", fontSize=13,
            textColor=RED, alignment=TA_CENTER, leading=18),

        "score_label": S("score_label",
            fontName="Helvetica-Bold", fontSize=8,
            textColor=ACCENT, alignment=TA_CENTER),

        "score_value": S("score_value",
            fontName="Helvetica-Bold", fontSize=11,
            textColor=NAVY, alignment=TA_CENTER),

        "table_header": S("table_header",
            fontName="Helvetica-Bold", fontSize=8.5,
            textColor=WHITE, alignment=TA_CENTER),

        "table_cell": S("table_cell",
            fontName="Helvetica", fontSize=8.5,
            textColor=DARK_GREY, alignment=TA_LEFT, leading=12),

        "table_cell_c": S("table_cell_c",
            fontName="Helvetica", fontSize=8.5,
            textColor=DARK_GREY, alignment=TA_CENTER, leading=12),

        "footer": S("footer",
            fontName="Helvetica", fontSize=7.5,
            textColor=MID_GREY, alignment=TA_CENTER),
    }


# ---------------------------------------------------------------------------
# Page templates (header bar + footer)
# ---------------------------------------------------------------------------

def make_page_templates(doc, styles):
    """Cover page (no header) + body pages (navy top bar + footer)."""

    def cover_bg(canvas, doc):
        canvas.saveState()
        # Full navy background
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # Gold accent strip at top
        canvas.setFillColor(GOLD)
        canvas.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, fill=1, stroke=0)
        # Accent strip at bottom
        canvas.setFillColor(ACCENT)
        canvas.rect(0, 0, PAGE_W, 6 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def body_page(canvas, doc):
        canvas.saveState()
        # Top bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)
        canvas.setFillColor(GOLD)
        canvas.rect(0, PAGE_H - 2 * mm, PAGE_W, 2 * mm, fill=1, stroke=0)
        # Header text
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(WHITE)
        canvas.drawString(MARGIN, PAGE_H - 9 * mm, "PRIOR ART SEARCH — PATENT ANALYSIS REPORT")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GREY)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 9 * mm,
                               datetime.now().strftime("%d %B %Y"))
        # Footer
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, PAGE_W, 10 * mm, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, 10 * mm, PAGE_W, 0.5 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(DARK_GREY)
        canvas.drawCentredString(PAGE_W / 2, 3.5 * mm,
                                 f"Page {doc.page}  |  Prior Art Search Pipeline  |  Confidential")
        canvas.restoreState()

    cover_frame = Frame(MARGIN, 30 * mm, PAGE_W - 2 * MARGIN,
                        PAGE_H - 50 * mm, id="cover")
    body_frame  = Frame(MARGIN, 14 * mm, PAGE_W - 2 * MARGIN,
                        PAGE_H - 28 * mm, id="body")

    return [
        PageTemplate(id="Cover", frames=[cover_frame], onPage=cover_bg),
        PageTemplate(id="Body",  frames=[body_frame],  onPage=body_page),
    ]


# ---------------------------------------------------------------------------
# Section header helper
# ---------------------------------------------------------------------------

def section_header(title: str, styles) -> list:
    """Navy pill background section header."""
    tbl = Table([[Paragraph(title, styles["h1"])]],
                colWidths=[PAGE_W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [NAVY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [3]),
    ]))
    return [Spacer(1, 4), tbl, Spacer(1, 4)]


# ---------------------------------------------------------------------------
# Parse Gemini markdown into flowables
# ---------------------------------------------------------------------------

def parse_analysis(text: str, styles) -> list:
    """
    Convert the Gemini response (markdown-ish) into reportlab flowables.
    Handles: ## headings, bullet lines (- / * / •), bold (**text**), body text.
    """
    story = []
    lines = text.splitlines()
    i     = 0

    def md_to_rl(s: str) -> str:
        """Convert **bold** and *italic* to ReportLab XML tags."""
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", s)
        # Escape raw ampersands not already part of an entity
        s = re.sub(r"&(?!amp;|lt;|gt;|#)", "&amp;", s)
        return s

    while i < len(lines):
        line = lines[i].rstrip()

        # Skip blank lines (add small space)
        if not line.strip():
            story.append(Spacer(1, 3))
            i += 1
            continue

        # ## Section heading
        if line.startswith("## "):
            heading = line[3:].strip()
            story.extend(section_header(heading, styles))
            i += 1
            continue

        # ### Sub-heading
        if line.startswith("### "):
            story.append(Paragraph(md_to_rl(line[4:].strip()), styles["h2"]))
            i += 1
            continue

        # Bullet point
        if re.match(r"^[\-\*•]\s+", line):
            content = re.sub(r"^[\-\*•]\s+", "", line)
            story.append(Paragraph(f"• {md_to_rl(content)}", styles["bullet"]))
            i += 1
            continue

        # Numbered list  (1. 2. etc.)
        if re.match(r"^\d+\.\s+", line):
            content = re.sub(r"^\d+\.\s+", "", line)
            story.append(Paragraph(f"• {md_to_rl(content)}", styles["bullet"]))
            i += 1
            continue

        # Verdict line — colour-code it
        upper = line.upper()
        if "RECOMMENDED" in upper and "NOT" not in upper and "CONDITIONAL" not in upper:
            story.append(Paragraph(md_to_rl(line), styles["verdict_rec"]))
            i += 1
            continue
        if "CONDITIONAL" in upper:
            story.append(Paragraph(md_to_rl(line), styles["verdict_cond"]))
            i += 1
            continue
        if "NOT RECOMMENDED" in upper:
            story.append(Paragraph(md_to_rl(line), styles["verdict_no"]))
            i += 1
            continue

        # Plain body text
        story.append(Paragraph(md_to_rl(line), styles["body"]))
        i += 1

    return story


# ---------------------------------------------------------------------------
# Score summary cards
# ---------------------------------------------------------------------------

def score_summary_table(final_df: pd.DataFrame, styles) -> list:
    """Compact score metadata row."""
    meta = final_df.iloc[0]
    cards = [
        ("Embedding",  str(meta.get("embed_backend",   "N/A")).upper()),
        ("Retrieval",  str(meta.get("retrieval_metric","N/A")).upper()),
        ("Reranker",   str(meta.get("rerank_backend",  "N/A")).upper()),
        ("Top Results",str(len(final_df))),
    ]
    cell_w = (PAGE_W - 2 * MARGIN) / len(cards)
    header_row = [Paragraph(c[0], styles["score_label"]) for c in cards]
    value_row  = [Paragraph(c[1], styles["score_value"]) for c in cards]

    tbl = Table([header_row, value_row],
                colWidths=[cell_w] * len(cards),
                rowHeights=[12, 20])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), ACCENT_LITE),
        ("LINEABOVE",     (0, 0), (-1, 0),  0.5, ACCENT),
        ("LINEBELOW",     (0, -1),(-1, -1), 0.5, ACCENT),
        ("LINEBEFORE",    (0, 0), (0, -1),  0.5, ACCENT),
        ("LINEAFTER",     (-1,0), (-1, -1), 0.5, ACCENT),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [tbl, Spacer(1, 8)]


# ---------------------------------------------------------------------------
# Top patents ranked table
# ---------------------------------------------------------------------------

def ranked_patents_table(final_df: pd.DataFrame, styles) -> list:
    col_widths = [10*mm, 16*mm, 22*mm, 22*mm, 22*mm, 14*mm, 60*mm]
    headers    = ["Rank", "Patent ID", "Final Score", "Rerank Score",
                  "Retrieval", "Year", "Title"]

    header_row = [Paragraph(h, styles["table_header"]) for h in headers]
    rows       = [header_row]

    for i, row in final_df.iterrows():
        rank  = i + 1
        title = str(row.get("title", ""))
        # Truncate long titles
        if len(title) > 55:
            title = title[:52] + "…"
        rows.append([
            Paragraph(str(rank),                          styles["table_cell_c"]),
            Paragraph(str(row.get("patent_id", "")),      styles["table_cell_c"]),
            Paragraph(f"{row['final_score']:.4f}",        styles["table_cell_c"]),
            Paragraph(f"{row['rerank_score']:.3f}",       styles["table_cell_c"]),
            Paragraph(f"{row['retrieval_score']:.4f}",    styles["table_cell_c"]),
            Paragraph(str(row.get("publication_year","")),styles["table_cell_c"]),
            Paragraph(title,                              styles["table_cell"]),
        ])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)

    row_colors = []
    for r in range(1, len(rows)):
        bg = LIGHT_GREY if r % 2 == 0 else WHITE
        row_colors.append(("ROWBACKGROUNDS", (0, r), (-1, r), [bg]))

    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8.5),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        # Borders
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("LINEBELOW",     (0, 0), (-1, 0),  1,   ACCENT),
        # Padding
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        *row_colors,
    ]))

    return [tbl, Spacer(1, 6)]


# ---------------------------------------------------------------------------
# Per-patent detail cards
# ---------------------------------------------------------------------------

def patent_detail_cards(final_df: pd.DataFrame, styles) -> list:
    story = []
    bar_w = PAGE_W - 2 * MARGIN

    for i, row in final_df.iterrows():
        rank    = i + 1
        title   = str(row.get("title",    "N/A"))
        pid     = str(row.get("patent_id","N/A"))
        year    = str(row.get("publication_year", "N/A"))
        abstract= str(row.get("abstract", "N/A"))
        ipc     = str(row.get("ipc_codes","N/A"))

        # Title bar
        title_tbl = Table(
            [[Paragraph(f"#{rank}  {pid} — {title}", styles["h2"])]],
            colWidths=[bar_w]
        )
        title_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), ACCENT_LITE),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LINEBELOW",     (0,0),(-1,-1), 1, ACCENT),
        ]))

        # Score row
        score_data = [
            [
                Paragraph("Final Score",    styles["score_label"]),
                Paragraph("Rerank Score",   styles["score_label"]),
                Paragraph("Retrieval",      styles["score_label"]),
                Paragraph("Citation Score", styles["score_label"]),
                Paragraph("PageRank",       styles["score_label"]),
                Paragraph("Year",           styles["score_label"]),
            ],
            [
                Paragraph(f"{row['final_score']:.4f}",      styles["score_value"]),
                Paragraph(f"{row['rerank_score']:.3f}",     styles["score_value"]),
                Paragraph(f"{row['retrieval_score']:.4f}",  styles["score_value"]),
                Paragraph(f"{row['citation_score']:.4f}",   styles["score_value"]),
                Paragraph(f"{row['pagerank_score']:.6f}",   styles["score_value"]),
                Paragraph(year,                             styles["score_value"]),
            ]
        ]
        score_tbl = Table(score_data, colWidths=[bar_w/6]*6, rowHeights=[12, 18])
        score_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), LIGHT_GREY),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, MID_GREY),
            ("BOX",           (0,0),(-1,-1), 0.5, MID_GREY),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))

        story.append(KeepTogether([
            title_tbl,
            Spacer(1, 3),
            score_tbl,
            Spacer(1, 4),
            Paragraph(f"<b>IPC/CPC:</b> {ipc}", styles["body"]),
            Paragraph(f"<b>Abstract:</b> {abstract}", styles["body"]),
            HRFlowable(width="100%", thickness=0.5, color=MID_GREY,
                       spaceAfter=6, spaceBefore=4),
        ]))

    return story


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def cover_page(query_row: pd.Series, styles) -> list:
    title    = str(query_row.get("title",    "Patent Analysis Report"))
    abstract = str(query_row.get("abstract", ""))
    keywords = str(query_row.get("keywords", ""))
    ipc      = str(query_row.get("ipc_codes",""))
    now      = datetime.now().strftime("%d %B %Y  |  %H:%M")

    return [
        Spacer(1, 30 * mm),
        Paragraph("PRIOR ART SEARCH", ParagraphStyle(
            "strap", fontName="Helvetica", fontSize=11,
            textColor=GOLD, alignment=TA_CENTER, spaceAfter=6, letterSpacing=3)),
        Paragraph("Patent Analysis Report", styles["cover_title"]),
        Spacer(1, 6 * mm),
        HRFlowable(width="60%", thickness=1.5, color=GOLD,
                   hAlign="CENTER", spaceAfter=8 * mm),
        Paragraph(f"<b>Invention:</b>  {title}", styles["cover_sub"]),
        Spacer(1, 4 * mm),
        Paragraph(abstract, ParagraphStyle(
            "cov_abs", fontName="Helvetica-Oblique", fontSize=10,
            textColor=ACCENT_LITE, alignment=TA_CENTER, leading=15)),
        Spacer(1, 6 * mm),
        Paragraph(f"Keywords: {keywords}  |  IPC: {ipc}", styles["cover_meta"]),
        Spacer(1, 4 * mm),
        Paragraph(f"Generated: {now}", styles["cover_meta"]),
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_pdf(
    query_csv:   str,
    final_csv:   str,
    analysis:    str,
    output_path: str,
):
    """
    Build and save the PDF report.

    Args:
        query_csv   : path to input.csv  (the user's invention)
        final_csv   : path to final_results.csv  (Stage 4 output)
        analysis    : raw Gemini analysis string
        output_path : destination PDF path
    """
    query_row = pd.read_csv(query_csv).iloc[0]
    final_df  = pd.read_csv(final_csv).reset_index(drop=True)
    styles    = build_styles()

    # Document
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="Prior Art Search — Patent Analysis Report",
        author="Prior Art Search Pipeline",
    )
    doc.addPageTemplates(make_page_templates(doc, styles))

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.extend(cover_page(query_row, styles))

    # Switch to body template from page 2 onward
    from reportlab.platypus import NextPageTemplate
    story.append(NextPageTemplate("Body"))

    # ── Section 1: Pipeline Metadata ──────────────────────────────────────
    story.extend(section_header("1.  PIPELINE METADATA", styles))
    meta_data = [
        ["Field", "Value"],
        ["Invention Title",   str(query_row.get("title",    "N/A"))],
        ["IPC / CPC Codes",   str(query_row.get("ipc_codes","N/A"))],
        ["Keywords",          str(query_row.get("keywords", "N/A"))],
        ["Embedding Backend", str(final_df.iloc[0].get("embed_backend",   "N/A"))],
        ["Retrieval Metric",  str(final_df.iloc[0].get("retrieval_metric","N/A"))],
        ["Reranker Backend",  str(final_df.iloc[0].get("rerank_backend",  "N/A"))],
        ["Report Generated",  datetime.now().strftime("%d %B %Y, %H:%M:%S")],
    ]
    meta_tbl = Table(
        [[Paragraph(r[0], styles["table_header"] if i == 0 else styles["h3"]),
          Paragraph(r[1], styles["table_header"] if i == 0 else styles["body"])]
         for i, r in enumerate(meta_data)],
        colWidths=[(PAGE_W - 2*MARGIN)*0.35, (PAGE_W - 2*MARGIN)*0.65],
    )
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("BACKGROUND",    (0, 1), (0, -1),  ACCENT_LITE),
        ("GRID",          (0, 0), (-1, -1), 0.4, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 8))

    # ── Section 2: Score summary ───────────────────────────────────────────
    story.extend(section_header("2.  PIPELINE SCORE SUMMARY", styles))
    story.extend(score_summary_table(final_df, styles))

    # ── Section 3: Ranked results table ───────────────────────────────────
    story.extend(section_header("3.  TOP SIMILAR PATENTS — RANKED", styles))
    story.extend(ranked_patents_table(final_df, styles))

    # ── Section 4: Per-patent detail ──────────────────────────────────────
    story.extend(section_header("4.  PATENT DETAIL CARDS", styles))
    story.extend(patent_detail_cards(final_df, styles))

    # ── Section 5: AI Analysis ────────────────────────────────────────────
    story.append(PageBreak())
    story.extend(section_header("5.  AI PATENT ANALYSIS", styles))

    if analysis.startswith("[ANALYSIS SKIPPED]"):
        story.append(Paragraph(
            "Gemini API key not configured. Set GEMINI_API_KEY in run_pipeline.py "
            "to enable the AI analysis section.",
            styles["body"]))
    else:
        story.extend(parse_analysis(analysis, styles))

    # Build
    doc.build(story)
    print(f"  ✓ PDF report saved → {output_path}")