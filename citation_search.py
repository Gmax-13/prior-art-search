import pandas as pd
import numpy as np
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Null-safe helpers
# ---------------------------------------------------------------------------

def safe_str(val) -> str:
    """Return empty string for any null/NaN/None value."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def parse_ids(raw, delimiter=";") -> list:
    """
    Split a delimited ID string into a clean list.
    Handles nulls, extra whitespace, empty tokens, and 'nan' strings.
    """
    s = safe_str(raw)
    if not s:
        return []
    return [tok.strip() for tok in s.split(delimiter)
            if tok.strip() and tok.strip().lower() != "nan"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_citation_graph(patents_df,
                          cites_column    = "cites",
                          cited_by_column = "cited_by"):
    """
    Build a bidirectional citation graph from the new schema.

    New schema has two explicit columns:
      cites    -- patents this patent cites       (backward edges)
      cited_by -- patents that cite this patent   (forward edges)

    Both are semicolon-separated. Either or both may be null.

    Returns:
        backward: dict[patent_id -> list of patents it cites]
        forward:  dict[patent_id -> list of patents that cite it]
    """
    backward = defaultdict(list)
    forward  = defaultdict(list)

    for _, row in patents_df.iterrows():
        src = safe_str(row.get("patent_id", ""))
        if not src:
            continue

        # Backward edges -- from cites column
        for tgt in parse_ids(row.get(cites_column, "")):
            if tgt and tgt != src:
                backward[src].append(tgt)
                forward[tgt].append(src)   # reciprocal forward edge

        # Forward edges -- from cited_by column
        for citer in parse_ids(row.get(cited_by_column, "")):
            if citer and citer != src:
                if src not in forward[citer]:
                    forward[citer].append(src)
                if citer not in backward[src]:
                    backward[src].append(citer)

    return dict(backward), dict(forward)


# ---------------------------------------------------------------------------
# PageRank on the citation graph
# ---------------------------------------------------------------------------

def compute_pagerank(all_ids, forward, damping=0.85, iterations=50):
    """
    Standard PageRank over the citation graph.
    A patent cited by high-PageRank patents scores higher than one cited
    by obscure patents -- key advantage over raw citation counts.
    """
    n = len(all_ids)
    if n == 0:
        return {}

    rank       = {pid: 1.0 / n for pid in all_ids}
    out_degree = {pid: len(forward.get(pid, [])) for pid in all_ids}

    for _ in range(iterations):
        new_rank = {}
        for pid in all_ids:
            incoming = forward.get(pid, [])
            contrib  = sum(
                rank[src] / max(out_degree.get(src, 1), 1)
                for src in incoming
                if src in rank
            )
            new_rank[pid] = (1 - damping) / n + damping * contrib
        rank = new_rank

    return rank


# ---------------------------------------------------------------------------
# BFS expansion
# ---------------------------------------------------------------------------

def bfs_expand(seed_ids, backward, forward, all_ids_set, max_depth=2):
    """
    Expand seed patents via BFS through both backward and forward citation edges.
    Returns dict: patent_id -> minimum hop distance from any seed patent.
    """
    visited = {}
    queue   = deque()

    for pid in seed_ids:
        if pid in all_ids_set:
            queue.append((pid, 0))
            visited[pid] = 0

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = (
            backward.get(current, []) +
            forward.get(current,  [])
        )
        for neighbor in neighbors:
            if neighbor in all_ids_set and neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return visited


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_citation_score(pid, hop_distance, pagerank, forward, backward,
                            seed_ids,
                            w_pagerank=0.5, w_forward=0.2,
                            w_backward=0.15, w_seed=0.1, w_hop=0.05):
    pr_score    = pagerank.get(pid, 0.0)
    fwd_count   = len(forward.get(pid,  []))
    bwd_count   = len(backward.get(pid, []))
    seed_bonus  = 1.0 if pid in seed_ids else 0.0
    hop_penalty = 1.0 / (1.0 + hop_distance)

    return (
        w_pagerank  * pr_score   +
        w_forward   * fwd_count  +
        w_backward  * bwd_count  +
        w_seed      * seed_bonus +
        w_hop       * hop_penalty
    )


# ---------------------------------------------------------------------------
# Null-safe corpus loader
# ---------------------------------------------------------------------------

def load_corpus(patents_csv: str) -> pd.DataFrame:
    """
    Load corpus CSV with null safety across all columns.

    - Rows with null patent_id are dropped (cannot be indexed)
    - title/abstract get placeholder text so embeddings never fail
    - ipc_codes/cpc_codes/cites/cited_by get empty string (safe for split)
    - publication_year gets 'Unknown' placeholder
    """
    df = pd.read_csv(patents_csv, dtype=str)

    # Drop rows with no patent_id
    before = len(df)
    df = df[df["patent_id"].notna() & (df["patent_id"].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        print(f"  [Null safety] Dropped {dropped} rows with missing patent_id")

    df["patent_id"] = df["patent_id"].str.strip()
    df["title"]     = df["title"].fillna("No title").str.strip()
    df["abstract"]  = df["abstract"].fillna("No abstract").str.strip()
    df["ipc_codes"] = df["ipc_codes"].fillna("")
    df["cpc_codes"] = df["cpc_codes"].fillna("")

    # Handle both old schema (citations) and new schema (cites + cited_by)
    if "cites" in df.columns:
        df["cites"]    = df["cites"].fillna("")
    elif "citations" in df.columns:
        df["cites"]    = df["citations"].fillna("")
    else:
        df["cites"]    = ""

    if "cited_by" in df.columns:
        df["cited_by"] = df["cited_by"].fillna("")
    else:
        df["cited_by"] = ""

    if "publication_year" in df.columns:
        df["publication_year"] = df["publication_year"].fillna("Unknown")

    print(f"  [Null safety] Corpus: {len(df)} valid patents "
          f"| {df['abstract'].eq('No abstract').sum()} missing abstracts "
          f"| {df['cites'].eq('').sum()} missing cites "
          f"| {df['cited_by'].eq('').sum()} missing cited_by")

    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def citation_search(
    keyword_csv,
    patents_csv,
    output_csv,
    max_depth      = 2,
    top_n          = 40,
    min_score      = 0.0,
    damping        = 0.85,
    pagerank_iters = 50,
):
    # --- Load data ---
    seed_df   = pd.read_csv(keyword_csv, dtype=str)
    corpus_df = load_corpus(patents_csv)

    seed_df["patent_id"] = seed_df["patent_id"].fillna("").str.strip()
    seed_ids    = set(seed_df["patent_id"].tolist()) - {""}
    all_ids     = corpus_df["patent_id"].tolist()
    all_ids_set = set(all_ids)

    # Guard: if keyword stage returned nothing, fall back to full corpus
    if not seed_ids:
        print("  [Warning] No seed patents from keyword stage — "
              "falling back to top-N by PageRank across full corpus.")

    # --- Build graph ---
    backward, forward = build_citation_graph(
        corpus_df,
        cites_column    = "cites",
        cited_by_column = "cited_by",
    )

    # --- PageRank over full corpus ---
    pagerank = compute_pagerank(all_ids, forward, damping, pagerank_iters)

    # --- BFS expansion from seed patents ---
    expanded = bfs_expand(seed_ids, backward, forward, all_ids_set, max_depth)

    # Seed patents not in corpus still get hop=0
    for pid in seed_ids:
        if pid not in expanded:
            expanded[pid] = 0

    # --- Score every expanded patent ---
    results = []
    for pid, hop in expanded.items():
        score = compute_citation_score(
            pid, hop, pagerank, forward, backward, seed_ids
        )
        results.append({
            "patent_id":          pid,
            "hop_distance":       hop,
            "pagerank_score":     round(pagerank.get(pid, 0.0), 6),
            "forward_citations":  len(forward.get(pid,  [])),
            "backward_citations": len(backward.get(pid, [])),
            "citation_score":     round(score, 6),
            "is_seed":            pid in seed_ids,
        })

    results_df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["patent_id","hop_distance","pagerank_score",
                 "forward_citations","backward_citations","citation_score","is_seed"]
    )

    if results_df.empty:
        print("  [Warning] No patents found after BFS expansion. "
              "Returning top-N from corpus by PageRank.")
        fallback = [{"patent_id": pid, "hop_distance": 0,
                     "pagerank_score": round(pagerank.get(pid,0.0),6),
                     "forward_citations": len(forward.get(pid,[])),
                     "backward_citations": len(backward.get(pid,[])),
                     "citation_score": round(pagerank.get(pid,0.0),6),
                     "is_seed": False}
                    for pid in all_ids]
        results_df = pd.DataFrame(fallback)

    # --- Filter and rank ---
    results_df = results_df[results_df["citation_score"] >= min_score]
    results_df = results_df.sort_values("citation_score", ascending=False)
    results_df = results_df.head(top_n)

    # --- Merge full patent metadata back in ---
    output_df = results_df.merge(corpus_df, on="patent_id", how="left")

    score_cols = ["patent_id", "citation_score", "pagerank_score",
                  "forward_citations", "backward_citations", "hop_distance", "is_seed"]
    meta_cols  = [c for c in output_df.columns if c not in score_cols]
    output_df  = output_df[score_cols + meta_cols]

    output_df.to_csv(output_csv, index=False)

    print("Citation Search Completed")
    print(f"  Seed patents (from keyword stage) : {len(seed_ids)}")
    print(f"  BFS expansion depth               : {max_depth}")
    print(f"  Candidates after expansion         : {len(expanded)}")
    print(f"  Patents returned (top-{top_n})       : {len(output_df)}")
    print(f"\n  Top 5 by citation score:")
    print(output_df[["patent_id", "citation_score", "pagerank_score",
                      "forward_citations", "hop_distance"]].head(5).to_string(index=False))

    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    citation_search(
        keyword_csv = "Output/keyword_filtered_patents.csv",
        patents_csv = "all_scrapped_patents.csv",
        output_csv  = "Output/citation_expanded_patents.csv",
        max_depth   = 2,
        top_n       = 40,
    )