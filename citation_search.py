import pandas as pd
import numpy as np
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_citation_graph(patents_df, citation_column="citations"):
    """
    Build a bidirectional citation graph from the full patent corpus.

    Returns:
        backward: dict[patent_id -> list of patents it cites]
        forward:  dict[patent_id -> list of patents that cite it]
    """
    backward = defaultdict(list)   # patent -> patents it cites (backward citations)
    forward  = defaultdict(list)   # patent -> patents that cite it (forward citations)

    for _, row in patents_df.iterrows():
        src = row["patent_id"]
        raw = str(row[citation_column]) if pd.notna(row[citation_column]) else ""
        cited = [c.strip() for c in raw.split(",") if c.strip() and c.strip() != "nan"]
        for tgt in cited:
            backward[src].append(tgt)
            forward[tgt].append(src)

    return dict(backward), dict(forward)


# ---------------------------------------------------------------------------
# PageRank on the citation graph
# ---------------------------------------------------------------------------

def compute_pagerank(all_ids, forward, damping=0.85, iterations=50):
    """
    Standard PageRank over the citation graph.
    A patent cited by high-PageRank patents scores higher than one cited by
    obscure patents — this is the key advantage over raw citation counts.

    In citation graphs, 'importance flows from citers to cited':
      - a patent's rank is boosted by the rank of every patent that cites it.
    """
    n = len(all_ids)
    rank = {pid: 1.0 / n for pid in all_ids}

    # Build out-degree map (number of patents each node cites)
    # We need backward graph for outgoing edges in the citation direction
    out_degree = defaultdict(int)
    for pid in all_ids:
        # outgoing edges = patents that cite pid (forward edges)
        out_degree[pid] = len(forward.get(pid, []))

    for _ in range(iterations):
        new_rank = {}
        for pid in all_ids:
            # Sum contributions from all patents that cite pid
            incoming = forward.get(pid, [])
            contrib = sum(
                rank[src] / max(out_degree[src], 1)
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
    Returns a dict: patent_id -> minimum hop distance from any seed patent.
    """
    visited = {}  # patent_id -> hop distance
    queue = deque()

    for pid in seed_ids:
        if pid in all_ids_set:
            queue.append((pid, 0))
            visited[pid] = 0

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = (
            backward.get(current, []) +  # patents this one cites
            forward.get(current, [])      # patents that cite this one
        )
        for neighbor in neighbors:
            if neighbor in all_ids_set and neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))

    return visited


# ---------------------------------------------------------------------------
# Final scoring
# ---------------------------------------------------------------------------

def compute_citation_score(pid, hop_distance, pagerank, forward, backward,
                            seed_ids, w_pagerank=0.5, w_forward=0.2,
                            w_backward=0.15, w_seed=0.1, w_hop=0.05):
    """
    Composite score combining:
      - PageRank (influence in the full citation network)
      - Forward citation count (how often this patent is cited)
      - Backward citation count (how many prior patents it builds on)
      - Seed bonus (was this patent already in the keyword-filtered set)
      - Hop penalty (closer to seed = more relevant)
    """
    pr_score       = pagerank.get(pid, 0.0)
    fwd_count      = len(forward.get(pid, []))
    bwd_count      = len(backward.get(pid, []))
    seed_bonus     = 1.0 if pid in seed_ids else 0.0
    hop_penalty    = 1.0 / (1.0 + hop_distance)

    # Normalisation happens downstream; raw composite here
    score = (
        w_pagerank  * pr_score   +
        w_forward   * fwd_count  +
        w_backward  * bwd_count  +
        w_seed      * seed_bonus +
        w_hop       * hop_penalty
    )
    return score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def citation_search(
    keyword_csv,          # output of keyword_search.py  (seed patents)
    patents_csv,          # full patent corpus
    output_csv,           # where to write results
    max_depth=2,          # BFS expansion depth (1 = direct neighbors only)
    top_n=40,             # maximum patents to return
    min_score=0.0,        # optional minimum score threshold
    damping=0.85,         # PageRank damping factor
    pagerank_iters=50,    # PageRank convergence iterations
):
    # --- Load data ---
    seed_df   = pd.read_csv(keyword_csv)
    corpus_df = pd.read_csv(patents_csv)

    seed_ids     = set(seed_df["patent_id"].astype(str).tolist())
    all_ids      = corpus_df["patent_id"].astype(str).tolist()
    all_ids_set  = set(all_ids)

    # --- Build graph ---
    backward, forward = build_citation_graph(corpus_df)

    # --- PageRank over full corpus ---
    pagerank = compute_pagerank(all_ids, forward, damping, pagerank_iters)

    # --- BFS expansion from seed patents ---
    expanded = bfs_expand(seed_ids, backward, forward, all_ids_set, max_depth)

    # Seed patents not found in corpus still get hop=0
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
            "patent_id":        pid,
            "hop_distance":     hop,
            "pagerank_score":   round(pagerank.get(pid, 0.0), 6),
            "forward_citations": len(forward.get(pid, [])),
            "backward_citations": len(backward.get(pid, [])),
            "citation_score":   round(score, 6),
            "is_seed":          pid in seed_ids,
        })

    results_df = pd.DataFrame(results)

    # --- Filter and rank ---
    results_df = results_df[results_df["citation_score"] >= min_score]
    results_df = results_df.sort_values("citation_score", ascending=False)
    results_df = results_df.head(top_n)

    # --- Merge full patent metadata back in ---
    output_df = results_df.merge(corpus_df, on="patent_id", how="left")

    # Reorder: score columns first, then metadata
    score_cols = ["patent_id", "citation_score", "pagerank_score",
                  "forward_citations", "backward_citations", "hop_distance", "is_seed"]
    meta_cols  = [c for c in output_df.columns if c not in score_cols]
    output_df  = output_df[score_cols + meta_cols]

    output_df.to_csv(output_csv, index=False)

    # --- Summary ---
    print("Citation Search Completed")
    print(f"Seed patents (from keyword stage): {len(seed_ids)}")
    print(f"BFS expansion depth:               {max_depth}")
    print(f"Candidates after expansion:        {len(expanded)}")
    print(f"Patents returned (top-{top_n}):        {len(output_df)}")
    print(f"\nTop 5 by citation score:")
    print(output_df[["patent_id", "citation_score", "pagerank_score",
                      "forward_citations", "hop_distance"]].head(5).to_string(index=False))

    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    citation_search(
        keyword_csv="keyword_filtered_patents.csv",
        patents_csv="all_scraped_patents.csv",
        output_csv="citation_expanded_patents.csv",
        max_depth=2,
        top_n=40,
    )