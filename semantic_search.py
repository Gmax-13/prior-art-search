import os
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_HOME",                         os.getenv("HF_HOME", "D:/huggingface_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1"))
os.environ.setdefault("HF_HUB_OFFLINE",                  os.getenv("HF_HUB_OFFLINE", "1"))

"""
Stage 4 — Semantic Search
=========================
Two-stage pipeline:
  Stage 4a  Bi-encoder embedding + Mahalanobis retrieval (fallback: cosine if corpus < 500)
  Stage 4b  Cross-encoder reranking on the top-K shortlist

Embedding backend (auto-selected):
  - sentence-transformers available  →  SPECTER2  (allenai/specter2_base)
  - fallback                         →  TF-IDF SVD (no external deps, CPU-only)

Cross-encoder backend (auto-selected):
  - sentence-transformers available  →  cross-encoder/ms-marco-MiniLM-L6-v2
  - fallback                         →  BM25-style token overlap scorer

Design decisions:
  - Mahalanobis requires estimating a 768×768 covariance matrix.
    Stable estimation needs n >> d; threshold set at 500 corpus patents.
    Below that, cosine is used for retrieval (same pipeline, different metric).
  - Citation score from Stage 3 is fused into the final score (alpha=0.7 semantic).
  - All score components are written to output for Stage 5 / analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import warnings
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAHALANOBIS_MIN_CORPUS   = 500    # minimum patents for stable covariance estimation
MAHALANOBIS_REGULARIZE   = 1e-5   # lambda * I added to covariance before inversion
TFIDF_SVD_COMPONENTS     = 128    # LSA dimensions for TF-IDF fallback
CITATION_FUSION_ALPHA    = 0.7    # weight on semantic score; (1-alpha) on citation score
SPECTER2_MODEL           = "sentence-transformers/allenai-specter"
CROSS_ENCODER_MODEL      = "cross-encoder/ms-marco-MiniLM-L6-v2"
RETRIEVAL_SHORTLIST_K    = 20     # candidates passed from Stage 4a to 4b


# ---------------------------------------------------------------------------
# Text representation
# ---------------------------------------------------------------------------

def build_text_repr(row: pd.Series) -> str:
    """
    Concatenate fields into a single string for embedding.
    Format: [TITLE] ... [ABSTRACT] ... [IPC] ...

    Field weighting is handled implicitly by token repetition in the string.
    Title is repeated twice — cheap way to upweight it without late fusion.
    """
    title    = str(row.get("title",    "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    ipc      = str(row.get("ipc_codes","")).strip()

    parts = []
    if title:
        parts.append(f"[TITLE] {title} [TITLE] {title}")   # repeat = 2× weight
    if abstract:
        parts.append(f"[ABSTRACT] {abstract}")
    if ipc and ipc.lower() not in ("nan", ""):
        parts.append(f"[IPC] {ipc}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class SPECTER2Embedder:
    """
    Dense bi-encoder using allenai/specter2_base.
    Trained on citation graphs — semantically aligned with our pipeline.
    Output vectors are L2-normalised (unit norm).
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {SPECTER2_MODEL} …")
        self.model = SentenceTransformer(SPECTER2_MODEL)
        logger.info("Model loaded.")

    def encode(self, texts: list, batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # unit norm for cosine/Mahalanobis
            convert_to_numpy=True,
        )
        return embeddings                # shape: (n, 768)


class TFIDFEmbedder:
    """
    Fallback bi-encoder: TF-IDF + truncated SVD (Latent Semantic Analysis).
    No external model download required.
    Lower quality than SPECTER2 but same interface.
    """

    def __init__(self, n_components: int = TFIDF_SVD_COMPONENTS):
        self.vectorizer  = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.svd         = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted     = False
        self.n_components = n_components

    def fit(self, texts: list):
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True

    def encode(self, texts: list, batch_size: int = 64) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() with the full corpus before encode().")
        tfidf      = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf)
        return normalize(embeddings, norm="l2")   # unit norm, shape: (n, n_components)


def get_embedder():
    """Auto-select embedding backend."""
    try:
        embedder = SPECTER2Embedder()
        logger.info("Bi-encoder backend: SPECTER2")
        return embedder, "specter2"
    except ImportError:
        logger.warning(
            "sentence-transformers not found. "
            "Install with: pip install sentence-transformers\n"
            "Falling back to TF-IDF + SVD embedder (lower quality)."
        )
        return TFIDFEmbedder(), "tfidf"


# ---------------------------------------------------------------------------
# Retrieval metric: Mahalanobis vs cosine
# ---------------------------------------------------------------------------

def compute_covariance_inverse(embeddings: np.ndarray, reg: float = MAHALANOBIS_REGULARIZE):
    """
    Estimate the inverse covariance matrix (precision matrix) of the embedding space.

    Mahalanobis distance: d(a,b) = sqrt((a-b)ᵀ Σ⁻¹ (a-b))

    The precision matrix Σ⁻¹ re-weights dimensions by their discriminativeness:
      - High-variance dimensions (noisy) get downweighted
      - Low-variance, informative dimensions get upweighted
    This corrects for the known anisotropy of transformer embedding spaces.

    Regularisation (λI) is added before inversion to ensure positive definiteness
    and numerical stability. λ = 1e-5 is conservative — increase if singular.

    Requires n >> d for stable estimation. Called only when corpus >= 500.
    """
    cov = np.cov(embeddings.T)                          # (d, d)
    cov += reg * np.eye(cov.shape[0])                   # regularise
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix inversion failed — using pseudo-inverse.")
        cov_inv = np.linalg.pinv(cov)
    return cov_inv                                       # (d, d)


def mahalanobis_similarities(query_vec: np.ndarray,
                              candidate_vecs: np.ndarray,
                              cov_inv: np.ndarray) -> np.ndarray:
    """
    Compute Mahalanobis-based similarity scores between the query and each candidate.

    We convert distance to similarity via: sim = 1 / (1 + distance)
    so higher = more similar, range (0, 1].
    """
    diff        = candidate_vecs - query_vec             # (n, d)
    # (n,d) @ (d,d) → (n,d), then element-wise multiply with diff, sum over d
    left        = diff @ cov_inv                         # (n, d)
    distances   = np.sqrt(np.maximum(
        np.einsum("ij,ij->i", left, diff), 0.0          # (n,) — clamp to ≥ 0
    ))
    return 1.0 / (1.0 + distances)                       # (n,) similarity scores


def retrieval_stage(query_vec: np.ndarray,
                    candidate_vecs: np.ndarray,
                    corpus_size: int,
                    cov_inv=None) -> np.ndarray:
    """
    Returns similarity scores for each candidate.
    Uses Mahalanobis if corpus >= MAHALANOBIS_MIN_CORPUS, else cosine.
    """
    if corpus_size >= MAHALANOBIS_MIN_CORPUS and cov_inv is not None:
        logger.info(
            f"Corpus size {corpus_size} ≥ {MAHALANOBIS_MIN_CORPUS}: "
            "using Mahalanobis distance."
        )
        return mahalanobis_similarities(query_vec, candidate_vecs, cov_inv)
    else:
        if corpus_size < MAHALANOBIS_MIN_CORPUS:
            logger.info(
                f"Corpus size {corpus_size} < {MAHALANOBIS_MIN_CORPUS}: "
                "insufficient data for stable covariance estimation. "
                "Falling back to cosine similarity."
            )
        scores = sk_cosine(query_vec.reshape(1, -1), candidate_vecs)[0]  # (n,)
        return scores


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross-encoder: takes (query_text, candidate_text) jointly as input.
    Full attention between both texts — captures token-level interactions
    that bi-encoders lose by compressing each text independently.

    model: cross-encoder/ms-marco-MiniLM-L6-v2
      - Trained on MS MARCO passage relevance
      - 6-layer MiniLM — fast on CPU (~0.1s per pair)
      - Outputs raw logit (higher = more relevant)
    """

    def __init__(self):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL} …")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("Cross-encoder loaded.")

    def rerank(self, query_text: str, candidate_texts: list) -> np.ndarray:
        """
        Returns relevance scores (raw logits) for each candidate.
        Shape: (n,)
        """
        pairs  = [(query_text, ct) for ct in candidate_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return np.array(scores)


class BM25Reranker:
    """
    Fallback reranker: BM25-style token overlap with IDF weighting.
    No external model — same interface as CrossEncoderReranker.

    BM25 formula:
      score(q,d) = Σ IDF(t) * (f(t,d) * (k1+1)) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
      k1=1.5, b=0.75 (standard parameters)

    Significantly weaker than the neural cross-encoder but captures
    exact term matches the bi-encoder may miss.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    def rerank(self, query_text: str, candidate_texts: list) -> np.ndarray:
        import math
        from collections import Counter

        def tokenize(text):
            return text.lower().split()

        query_tokens = set(tokenize(query_text))
        N            = len(candidate_texts)
        tokenized    = [tokenize(ct) for ct in candidate_texts]
        avgdl        = np.mean([len(d) for d in tokenized]) if tokenized else 1.0

        # IDF per query term
        def idf(term):
            df = sum(1 for d in tokenized if term in d)
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

        scores = []
        for doc_tokens in tokenized:
            freq   = Counter(doc_tokens)
            dl     = len(doc_tokens)
            score  = 0.0
            for term in query_tokens:
                f = freq.get(term, 0)
                score += idf(term) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1))
                )
            scores.append(score)

        return np.array(scores)


def get_reranker():
    """Auto-select reranker backend."""
    try:
        reranker = CrossEncoderReranker()
        logger.info("Reranker backend: CrossEncoder (ms-marco-MiniLM-L6-v2)")
        return reranker, "cross_encoder"
    except ImportError:
        logger.warning(
            "sentence-transformers not found. "
            "Falling back to BM25 reranker (lower quality)."
        )
        return BM25Reranker(), "bm25"


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def fuse_scores(semantic_scores: np.ndarray,
                citation_scores: np.ndarray,
                alpha: float = CITATION_FUSION_ALPHA) -> np.ndarray:
    """
    Fuse reranker semantic score with citation score from Stage 3.

    final = alpha * norm(semantic) + (1 - alpha) * norm(citation)

    Both are min-max normalised to [0,1] before fusion so they are
    on the same scale. Citation score encodes network influence (PageRank +
    forward/backward counts + hop distance) — retaining it avoids discarding
    Stage 3 signal entirely.
    """
    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)

    sem_norm  = minmax(semantic_scores)
    cite_norm = minmax(citation_scores)
    return alpha * sem_norm + (1.0 - alpha) * cite_norm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def semantic_search(
    citation_csv:   str,           # output of citation_search.py
    input_csv:      str,           # query patent
    all_patents_csv: str,          # full corpus (for covariance estimation)
    output_csv:     str,           # where to write final results
    top_k_retrieval: int = RETRIEVAL_SHORTLIST_K,
    top_n_final:    int  = 5,      # final results returned
    alpha:          float = CITATION_FUSION_ALPHA,
    batch_size:     int   = 64,
):
    # --- Load data ---
    candidates_df = pd.read_csv(citation_csv)
    query_df      = pd.read_csv(input_csv)
    corpus_df     = pd.read_csv(all_patents_csv)

    query_row     = query_df.iloc[0]
    query_text    = build_text_repr(query_row)
    cand_texts    = candidates_df.apply(build_text_repr, axis=1).tolist()
    corpus_texts  = corpus_df.apply(build_text_repr, axis=1).tolist()
    corpus_size   = len(corpus_df)

    logger.info(f"Query patent:     '{query_row.get('title', 'N/A')}'")
    logger.info(f"Candidates:        {len(candidates_df)}")
    logger.info(f"Full corpus size:  {corpus_size}")

    # -----------------------------------------------------------------------
    # STAGE 4a — Bi-encoder embedding + retrieval
    # -----------------------------------------------------------------------

    embedder, embed_backend = get_embedder()

    # TF-IDF fallback needs fit on full corpus for IDF weights
    if embed_backend == "tfidf":
        embedder.fit(corpus_texts + [query_text])

    # Embed query and candidates
    all_texts       = [query_text] + cand_texts
    all_embeddings  = embedder.encode(all_texts, batch_size=batch_size)
    query_vec       = all_embeddings[0]             # (d,)
    cand_vecs       = all_embeddings[1:]            # (n_candidates, d)

    # Covariance inverse from full corpus (only if corpus large enough)
    cov_inv = None
    if corpus_size >= MAHALANOBIS_MIN_CORPUS:
        corpus_embeddings = embedder.encode(corpus_texts, batch_size=batch_size)
        cov_inv = compute_covariance_inverse(corpus_embeddings)

    # Retrieval scores
    retrieval_scores = retrieval_stage(query_vec, cand_vecs, corpus_size, cov_inv)

    # Shortlist top-K for cross-encoder
    k               = min(top_k_retrieval, len(candidates_df))
    shortlist_idx   = np.argsort(retrieval_scores)[::-1][:k]
    shortlist_df    = candidates_df.iloc[shortlist_idx].copy().reset_index(drop=True)
    shortlist_texts = [cand_texts[i] for i in shortlist_idx]
    shortlist_ret   = retrieval_scores[shortlist_idx]

    logger.info(
        f"Stage 4a ({embed_backend} + "
        f"{'mahalanobis' if cov_inv is not None else 'cosine'}): "
        f"shortlisted {len(shortlist_df)} candidates."
    )

    # -----------------------------------------------------------------------
    # STAGE 4b — Cross-encoder reranking
    # -----------------------------------------------------------------------

    reranker, rerank_backend = get_reranker()
    rerank_scores = reranker.rerank(query_text, shortlist_texts)

    logger.info(f"Stage 4b ({rerank_backend}): reranking complete.")

    # -----------------------------------------------------------------------
    # Score fusion + final ranking
    # -----------------------------------------------------------------------

    citation_scores = shortlist_df["citation_score"].fillna(0.0).values
    final_scores    = fuse_scores(rerank_scores, citation_scores, alpha)

    shortlist_df["retrieval_score"]  = np.round(shortlist_ret,   6)
    shortlist_df["rerank_score"]     = np.round(rerank_scores,   6)
    shortlist_df["final_score"]      = np.round(final_scores,    6)
    shortlist_df["embed_backend"]    = embed_backend
    shortlist_df["rerank_backend"]   = rerank_backend
    shortlist_df["retrieval_metric"] = "mahalanobis" if cov_inv is not None else "cosine"

    output_df = (
        shortlist_df
        .sort_values("final_score", ascending=False)
        .head(top_n_final)
        .reset_index(drop=True)
    )

    # Reorder columns: score columns first
    score_cols = [
        "patent_id", "final_score", "rerank_score", "retrieval_score",
        "citation_score", "pagerank_score", "retrieval_metric",
        "embed_backend", "rerank_backend",
    ]
    meta_cols  = [c for c in output_df.columns if c not in score_cols]
    output_df  = output_df[score_cols + meta_cols]

    output_df.to_csv(output_csv, index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Semantic Search Completed")
    print("=" * 60)
    print(f"  Embedding backend  : {embed_backend}")
    print(f"  Retrieval metric   : {'mahalanobis' if cov_inv is not None else 'cosine (corpus < 500)'}")
    print(f"  Reranker backend   : {rerank_backend}")
    print(f"  Candidates in      : {len(candidates_df)}")
    print(f"  Shortlisted (4a)   : {len(shortlist_df)}")
    print(f"  Final results (4b) : {len(output_df)}")
    print(f"  Citation fusion α  : {alpha}")
    print()
    print(f"  Top {top_n_final} results:")
    print(
        output_df[[
            "patent_id", "final_score", "rerank_score",
            "retrieval_score", "title"
        ]].to_string(index=False)
    )
    print("=" * 60)

    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    semantic_search(
        citation_csv    = "citation_expanded_patents.csv",
        input_csv       = "input.csv",
        all_patents_csv = "all_scraped_patents.csv",
        output_csv      = "final_results.csv",
        top_k_retrieval = 20,
        top_n_final     = 5,
        alpha           = 0.7,
    )