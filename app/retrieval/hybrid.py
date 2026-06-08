"""
app/retrieval/hybrid.py — Hybrid retrieval + Gemini Flash re-ranking
=====================================================================
Combines BM25 (lexical) and dense (embedding) retrieval via Reciprocal Rank
Fusion (RRF), then optionally re-ranks with Gemini Flash.

Architecture:
    BM25 (top-20) + Dense (top-20) → RRF fusion → Gemini Flash re-rank → top-5

Interview talking points:
  - BM25 excels at exact keyword matches (acronyms, error codes).
    Dense embeddings handle paraphrase and semantic similarity.
    Hybrid = best of both → higher recall on diverse query types.
  - RRF works purely on rank positions — no score normalisation needed.
  - Re-ranking with Flash is a precision booster: cheap, fast, and catches
    nuances that bag-of-words and embeddings miss.
  - Two-stage retrieve-then-rerank is standard in production RAG.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import (
    PROJECT_ID, REGION, OFFLINE_LLM, ENABLE_RERANK,
    RERANK_MODEL, RRF_K, PARQUET_PATH, EMBEDDING_MODEL,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
_parquet_df = None


def _load_parquet() -> pd.DataFrame:
    global _parquet_df
    if _parquet_df is None:
        _parquet_df = pd.read_parquet(PARQUET_PATH)
    return _parquet_df


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------
_bm25_index = None
_bm25_df = None


def _get_bm25():
    """Lazily build BM25 index over corpus."""
    global _bm25_index, _bm25_df
    if _bm25_index is None:
        _bm25_df = _load_parquet()
        corpus_tokens = [str(t).lower().split() for t in _bm25_df["text"]]
        _bm25_index = BM25Okapi(corpus_tokens)
        log.info(f"BM25 index built: {len(corpus_tokens)} documents")
    return _bm25_index, _bm25_df


def _bm25_retrieve_parquet(query: str, top_k: int = 20) -> list[dict]:
    """Offline fallback: in-memory BM25 over parquet corpus."""
    bm25, df = _get_bm25()
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "source": df.iloc[i]["source"],
            "chunk_index": int(df.iloc[i]["chunk_index"]),
            "text": df.iloc[i]["text"],
            "bm25_score": float(scores[i]),
            "bm25_rank": rank,
        }
        for rank, i in enumerate(top_indices)
    ]


_BM25_SQL = """
SELECT
    id, source, chunk_index, text,
    ts_rank_cd(text_tsv, websearch_to_tsquery('english', %s)) AS rank_score
FROM document_chunks
WHERE text_tsv @@ websearch_to_tsquery('english', %s)
ORDER BY rank_score DESC
LIMIT %s;
"""


def _bm25_retrieve_pgvector(query: str, top_k: int = 20) -> list[dict]:
    """
    Production path: Postgres full-text search via tsvector + GIN index.

    Interview talking points:
      - ts_rank_cd uses cover density ranking — accounts for how close matching
        terms are to each other, similar to BM25's proximity signal.
      - websearch_to_tsquery handles natural language queries (AND/OR/NOT, phrases)
        more gracefully than plainto_tsquery.
      - The GIN index makes this sub-millisecond. No external search engine needed.
      - Trade-off vs Elasticsearch: Postgres FTS is simpler to operate (no extra
        cluster) but lacks features like custom analysers, fuzzy matching, and
        distributed sharding. At this corpus size, Postgres is the right call.
    """
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    )

    with conn.cursor() as cur:
        cur.execute(_BM25_SQL, (query, query, top_k))
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "source": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "bm25_score": round(float(row[4]), 4),
            "bm25_rank": rank,
        }
        for rank, row in enumerate(rows)
    ]


def bm25_retrieve(query: str, top_k: int = 20) -> list[dict]:
    """Dispatch: Postgres FTS in production, in-memory BM25 offline."""
    if OFFLINE_LLM:
        return _bm25_retrieve_parquet(query, top_k)
    return _bm25_retrieve_pgvector(query, top_k)


# ---------------------------------------------------------------------------
# Dense retrieval — pgvector (production) / parquet (offline)
# ---------------------------------------------------------------------------

_DENSE_SQL = """
SELECT
    id, source, chunk_index, text,
    1 - (embedding <=> %s::vector) AS cosine_similarity
FROM document_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def _embed_query(query: str) -> list[float]:
    """Embed a query via Vertex AI text-embedding-005."""
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    result = model.get_embeddings(
        [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    )
    return result[0].values


def _dense_retrieve_pgvector(query: str, top_k: int = 20) -> list[dict]:
    """
    Production path: embed query → cosine ANN search via pgvector HNSW index.

    Interview note: pgvector's <=> operator computes cosine distance. The HNSW
    index (m=16, ef_construction=64) makes this sub-millisecond even at millions
    of rows. We convert distance to similarity (1 - distance) for readability.
    """
    query_vec = _embed_query(query)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    )
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute(_DENSE_SQL, (query_vec, query_vec, top_k))
        rows = cur.fetchall()
    conn.close()

    return [
        {
            "source": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "dense_score": round(float(row[4]), 4),
            "dense_rank": rank,
        }
        for rank, row in enumerate(rows)
    ]


def _dense_retrieve_parquet(query: str, top_k: int = 20) -> list[dict]:
    """
    Offline fallback: token-overlap proxy against parquet.
    No embedding API or database needed — useful for local dev and eval.
    """
    df = _load_parquet()
    query_tokens = set(query.lower().split())
    scores = np.array([
        len(query_tokens & set(str(t).lower().split())) / max(len(query_tokens), 1)
        for t in df["text"]
    ])
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "source": df.iloc[i]["source"],
            "chunk_index": int(df.iloc[i]["chunk_index"]),
            "text": df.iloc[i]["text"],
            "dense_score": float(scores[i]),
            "dense_rank": rank,
        }
        for rank, i in enumerate(top_indices)
    ]


def dense_retrieve(query: str, top_k: int = 20) -> list[dict]:
    """Dispatch: pgvector in production, parquet fallback in offline mode."""
    if OFFLINE_LLM:
        return _dense_retrieve_parquet(query, top_k)
    return _dense_retrieve_pgvector(query, top_k)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(*result_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Merge ranked lists using RRF: score(d) = sum(1/(k + rank_i(d))).
    Retriever-agnostic — works on rank positions, not raw scores.
    """
    fused: dict[tuple, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = (doc["source"], doc["chunk_index"])
            if key not in fused:
                fused[key] = {
                    "source": doc["source"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"],
                    "rrf_score": 0.0,
                    "retrieval_sources": [],
                }
            fused[key]["rrf_score"] += 1.0 / (k + rank)
            if "bm25_rank" in doc:
                fused[key]["retrieval_sources"].append(f"bm25@{doc['bm25_rank']}")
            if "dense_rank" in doc:
                fused[key]["retrieval_sources"].append(f"dense@{doc['dense_rank']}")

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)


# ---------------------------------------------------------------------------
# Gemini Flash re-ranking
# ---------------------------------------------------------------------------

_RERANK_PROMPT = """\
You are a relevance judge. Given a user query and a list of text passages,
rate each passage's relevance to the query on a scale of 0-10.

Return ONLY a JSON array of objects with "index" and "score" fields, ordered
by score descending. Example: [{"index": 0, "score": 9}, {"index": 1, "score": 3}]

Query: {query}

Passages:
{passages}

JSON response (no explanation, just the array):"""


def rerank_with_flash(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank candidates using Gemini Flash as a relevance judge."""
    if OFFLINE_LLM:
        return candidates[:top_k]

    from google import genai

    passages_text = "\n\n".join(
        f"[{i}] {c['text'][:500]}" for i, c in enumerate(candidates)
    )
    prompt = _RERANK_PROMPT.format(query=query, passages=passages_text)

    try:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        response = client.models.generate_content(model=RERANK_MODEL, contents=prompt)

        response_text = response.text.strip()
        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        rankings = json.loads(response_text)

        # Defensive: handle whatever key names the model decides to use
        reranked = []
        for item in rankings[:top_k]:
            if not isinstance(item, dict):
                continue
            # Try common key names for the index field
            idx = item.get("index") or item.get("idx") or item.get("passage_index")
            score = item.get("score") or item.get("relevance") or item.get("relevance_score")
            if idx is None or score is None:
                # Last resort: grab first int-like and first number-like values
                vals = list(item.values())
                idx = next((v for v in vals if isinstance(v, int)), None)
                score = next((v for v in vals if isinstance(v, (int, float))), None)
            if idx is not None and 0 <= int(idx) < len(candidates):
                candidate = candidates[int(idx)].copy()
                candidate["rerank_score"] = float(score or 0)
                reranked.append(candidate)
        if reranked:
            log.debug(f"Re-ranked {len(reranked)} candidates")
            return reranked
    except Exception as e:
        log.warning(f"Re-ranking failed ({e}); falling back to RRF order")

    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 5,
    retrieval_k: int = 20,
    enable_rerank: bool | None = None,
) -> tuple[list[dict], dict]:
    """
    Full hybrid pipeline: BM25 + Dense → RRF → optional Gemini Flash re-rank.

    Returns (results, timing_info). Results include a 'cosine_similarity' key
    for backward compatibility with the QueryResponse model.
    """
    if enable_rerank is None:
        enable_rerank = ENABLE_RERANK

    timing = {}

    t0 = time.monotonic()
    bm25_results = bm25_retrieve(query, top_k=retrieval_k)
    timing["bm25_ms"] = round((time.monotonic() - t0) * 1000, 1)

    t0 = time.monotonic()
    dense_results = dense_retrieve(query, top_k=retrieval_k)
    timing["dense_ms"] = round((time.monotonic() - t0) * 1000, 1)

    t0 = time.monotonic()
    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    timing["rrf_ms"] = round((time.monotonic() - t0) * 1000, 1)

    if enable_rerank:
        t0 = time.monotonic()
        results = rerank_with_flash(query, fused[:retrieval_k], top_k=top_k)
        timing["rerank_ms"] = round((time.monotonic() - t0) * 1000, 1)
    else:
        results = fused[:top_k]
        timing["rerank_ms"] = 0.0

    timing["total_ms"] = round(sum(timing.values()), 1)

    # Backward-compatible score key
    for r in results:
        if "rerank_score" in r:
            r["cosine_similarity"] = r["rerank_score"] / 10.0
        else:
            r["cosine_similarity"] = r.get("rrf_score", 0.0)

    return results, timing
