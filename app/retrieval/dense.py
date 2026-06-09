"""
app/retrieval/dense.py — Dense vector search via pgvector
=========================================================
Embeds the query with text-embedding-005, runs cosine-similarity ANN
search against pgvector's HNSW index.

Interview talking points:
  - Same model for query + document embeddings is essential.
  - task_type=RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT — Vertex AI applies
    different prompt prefixes to tune the embedding space.
  - HNSW index makes this sub-millisecond even at millions of chunks.
  - Cosine similarity = 1 − cosine_distance. pgvector uses distance
    (lower is more similar); we convert to similarity for readability.
"""

from __future__ import annotations

import logging
import time

import psycopg2
import vertexai
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from app.config import (
    PROJECT_ID, REGION, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# Lazy init: defer vertexai.init + from_pretrained until the first dense query.
# Initializing at module scope would resolve ADC at import time, breaking
# startup in offline/parquet mode (and any environment without credentials).
_embed_model = None


def _get_embed_model() -> TextEmbeddingModel:
    """Lazily construct the embedding model on first use."""
    global _embed_model
    if _embed_model is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        _embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    return _embed_model


def embed_query(query: str) -> list[float]:
    """Embed a query string with RETRIEVAL_QUERY task type."""
    result = _get_embed_model().get_embeddings(
        [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    )
    return result[0].values


# ---------------------------------------------------------------------------
# Nearest-neighbour search
# ---------------------------------------------------------------------------

_SEARCH_SQL = """
SELECT
    id, source, chunk_index, text,
    1 - (embedding <=> %s::vector) AS cosine_similarity
FROM document_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


def search(query: str, top_k: int = 3) -> tuple[list[dict], float]:
    """
    Embed query, run ANN search, return (results, retrieval_ms).

    Each result dict has keys: id, source, chunk_index, text, cosine_similarity.
    """
    t0 = time.monotonic()
    query_vec = embed_query(query)
    embed_ms = round((time.monotonic() - t0) * 1000, 1)

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    )
    register_vector(conn)

    t1 = time.monotonic()
    with conn.cursor() as cur:
        cur.execute(_SEARCH_SQL, (query_vec, query_vec, top_k))
        rows = cur.fetchall()
    search_ms = round((time.monotonic() - t1) * 1000, 1)
    conn.close()

    results = [
        {
            "id": row[0],
            "source": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "cosine_similarity": round(row[4], 4),
        }
        for row in rows
    ]

    retrieval_ms = round(embed_ms + search_ms, 1)
    log.debug(f"search: embed={embed_ms}ms pgvector={search_ms}ms total={retrieval_ms}ms")
    return results, retrieval_ms
