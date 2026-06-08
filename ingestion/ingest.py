"""
ingest.py — Day 2: RAG ingestion pipeline
==========================================
Reads .txt or .pdf files from a corpus directory, chunks them with
LangChain's RecursiveCharacterTextSplitter, embeds with Vertex AI
text-embedding-005, and upserts into pgvector on Cloud SQL.

Usage:
    # 1. Download corpus first (if you haven't already):
    python download_corpus.py --out corpus

    # 2. Set env vars in .env (see .env.example), then:
    python ingest.py --corpus corpus --batch-size 5

    # Dry-run (no DB writes, saves embeddings to parquet instead):
    python ingest.py --corpus corpus --dry-run

Architecture note (for interview):
    - Chunk size 800 tokens / 100-token overlap: balances context richness
      vs. retrieval precision. Larger chunks → more context per hit but
      noisier similarity scores. 800/100 is a common starting point.
    - text-embedding-005: Google's latest text embedding model.
      768 dimensions (default), L2-normalised → cosine similarity = dot product.
    - Batch size 5: Vertex AI Embeddings API quota is 5 texts/request
      on the free tier; increase to 250 on paid.
    - Upsert strategy: ON CONFLICT DO NOTHING on (source, chunk_index).
      Re-running ingest is idempotent.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GCP
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# PostgreSQL + pgvector
import psycopg2
from pgvector.psycopg2 import register_vector

# Centralised config
from app.config import (
    PROJECT_ID, REGION, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    EMBEDDING_MODEL, EMBEDDING_DIM, CHUNK_SIZE, CHUNK_OVERLAP, CORPUS_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMBEDDING_TASK = "RETRIEVAL_DOCUMENT"

# ---------------------------------------------------------------------------
# Vertex AI init
# ---------------------------------------------------------------------------
vertexai.init(project=PROJECT_ID, location=REGION)
embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(corpus_dir: Path) -> list[dict]:
    """
    Load all .txt and .pdf files from corpus_dir.
    Returns a list of {source, text} dicts.
    """
    docs = []

    # .txt files (from download_corpus.py)
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 100:
            docs.append({"source": path.name, "text": text})

    # .pdf files
    try:
        from pypdf import PdfReader
        for path in sorted(corpus_dir.glob("*.pdf")):
            try:
                reader = PdfReader(str(path))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n\n".join(pages)
                if len(text) > 100:
                    docs.append({"source": path.name, "text": text})
            except Exception as exc:
                log.warning(f"Could not read PDF {path.name}: {exc}")
    except ImportError:
        log.info("pypdf not installed — skipping PDF files")

    log.info(f"Loaded {len(docs)} documents from {corpus_dir}")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split each document into overlapping text chunks.
    Returns list of {source, chunk_index, text}.

    Interview talking point: RecursiveCharacterTextSplitter tries to split
    on paragraphs → sentences → words → chars in order, so chunk boundaries
    fall at natural language boundaries rather than mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "source": doc["source"],
                "chunk_index": i,
                "text": text.strip(),
            })

    log.info(
        f"Chunked into {len(chunks)} chunks "
        f"(avg {sum(len(c['text']) for c in chunks) // max(len(chunks), 1)} chars)"
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts via Vertex AI text-embedding-005.
    Returns list of 768-dim float vectors.
    """
    inputs = [TextEmbeddingInput(text=t, task_type=EMBEDDING_TASK) for t in texts]
    result = embed_model.get_embeddings(inputs)
    return [r.values for r in result]


def embed_chunks(chunks: list[dict], batch_size: int = 5) -> list[dict]:
    """
    Add 'embedding' field to each chunk dict.
    Processes in batches; retries on transient errors with exponential back-off.
    """
    total = len(chunks)
    log.info(f"Embedding {total} chunks in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]

        for attempt in range(1, 4):
            try:
                embeddings = embed_batch(texts)
                for chunk, emb in zip(batch, embeddings):
                    chunk["embedding"] = emb
                break
            except Exception as exc:
                wait = 2 ** attempt
                log.warning(f"Embed attempt {attempt} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)
        else:
            log.error(f"Embedding failed for batch starting at {start}; skipping")

        done = min(start + batch_size, total)
        if done % 50 == 0 or done == total:
            log.info(f"  Embedded {done}/{total} chunks")

    return chunks


# ---------------------------------------------------------------------------
# pgvector upsert
# ---------------------------------------------------------------------------

def get_db_conn():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


UPSERT_SQL = """
INSERT INTO document_chunks (source, chunk_index, text, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""


def upsert_chunks(chunks: list[dict]) -> int:
    """
    Upsert all chunks into pgvector. Returns count of rows inserted.
    Skips chunks that are missing embeddings.
    """
    ready = [c for c in chunks if "embedding" in c]
    if not ready:
        log.warning("No embedded chunks to upsert")
        return 0

    conn = get_db_conn()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for chunk in ready:
                cur.execute(
                    UPSERT_SQL,
                    (
                        chunk["source"],
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk["embedding"],  # pgvector accepts list[float]
                    ),
                )
                inserted += cur.rowcount
        conn.commit()
        log.info(f"Upserted {inserted} new chunks into pgvector")
    finally:
        conn.close()

    return inserted


# ---------------------------------------------------------------------------
# Dry-run: save to parquet for inspection
# ---------------------------------------------------------------------------

def save_parquet(chunks: list[dict], out_path: str = "embeddings.parquet") -> None:
    """Save chunks + embeddings to parquet for offline inspection."""
    df = pd.DataFrame([
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "embedding": c.get("embedding"),
        }
        for c in chunks
    ])
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} rows → {out_path}")
    log.info(df[["source", "chunk_index", "text"]].head(5).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline — Day 2")
    parser.add_argument("--corpus", default="corpus", help="Corpus directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Embedding batch size")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed but don't write to DB; save to embeddings.parquet instead",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        log.error(f"Corpus directory not found: {corpus_dir}")
        log.error("Run: python download_corpus.py --out corpus")
        sys.exit(1)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    t0 = time.monotonic()

    docs = load_documents(corpus_dir)
    if not docs:
        log.error("No documents found. Run download_corpus.py first.")
        sys.exit(1)

    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks, batch_size=args.batch_size)

    if args.dry_run:
        log.info("DRY RUN — saving to parquet (no DB write)")
        save_parquet(chunks)
    else:
        inserted = upsert_chunks(chunks)
        log.info(f"Ingestion complete. {inserted} new chunks stored in pgvector.")

    elapsed = time.monotonic() - t0
    log.info(
        json.dumps({
            "event": "ingest_complete",
            "documents": len(docs),
            "chunks": len(chunks),
            "elapsed_s": round(elapsed, 2),
            "dry_run": args.dry_run,
        })
    )


if __name__ == "__main__":
    main()
