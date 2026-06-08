-- schema.sql — pgvector schema for the Vertex Knowledge Assistant
-- Run this once against your Cloud SQL (or local) PostgreSQL database.
--
-- Usage (local):
--   psql -U postgres -d ragdb -f schema.sql
--
-- Usage (Cloud SQL Auth Proxy):
--   psql "host=127.0.0.1 port=5432 dbname=ragdb user=rag_user" -f schema.sql

-- Enable the pgvector extension (requires PostgreSQL >= 14)
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- document_chunks: one row per chunk, embedding stored as vector(768)
-- text-embedding-005 produces 768-dimensional vectors by default.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL,          -- source URL or filename
    chunk_index   INT         NOT NULL,          -- position within document
    text          TEXT        NOT NULL,          -- raw chunk text
    embedding     vector(768) NOT NULL,          -- Vertex AI text-embedding-005
    char_count    INT         GENERATED ALWAYS AS (char_length(text)) STORED,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Index: HNSW for approximate nearest-neighbour search (cosine distance)
-- HNSW is generally faster at query time than IVFFlat; use IVFFlat if you
-- need a smaller index size or have < 1M rows.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Composite index for filtering by source
CREATE INDEX IF NOT EXISTS document_chunks_source_idx
    ON document_chunks (source);

-- ---------------------------------------------------------------------------
-- Full-text search: tsvector column + GIN index (Day 6 — hybrid BM25)
-- ---------------------------------------------------------------------------
-- ts_rank_cd uses cover density ranking, which approximates BM25's
-- term-frequency saturation + document-length normalisation.
-- The 'english' config applies stemming + stop-word removal.
--
-- Interview note: Postgres FTS is "good enough BM25" without standing up
-- Elasticsearch. GIN index keeps queries sub-millisecond. For true BM25
-- at scale, you'd add Elasticsearch or use Cloud SQL's built-in FTS.
-- ---------------------------------------------------------------------------
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS document_chunks_fts_idx
    ON document_chunks USING gin (text_tsv);

-- ---------------------------------------------------------------------------
-- View: quick inspection of what's in the store
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW corpus_summary AS
    SELECT
        source,
        COUNT(*)         AS chunk_count,
        SUM(char_count)  AS total_chars,
        MIN(created_at)  AS first_ingested
    FROM document_chunks
    GROUP BY source
    ORDER BY first_ingested DESC;
