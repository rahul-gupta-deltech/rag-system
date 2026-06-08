"""
app/config.py — Centralised configuration
==========================================
All environment variables are read here and imported by other modules.
Single source of truth — no scattered os.getenv() calls.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# GCP
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "unknown-project")
REGION = os.getenv("REGION", "us-central1")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-2.0-flash-001")
EMBEDDING_MODEL = "text-embedding-005"
EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
USE_HYBRID = os.getenv("USE_HYBRID", "1") == "1"
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "0") == "1"
RRF_K = 60  # Reciprocal Rank Fusion constant

# ---------------------------------------------------------------------------
# Database (pgvector on Cloud SQL)
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
OFFLINE_LLM = os.getenv("OFFLINE_LLM", "0") == "1"
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "auto")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PROJECT_ROOT
PARQUET_PATH = _PROJECT_ROOT / "embeddings.parquet"
GOLDEN_SET_PATH = _PROJECT_ROOT / "golden_set.jsonl"
CORPUS_DIR = _PROJECT_ROOT / "corpus"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
