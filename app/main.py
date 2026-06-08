"""
app/main.py — Vertex Knowledge Assistant (FastAPI)
==================================================
Entry point: uvicorn app.main:app --port 8080

Day 1: scaffold + /health
Day 3: full /query — retrieval → grounded prompt → LLM → structured logs
Day 6: hybrid retrieval (BM25 + dense + RRF + Gemini Flash re-ranking)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import (
    PROJECT_ID, REGION, LLM_MODEL, OFFLINE_LLM, USE_HYBRID, ENABLE_RERANK,
)
from app.models import QueryRequest, QueryResponse, SourceChunk
from app.llm import call_llm
from app.retrieval import dense_search, hybrid_search

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _json_log(severity: str, **fields: Any) -> None:
    """
    Emit a structured JSON log line to stdout.
    Cloud Logging parses and indexes every field automatically.
    """
    print(json.dumps({"severity": severity, **fields}), flush=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vertex Knowledge Assistant",
    version="0.6.0",
    description="RAG over GCP docs — hybrid retrieval + Gemini Flash re-ranking",
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.monotonic()

    response = await call_next(request)

    latency_ms = round((time.monotonic() - start) * 1000, 2)
    if request.url.path != "/query":
        _json_log(
            severity="INFO",
            event="http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def healthz():
    """Liveness probe for Cloud Run and uptime checks."""
    return {
        "status": "ok",
        "project": PROJECT_ID,
        "region": REGION,
        "model": LLM_MODEL,
        "offline_llm": OFFLINE_LLM,
        "hybrid_retrieval": USE_HYBRID,
        "reranking": ENABLE_RERANK,
    }


@app.get("/", tags=["ops"])
def root():
    return {
        "message": "Vertex Knowledge Assistant — Day 6",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(payload: QueryRequest, request: Request):
    """
    Full RAG query pipeline:
      1. Retrieve (hybrid: BM25 + dense + RRF + re-rank, or dense-only)
      2. Build grounded prompt with numbered citations
      3. Call LLM (Gemini)
      4. Return answer + sources + metadata
      5. Emit structured JSON log
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    t_total_start = time.monotonic()

    # ── 1. Retrieve ──────────────────────────────────────────────────────
    if USE_HYBRID:
        chunks, retrieval_timing = hybrid_search(
            payload.question,
            top_k=payload.top_k,
            retrieval_k=20,
            enable_rerank=ENABLE_RERANK,
        )
        retrieval_ms = retrieval_timing["total_ms"]
    else:
        chunks, retrieval_ms = dense_search(payload.question, top_k=payload.top_k)

    # ── 2+3. Generate ────────────────────────────────────────────────────
    t_llm_start = time.monotonic()
    answer, tokens_in, tokens_out = call_llm(payload.question, chunks)
    llm_ms = round((time.monotonic() - t_llm_start) * 1000, 1)

    total_latency_ms = round((time.monotonic() - t_total_start) * 1000, 2)

    # ── 4. Build response ────────────────────────────────────────────────
    sources = [
        SourceChunk(
            source=c["source"],
            chunk_index=c["chunk_index"],
            text_preview=c["text"][:300],
            score=c["cosine_similarity"],
        )
        for c in chunks
    ]

    # ── 5. Structured log ────────────────────────────────────────────────
    _json_log(
        severity="INFO",
        event="query",
        request_id=request_id,
        question_preview=payload.question[:80],
        retrieval_hit_count=len(chunks),
        retrieval_mode="hybrid" if USE_HYBRID else "dense",
        top_source=chunks[0]["source"] if chunks else None,
        top_score=chunks[0]["cosine_similarity"] if chunks else None,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        latency_ms=total_latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model=LLM_MODEL,
    )

    return QueryResponse(
        request_id=request_id,
        question=payload.question,
        answer=answer,
        sources=sources,
        latency_ms=total_latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        retrieval_hit_count=len(chunks),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    _json_log(
        severity="ERROR",
        event="unhandled_exception",
        request_id=request_id,
        path=str(request.url.path),
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "request_id": request_id,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )
