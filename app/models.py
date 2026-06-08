"""
app/models.py — Pydantic request/response models
=================================================
Shared across the API and eval pipelines.
"""

from pydantic import BaseModel, Field

from app.config import TOP_K


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="Number of chunks to retrieve")

    model_config = {"json_schema_extra": {"example": {"question": "What is a Kubernetes Pod?"}}}


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    text_preview: str   # first 300 chars — enough for citation rendering
    score: float


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    retrieval_hit_count: int
