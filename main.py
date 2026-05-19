import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Cloud Logging picks up JSON lines natively
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vertex Knowledge Assistant", version="0.1.0")

PROJECT_ID = os.getenv("PROJECT_ID", "unknown-project")
REGION = os.getenv("REGION", "us-central1")


# ---------------------------------------------------------------------------
# Middleware: attach a request_id and log every request
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.monotonic()

    response = await call_next(request)

    latency_ms = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["ops"])
def healthz():
    """Liveness probe — Cloud Run / uptime checks hit this."""
    return {"status": "ok", "project": PROJECT_ID, "region": REGION}


@app.get("/", tags=["ops"])
def root():
    return {
        "message": "Vertex Knowledge Assistant — Day 1 scaffold",
        "docs": "/docs",
        "health": "/healthz",
    }


@app.post("/query", tags=["rag"])
async def query(request: Request, payload: dict):
    """
    Placeholder /query endpoint.
    Day 3 will wire this to Vertex AI + Vector Search.
    """
    question = payload.get("question", "")
    request_id = request.state.request_id

    # TODO (Day 3): call Vertex AI Vector Search + Gemini
    return JSONResponse(
        {
            "request_id": request_id,
            "question": question,
            "answer": "RAG pipeline not yet wired — check back on Day 3!",
            "sources": [],
        }
    )
