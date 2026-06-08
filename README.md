# Vertex Knowledge Assistant — RAG System

FastAPI backend for a Retrieval-Augmented Generation (RAG) system over GCP docs.
Built on Cloud Run · pgvector · Vertex AI (text-embedding-005 + Gemini 2.5 Pro).

---

## Project layout

```
rag-system/
├── main.py              # FastAPI app — /health, /query endpoints
├── similarity_search.py # pgvector ANN search (also a CLI tool)
├── ingest.py            # Ingestion pipeline: load → chunk → embed → upsert
├── download_corpus.py   # Fetches GCP/k8s docs into corpus/
├── schema.sql           # Cloud SQL table + HNSW index definitions
├── setup_cloudsql.sh    # One-time Cloud SQL instance setup
├── Dockerfile           # Multi-stage image for Cloud Run
├── requirements.txt
├── .env                 # Local secrets (git-ignored)
└── .env.example         # Template — copy to .env and fill in
```

---

## Prerequisites

- Python 3.12 and the `.venv` already created in this repo
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — run `gcloud auth application-default login`
- Cloud SQL Auth Proxy binary already present in this folder (`./cloud-sql-proxy`)
- GCP project `rag-system-496722` with Vertex AI and Cloud Run APIs enabled

---

## Local development

### 1. Start the Cloud SQL Auth Proxy

Open a dedicated terminal and keep it running:

```bash
cd rag-system/
./cloud-sql-proxy rag-system-496722:us-central1:rag-pg --port=5432
```

You should see `Listening on 127.0.0.1:5432`. Leave this terminal open.

### 2. Activate the venv and start the API

In a second terminal:

```bash
cd rag-system/
source .venv/bin/activate
uvicorn main:app --reload --port 8080
```

The `--reload` flag restarts the server automatically on every file save.

### 3. Test the endpoints

```bash
# Liveness check
curl http://localhost:8080/health

# RAG query
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Kubernetes Pod?", "top_k": 5}' | python3 -m json.tool

# Interactive API docs (browser)
open http://localhost:8080/docs
```

Expected response shape:
```json
{
  "request_id": "...",
  "question": "What is a Kubernetes Pod?",
  "answer": "A Pod is the smallest deployable unit in Kubernetes...",
  "sources": [
    { "source": "docs-concepts-workloads-pods.txt", "chunk_index": 0,
      "text_preview": "...", "score": 0.87 }
  ],
  "latency_ms": 1842.5,
  "tokens_in": 2100,
  "tokens_out": 180,
  "retrieval_hit_count": 5
}
```

### 4. Test the similarity search CLI directly

```bash
# Single query
python similarity_search.py "How does Cloud Run autoscaling work?"

# Five demo queries at once
python similarity_search.py --demo --top-k 3
```

---

## Re-ingesting the corpus

Run this if you've added new documents to `corpus/` or changed chunking settings.
The proxy must be running (step 1 above).

```bash
# Full ingest (writes to pgvector)
python ingest.py --corpus corpus

# Inspect without writing to DB (saves embeddings.parquet locally)
python ingest.py --corpus corpus --dry-run
```

---

## Deploying to Cloud Run

### Option A — one command (recommended)

Cloud Build builds the image automatically from source:

```bash
gcloud run deploy rag-backend \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --add-cloudsql-instances rag-system-496722:us-central1:rag-pg \
  --set-env-vars "PROJECT_ID=rag-system-496722,REGION=us-central1,DB_HOST=/cloudsql/rag-system-496722:us-central1:rag-pg,DB_PORT=5432,DB_NAME=ragdb,DB_USER=rag_user" \
  --set-secrets "DB_PASSWORD=rag-db-password:latest" \
  --service-account rag-backend-sa@rag-system-496722.iam.gserviceaccount.com
```

> **Note on `DB_HOST`:** On Cloud Run the proxy runs as a Unix socket, so
> `DB_HOST` changes from `127.0.0.1` to the socket path
> `/cloudsql/PROJECT:REGION:INSTANCE`. The `--add-cloudsql-instances` flag
> mounts it automatically.



### Option B — build and push manually

```bash
# Build
docker build -t us-central1-docker.pkg.dev/rag-system-496722/rag/backend:latest .

# Push (create the Artifact Registry repo first if needed)
docker push us-central1-docker.pkg.dev/rag-system-496722/rag/backend:latest

# Deploy the pushed image
gcloud run deploy rag-backend \
  --image us-central1-docker.pkg.dev/rag-system-496722/rag/backend:latest \
  --region us-central1 \
  --add-cloudsql-instances rag-system-496722:us-central1:rag-pg \
  --set-env-vars "PROJECT_ID=rag-system-496722,REGION=us-central1,DB_HOST=/cloudsql/rag-system-496722:us-central1:rag-pg,DB_PORT=5432,DB_NAME=ragdb,DB_USER=rag_user" \
  --set-secrets "DB_PASSWORD=rag-db-password:latest"
```

### After deploying

```bash
# Get the service URL
gcloud run services describe rag-backend --region us-central1 \
  --format "value(status.url)"

# Hit the live endpoint
curl https://<SERVICE_URL>/health
```

---

## IAM roles the Cloud Run service account needs

| Role | Why |
|---|---|
| `roles/aiplatform.user` | Call Vertex AI (embeddings + Gemini) |
| `roles/cloudsql.client` | Connect to Cloud SQL via Auth Proxy |
| `roles/secretmanager.secretAccessor` | Read DB_PASSWORD from Secret Manager |

```bash
SA="rag-backend-sa@rag-system-496722.iam.gserviceaccount.com"
PROJECT="rag-system-496722"

for role in roles/aiplatform.user roles/cloudsql.client roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" --role="$role" --quiet
done
```

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ID` | *(required)* | GCP project ID |
| `REGION` | `us-central1` | GCP region |
| `LLM_MODEL` | `gemma-4-31b-it` | Vertex AI model for generation — swap without code changes |
| `TOP_K` | `5` | Default chunks retrieved per query |
| `DB_HOST` | `127.0.0.1` | `127.0.0.1` locally; socket path on Cloud Run |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `ragdb` | Database name |
| `DB_USER` | `rag_user` | Database user |
| `DB_PASSWORD` | *(required)* | Database password — use Secret Manager in prod |
| `OFFLINE_LLM` | `0` | Set to `1` to skip Gemini and return a stub |

---

## Structured log fields (Cloud Logging)

Every `/query` call emits one JSON line. Key fields:

| Field | Use |
|---|---|
| `request_id` | Correlate logs across services |
| `retrieval_hit_count` | `0` = recall failure; alert on this |
| `tokens_in` / `tokens_out` | Cost proxy — build a budget alert |
| `latency_ms` | SLO target: p95 < 3000ms |
| `retrieval_ms` / `llm_ms` | Latency breakdown for diagnosis |
| `top_source` | Which doc answered the question |
