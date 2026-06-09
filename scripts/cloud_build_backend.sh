#!/usr/bin/env bash
# scripts/deploy_cloudrun.sh
# -----------------------------------------------------------------------
# Day 4: Build + push + deploy backend AND frontend to Cloud Run
#
# Architecture:
#   [user] → [IAP] → [frontend Cloud Run] → [backend Cloud Run] → [Cloud SQL]
#
# The frontend and backend are separate Cloud Run services.
# The frontend calls the backend using its internal service URL (no public DNS).
# No load balancer or Ingress needed — Cloud Run provides HTTPS endpoints.
#
# Interview talking point:
#   Cloud Run service-to-service auth: the frontend uses its own identity
#   token to call the backend (OIDC). Backend is set to --no-allow-unauthenticated
#   so only the frontend SA can invoke it. Frontend is protected by IAP.
#
# Usage:
#   export PROJECT_ID=your-project-id
#   export REGION=us-central1
#   export DB_PASSWORD_SECRET=rag-db-password   # Secret Manager secret name
#   bash scripts/deploy_cloudrun.sh
# -----------------------------------------------------------------------
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/rag-system"

echo "==> Project: $PROJECT_ID  Region: $REGION"
echo "==> Artifact Registry repo: $REPO"
echo ""

# ── 0. Ensure Artifact Registry repo exists ────────────────────────────
echo "==> Ensuring Artifact Registry repo 'rag' exists..."
gcloud artifacts repositories create rag \
  --repository-format=docker \
  --location="${REGION}" \
  --description="RAG system images" \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (repo already exists)"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── 1. Build + push backend ────────────────────────────────────────────
BACKEND_IMAGE="${REPO}/backend:$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
echo ""
echo "==> Building backend → ${BACKEND_IMAGE}"
# Build from rag-system/ root (where the backend Dockerfile lives)
gcloud builds submit --project="${PROJECT_ID}" --region="${REGION}" --tag="${BACKEND_IMAGE}" "$(dirname "$0")/.."

