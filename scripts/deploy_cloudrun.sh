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


# ── 2. Deploy backend Cloud Run service ────────────────────────────────
echo ""
echo "==> Deploying backend Cloud Run service..."
gcloud run deploy rag-backend \
  --image="${BACKEND_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --no-allow-unauthenticated \
  --min-instances=0 \
  --max-instances=5 \
  --cpu=1 \
  --memory=1Gi \
  --timeout=60 \
  --concurrency=80 \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},REGION=${REGION},USE_HYBRID=1,ENABLE_RERANK=1" \
  --set-secrets="DB_PASSWORD=${DB_PASSWORD_SECRET:-rag-db-password}:latest" \
  --service-account="rag-backend@${PROJECT_ID}.iam.gserviceaccount.com"

# Capture backend URL — used to wire the frontend
BACKEND_URL=$(gcloud run services describe rag-backend \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")
echo "==> Backend URL: ${BACKEND_URL}"

# ── 3. Build + push frontend ───────────────────────────────────────────
FRONTEND_IMAGE="${REPO}/frontend:$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
echo ""
echo "==> Building frontend → ${FRONTEND_IMAGE}"
docker build -t "${FRONTEND_IMAGE}" "$(dirname "$0")/../frontend"
docker push "${FRONTEND_IMAGE}"

# ── 4. Deploy frontend Cloud Run service ──────────────────────────────
# Interview talking point:
#   --allow-unauthenticated here lets Cloud Run accept the request at the
#   network layer. IAP (set up separately) adds the auth layer in front.
#   Without IAP, the URL would be publicly accessible. With IAP, Cloud Run
#   still sees the request but IAP has already verified the Google identity.
echo ""
echo "==> Deploying frontend Cloud Run service..."
gcloud run deploy rag-frontend \
  --image="${FRONTEND_IMAGE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=3 \
  --cpu=1 \
  --memory=512Mi \
  --timeout=30 \
  --concurrency=20 \
  --set-env-vars="BACKEND_URL=${BACKEND_URL},APP_TITLE=RAG Knowledge Assistant"

FRONTEND_URL=$(gcloud run services describe rag-frontend \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")
echo "==> Frontend URL: ${FRONTEND_URL}"

# ── 5. Grant frontend SA permission to invoke backend ─────────────────
# Interview talking point:
#   Cloud Run service-to-service auth uses OIDC tokens.
#   The frontend's service account gets roles/run.invoker on the backend.
#   The frontend fetches its own ID token and sends it as Authorization: Bearer <token>.
#   Without this binding, the backend returns 403.
echo ""
echo "==> Granting frontend SA invoke permission on backend..."
gcloud run services add-iam-policy-binding rag-backend \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:rag-frontend@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.invoker"

echo ""
echo "✅ Deployment complete!"
echo "   Backend:  ${BACKEND_URL}"
echo "   Frontend: ${FRONTEND_URL}"
echo ""
echo "Next: run scripts/setup_iap.sh to put IAP in front of the frontend."
