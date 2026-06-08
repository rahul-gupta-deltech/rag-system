#!/usr/bin/env bash
# setup_cloudsql.sh — One-time Cloud SQL setup for the RAG vector store.
#
# Prerequisites:
#   gcloud auth application-default login
#   gcloud config set project YOUR_PROJECT_ID
#
# Run:
#   chmod +x setup_cloudsql.sh
#   ./setup_cloudsql.sh
#
# After this script finishes, copy the printed connection string into your .env

set -euo pipefail

# ── Config (override via env or edit here) ──────────────────────────────────
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
INSTANCE_NAME="${INSTANCE_NAME:-rag-pg}"
DB_NAME="${DB_NAME:-ragdb}"
DB_USER="${DB_USER:-rag_user}"
# Password is prompted below; never hard-code it.

echo "================================================================"
echo " Vertex Knowledge Assistant — Cloud SQL + pgvector setup"
echo " Project : $PROJECT_ID"
echo " Region  : $REGION"
echo " Instance: $INSTANCE_NAME"
echo "================================================================"

# ── 1. Enable APIs ───────────────────────────────────────────────────────────
echo ""
echo "[1/6] Enabling required APIs..."
gcloud services enable \
  sqladmin.googleapis.com \
  sql-component.googleapis.com \
  --project="$PROJECT_ID" \
  --quiet

# ── 2. Create Cloud SQL instance (PostgreSQL 16) ─────────────────────────────
echo ""
echo "[2/6] Creating Cloud SQL instance '$INSTANCE_NAME' (this takes ~5 min)..."
gcloud sql instances create "$INSTANCE_NAME" \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region="$REGION" \
  --storage-type=SSD \
  --storage-size=10GB \
  --no-storage-auto-increase \
  --insights-config-query-insights-enabled \
  --project="$PROJECT_ID" \
  --quiet || echo "  (instance may already exist — continuing)"

# ── 3. Create the database ───────────────────────────────────────────────────
echo ""
echo "[3/6] Creating database '$DB_NAME'..."
gcloud sql databases create "$DB_NAME" \
  --instance="$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --quiet || echo "  (database may already exist — continuing)"

# ── 4. Create the user ───────────────────────────────────────────────────────
echo ""
echo "[4/6] Creating database user '$DB_USER'..."
read -rsp "Enter password for '$DB_USER': " DB_PASSWORD
echo ""

gcloud sql users create "$DB_USER" \
  --instance="$INSTANCE_NAME" \
  --password="$DB_PASSWORD" \
  --project="$PROJECT_ID" \
  --quiet || echo "  (user may already exist — continuing)"

# ── 5. Print connection info ─────────────────────────────────────────────────
CONN_NAME="$PROJECT_ID:$REGION:$INSTANCE_NAME"

echo ""
echo "[5/6] Cloud SQL instance ready."
echo ""
echo "================================================================"
echo " Add these to your .env:"
echo ""
echo "  PROJECT_ID=$PROJECT_ID"
echo "  REGION=$REGION"
echo "  CLOUD_SQL_CONNECTION_NAME=$CONN_NAME"
echo "  DB_NAME=$DB_NAME"
echo "  DB_USER=$DB_USER"
echo "  DB_PASSWORD=<the password you just set>"
echo "================================================================"

# ── 6. Apply schema via Cloud SQL Auth Proxy ─────────────────────────────────
echo ""
echo "[6/6] To apply schema.sql, start the Cloud SQL Auth Proxy then run:"
echo ""
echo "  # Terminal A — start proxy:"
echo "  cloud-sql-proxy $CONN_NAME --port=5432"
echo ""
echo "  # Terminal B — apply schema:"
echo "  sql -h 127.0.0.1 -p 5432 -d $DB_NAME -U $DB_USER -f schema.sql "
echo ""
echo "Or install the proxy with:"
echo "  curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.14.1/cloud-sql-proxy.darwin.amd64"
echo "  chmod +x cloud-sql-proxy"
