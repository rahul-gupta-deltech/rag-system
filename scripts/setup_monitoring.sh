#!/usr/bin/env bash
# scripts/setup_monitoring.sh
# -----------------------------------------------------------------------
# Day 4: Cloud Monitoring uptime check + alert policy for /healthz
#
# What this does:
#   1. Creates an HTTPS uptime check that probes /healthz every 60s
#      from multiple GCP regions (gives you multi-region health signal)
#   2. Creates a notification channel (email)
#   3. Creates an alert policy that fires if uptime check fails for > 1 min
#
# Prerequisites:
#   gcloud auth login && gcloud config set project $PROJECT_ID
#   roles needed: roles/monitoring.admin on the project
#
# Usage:
#   export PROJECT_ID=your-project-id
#   export FRONTEND_URL=https://<frontend-service-url>   # Cloud Run URL
#   export ALERT_EMAIL=you@example.com
#   bash scripts/setup_monitoring.sh
# -----------------------------------------------------------------------
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${FRONTEND_URL:?Set FRONTEND_URL — Cloud Run frontend URL}"
: "${ALERT_EMAIL:?Set ALERT_EMAIL}"

# Strip protocol for the host header
FRONTEND_HOST=$(echo "$FRONTEND_URL" | sed 's|https://||')

echo "==> Project:  $PROJECT_ID"
echo "==> Frontend: $FRONTEND_URL"
echo "==> Alert to: $ALERT_EMAIL"
echo ""

# ── 1. Uptime check ────────────────────────────────────────────────────
# Interview talking point:
#   Cloud Monitoring uptime checks emit a metric:
#     monitoring.googleapis.com/uptime_check/check_passed
#   Cloud Run /healthz responding 200 = check passes.
#   Checks run from multiple regions (us-central1, europe-west1, asia-east1, etc.)
#   so you get global availability visibility, not just from one PoP.
#
#   gcloud doesn't yet have a first-class `uptime-checks create` verb, so we
#   use the Monitoring API via curl (or the Python client). Here we use the
#   REST API via gcloud auth print-access-token.

ACCESS_TOKEN=$(gcloud auth print-access-token)

UPTIME_CHECK_BODY=$(cat <<EOF
{
  "displayName": "rag-frontend-healthz",
  "httpCheck": {
    "path": "/healthz",
    "port": 443,
    "useSsl": true,
    "validateSsl": true,
    "headers": {"Host": "${FRONTEND_HOST}"}
  },
  "monitoredResource": {
    "type": "uptime_url",
    "labels": {
      "project_id": "${PROJECT_ID}",
      "host": "${FRONTEND_HOST}"
    }
  },
  "period": "60s",
  "timeout": "10s",
  "selectedRegions": [
    "USA", "EUROPE", "ASIA_PACIFIC"
  ]
}
EOF
)

echo "==> Creating uptime check..."
UPTIME_RESPONSE=$(curl -s -X POST \
  "https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}/uptimeCheckConfigs" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${UPTIME_CHECK_BODY}")

echo "$UPTIME_RESPONSE" | python3 -m json.tool
UPTIME_CHECK_NAME=$(echo "$UPTIME_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
echo "==> Uptime check created: $UPTIME_CHECK_NAME"

# ── 2. Notification channel (email) ────────────────────────────────────
echo ""
echo "==> Creating email notification channel..."
CHANNEL_RESPONSE=$(curl -s -X POST \
  "https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}/notificationChannels" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"email\",
    \"displayName\": \"RAG alerts — ${ALERT_EMAIL}\",
    \"labels\": {\"email_address\": \"${ALERT_EMAIL}\"}
  }")

echo "$CHANNEL_RESPONSE" | python3 -m json.tool
CHANNEL_NAME=$(echo "$CHANNEL_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
echo "==> Notification channel: $CHANNEL_NAME"

# ── 3. Alert policy ────────────────────────────────────────────────────
# Interview talking point:
#   Alert policy = condition + notification channels + alerting strategy.
#   The condition here is:
#     metric: monitoring.googleapis.com/uptime_check/check_passed
#     filter:  check_id matches our uptime check
#     threshold: < 1 (i.e. the check is failing)
#     duration: 60s  → fires after 1 minute of failure
#
#   In a real system you'd also have:
#     - Latency SLO alerts (p95 > 3s for 5 minutes)
#     - Log-based metric alerts (ERROR rate > N/min)
#     - Custom business metric alerts (retrieval_hit_count=0 rate rising)
echo ""
echo "==> Creating alert policy..."

# Extract the check_id from the full name (projects/.../uptimeCheckConfigs/<id>)
UPTIME_CHECK_ID=$(echo "$UPTIME_CHECK_NAME" | awk -F'/' '{print $NF}')

ALERT_BODY=$(cat <<EOF
{
  "displayName": "RAG frontend /healthz failing",
  "conditions": [
    {
      "displayName": "Uptime check failing > 1 min",
      "conditionThreshold": {
        "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND metric.labels.check_id=\"${UPTIME_CHECK_ID}\"",
        "comparison": "COMPARISON_LT",
        "thresholdValue": 1,
        "duration": "60s",
        "aggregations": [
          {
            "alignmentPeriod": "60s",
            "perSeriesAligner": "ALIGN_NEXT_OLDER",
            "crossSeriesReducer": "REDUCE_COUNT_TRUE",
            "groupByFields": ["resource.labels.*"]
          }
        ]
      }
    }
  ],
  "notificationChannels": ["${CHANNEL_NAME}"],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true
}
EOF
)

ALERT_RESPONSE=$(curl -s -X POST \
  "https://monitoring.googleapis.com/v3/projects/${PROJECT_ID}/alertPolicies" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${ALERT_BODY}")

echo "$ALERT_RESPONSE" | python3 -m json.tool
ALERT_NAME=$(echo "$ALERT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")
echo ""
echo "==> Alert policy created: $ALERT_NAME"
echo ""
echo "✅ Monitoring setup complete!"
echo "   Uptime check:  $UPTIME_CHECK_NAME"
echo "   Notif channel: $CHANNEL_NAME"
echo "   Alert policy:  $ALERT_NAME"
echo ""
echo "   View in console: https://console.cloud.google.com/monitoring/uptime?project=${PROJECT_ID}"
