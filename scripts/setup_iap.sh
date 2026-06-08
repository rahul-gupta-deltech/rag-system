#!/usr/bin/env bash
# scripts/setup_iap.sh
# -----------------------------------------------------------------------
# Day 4: Put IAP (Identity-Aware Proxy) in front of the Cloud Run frontend
#
# Architecture after this script:
#   [user browser]
#       │
#       ▼ HTTPS (custom domain or LB IP)
#   [Google Cloud Load Balancer]  ← SSL cert managed by Google
#       │
#       ▼
#   [IAP]  ← Google account login required here
#       │   (IAP checks user is in the allowed IAM binding)
#       ▼
#   [Cloud Run backend: rag-frontend]
#       │
#       ▼ (internal, OIDC token auth)
#   [Cloud Run backend: rag-backend]
#
# Interview talking points:
#   - IAP works at layer 7 (HTTP). It intercepts requests before they reach
#     Cloud Run, performs an OIDC login flow, then forwards with
#     X-Goog-Authenticated-User-Email and X-Goog-IAP-JWT-Assertion headers.
#   - Cloud Run --allow-unauthenticated is still set — the LB/IAP layer
#     enforces auth, not Cloud Run itself. For belt-and-suspenders security,
#     you can also verify the IAP JWT in your app.
#   - IAP is NOT the same as Cloud Run's built-in auth (--no-allow-unauthenticated).
#     IAP = human user auth (Google accounts / Workspace). Cloud Run auth =
#     service account / OIDC for service-to-service.
#   - Unauthenticated requests to the LB get a 302 to accounts.google.com.
#     After login, IAP checks the IAM binding and allows/denies.
#
# Prerequisites:
#   1. A domain you control (for the SSL cert + HTTPS LB)
#   2. OAuth consent screen configured in the project
#   3. An OAuth 2.0 client ID created for IAP (Web application type)
#
# Usage:
#   export PROJECT_ID=your-project-id
#   export REGION=us-central1
#   export DOMAIN=rag.yourdomain.com          # domain pointing to LB IP
#   export IAP_CLIENT_ID=<oauth-client-id>
#   export IAP_CLIENT_SECRET=<oauth-client-secret>
#   export ALLOWED_EMAIL=you@example.com      # who gets access
#   bash scripts/setup_iap.sh
# -----------------------------------------------------------------------
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${DOMAIN:?Set DOMAIN — the hostname for the LB}"
: "${IAP_CLIENT_ID:?Set IAP_CLIENT_ID}"
: "${IAP_CLIENT_SECRET:?Set IAP_CLIENT_SECRET}"
: "${ALLOWED_EMAIL:?Set ALLOWED_EMAIL}"

echo "==> Project: $PROJECT_ID  Region: $REGION"
echo "==> Domain:  $DOMAIN"
echo "==> Allowed: $ALLOWED_EMAIL"
echo ""

# ── 1. Reserve a global static IP ─────────────────────────────────────
echo "==> Reserving global static IP..."
gcloud compute addresses create rag-frontend-ip \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (IP already exists)"

LB_IP=$(gcloud compute addresses describe rag-frontend-ip \
  --global --project="${PROJECT_ID}" --format="value(address)")
echo "   LB IP: ${LB_IP}"
echo "   → Point your DNS A record: ${DOMAIN} → ${LB_IP}"

# ── 2. Create a Serverless NEG for the Cloud Run frontend ─────────────
# A Serverless NEG (Network Endpoint Group) lets the HTTPS LB route traffic
# to a Cloud Run service. Interview note: NEGs are how GCP LBs talk to
# serverless backends (Cloud Run, App Engine, Cloud Functions).
echo ""
echo "==> Creating Serverless NEG for rag-frontend..."
gcloud compute network-endpoint-groups create rag-frontend-neg \
  --region="${REGION}" \
  --network-endpoint-type=serverless \
  --cloud-run-service=rag-frontend \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (NEG already exists)"

# ── 3. Backend service ─────────────────────────────────────────────────
echo "==> Creating backend service..."
gcloud compute backend-services create rag-frontend-bs \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (backend service already exists)"

gcloud compute backend-services add-backend rag-frontend-bs \
  --global \
  --network-endpoint-group=rag-frontend-neg \
  --network-endpoint-group-region="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (backend already added)"

# ── 4. URL map + HTTPS proxy + forwarding rule ─────────────────────────
echo "==> Creating URL map..."
gcloud compute url-maps create rag-frontend-urlmap \
  --default-service=rag-frontend-bs \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (URL map already exists)"

echo "==> Creating managed SSL cert..."
gcloud compute ssl-certificates create rag-frontend-cert \
  --domains="${DOMAIN}" \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (cert already exists)"

echo "==> Creating HTTPS target proxy..."
gcloud compute target-https-proxies create rag-frontend-https-proxy \
  --url-map=rag-frontend-urlmap \
  --ssl-certificates=rag-frontend-cert \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (proxy already exists)"

echo "==> Creating forwarding rule..."
gcloud compute forwarding-rules create rag-frontend-fwd \
  --target-https-proxy=rag-frontend-https-proxy \
  --address=rag-frontend-ip \
  --ports=443 \
  --global \
  --project="${PROJECT_ID}" 2>/dev/null || echo "   (forwarding rule already exists)"

# ── 5. Enable IAP on the backend service ──────────────────────────────
echo ""
echo "==> Enabling IAP on backend service..."
gcloud iap web enable \
  --resource-type=backend-services \
  --service=rag-frontend-bs \
  --oauth2-client-id="${IAP_CLIENT_ID}" \
  --oauth2-client-secret="${IAP_CLIENT_SECRET}" \
  --project="${PROJECT_ID}"

# ── 6. Grant IAP access to the allowed user ───────────────────────────
# Interview talking point:
#   roles/iap.httpsResourceAccessor grants a user the ability to pass through
#   IAP. Without this binding, IAP returns 403 even after successful Google login.
#   You can bind to:
#     user:email            — individual
#     group:email           — Google Group (recommended for teams)
#     domain:yourdomain.com — whole Workspace domain
echo ""
echo "==> Granting IAP access to ${ALLOWED_EMAIL}..."
gcloud iap web add-iam-policy-binding \
  --resource-type=backend-services \
  --service=rag-frontend-bs \
  --member="user:${ALLOWED_EMAIL}" \
  --role="roles/iap.httpsResourceAccessor" \
  --project="${PROJECT_ID}"

echo ""
echo "✅ IAP setup complete!"
echo ""
echo "DNS: Point ${DOMAIN} → ${LB_IP} (A record)"
echo "     SSL cert provisioning takes ~15 minutes after DNS propagates."
echo ""
echo "Test unauthenticated rejection:"
echo "  curl -I https://${DOMAIN}/"
echo "  # Expected: HTTP/2 302 → accounts.google.com (or 401 if curl doesn't follow redirects)"
echo ""
echo "Test authenticated (use gcloud to get a token):"
echo "  TOKEN=\$(gcloud auth print-identity-token)"
echo "  curl -H \"Authorization: Bearer \$TOKEN\" https://${DOMAIN}/healthz"
echo "  # Expected: HTTP/2 200 {\"status\": \"ok\", ...}"
