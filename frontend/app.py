"""
frontend/app.py — Streamlit Chat UI (Day 4)
============================================
Sends questions to the FastAPI RAG backend (/query) and renders:
  - The LLM answer with inline [1], [2] citation markers
  - Source chunks as clickable links (source filename + score)
  - Latency + token metadata in an expander

Run locally (backend must be up on port 8080):
    BACKEND_URL=http://localhost:8080 streamlit run frontend/app.py

Run against Cloud Run:
    BACKEND_URL=https://<backend-service-url> streamlit run frontend/app.py

Environment variables:
    BACKEND_URL   — base URL of the FastAPI backend (default: http://localhost:8080)
    APP_TITLE     — displayed in the browser tab (default: "RAG Knowledge Assistant")

Interview talking points:
  - Streamlit re-runs top-to-bottom on every interaction; st.session_state
    persists chat history across re-runs (similar to React state).
  - BACKEND_URL is injected at deploy time — no hardcoded service URLs.
    In Cloud Run, set this to the backend service URL so the frontend
    never goes through the public internet for internal calls.
  - Citations rendered as links let evaluators quickly verify grounding —
    key for demos where the interviewer asks "how do you know this is correct?"
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st
from streamlit_google_auth import Authenticate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")
APP_TITLE = os.getenv("APP_TITLE", "RAG Knowledge Assistant")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_S", "30"))

# ---------------------------------------------------------------------------
# Auth helper — attach OIDC identity token when running on Cloud Run
# ---------------------------------------------------------------------------
def _get_auth_headers() -> dict[str, str]:
    """
    When running on Cloud Run, fetch an OIDC identity token from the metadata
    server and attach it as Authorization: Bearer <token>.

    Cloud Run's --no-allow-unauthenticated requires every inbound request to
    carry a valid identity token signed by Google, even from other Cloud Run
    services. The IAM binding (roles/run.invoker) grants *permission*, but the
    token is what *proves* the caller's identity.

    When running locally (no metadata server), this returns an empty dict so
    plain httpx calls still work against a locally-running backend.
    """
    try:
        # The audience must be the exact backend Cloud Run URL (no trailing slash)
        audience = BACKEND_URL
        token_url = (
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts"
            f"/default/identity?audience={audience}"
        )
        resp = httpx.get(token_url, headers={"Metadata-Flavor": "Google"}, timeout=2)
        resp.raise_for_status()
        return {"Authorization": f"Bearer {resp.text.strip()}"}
    except Exception:
        # Not on GCP (local dev) — proceed without auth header
        return {}

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Google OAuth gate — runs before any app content is rendered
# ---------------------------------------------------------------------------
# Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET as env vars (or Secret Manager).
# SKIP_AUTH=1 disables the gate locally so you don't need creds to develop.
#
# How it works (vs IAP):
#   IAP = Google-managed proxy in front of the LB — auth happens outside your app.
#   This = OAuth 2.0 flow inside the app — equivalent user experience, no LB needed.
#   Both use Google accounts; both bounce unauthenticated users to Google login.
#   Interview talking point: app-layer auth is portable (works on any host);
#   IAP is GCP-native and zero-code but requires the LB infrastructure.
SKIP_AUTH = os.getenv("SKIP_AUTH", "0") == "1"

if not SKIP_AUTH:
    authenticator = Authenticate(
        secret_credentials_path=None,
        client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
        redirect_uri=os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501"),
        cookie_name="rag_auth",
        cookie_key=os.getenv("COOKIE_SECRET", "change-me-in-prod"),
        cookie_expiry_days=1,
    )
    authenticator.check_authentification()

    if not st.session_state.get("connected"):
        authenticator.login()
        st.stop()   # Render nothing else until the user is logged in

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, Any]] = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.session_state.top_k = st.slider(
        "Chunks to retrieve (top_k)",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k,
        help="More chunks = more context for the LLM, but slower and more tokens.",
    )

    st.divider()

    # Backend health check
    if st.button("🏥 Check backend health"):
        try:
            resp = httpx.get(f"{BACKEND_URL}/health", headers=_get_auth_headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"✅ Backend healthy\n\nProject: `{data.get('project')}`\nModel: `{data.get('model')}`")
            else:
                st.error(f"❌ Backend returned {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Could not reach backend: {e}")

    st.divider()

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Backend: `{BACKEND_URL}`")

    if not SKIP_AUTH and st.session_state.get("connected"):
        st.divider()
        st.caption(f"Signed in as **{st.session_state.get('user_info', {}).get('email', '')}**")
        authenticator.logout()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"🔍 {APP_TITLE}")
st.caption(
    "Ask questions about GCP, Kubernetes, or Vertex AI. "
    "Answers are grounded in the ingested document corpus with cited sources."
)

# ---------------------------------------------------------------------------
# Helper: render source citations
# (defined before the chat loop that calls it)
# ---------------------------------------------------------------------------
def _render_sources(sources: list[dict], meta: dict) -> None:
    """Render source citations as a nice expander with links and scores."""
    with st.expander(f"📚 Sources ({len(sources)} chunks retrieved)", expanded=False):
        for i, src in enumerate(sources, 1):
            score_pct = round(src["score"] * 100, 1)
            score_bar = "█" * int(score_pct / 10) + "░" * (10 - int(score_pct / 10))
            st.markdown(
                f"**[{i}]** `{src['source']}` — chunk `{src['chunk_index']}` "
                f"— score **{score_pct}%** `{score_bar}`"
            )
            st.caption(f"> {src['text_preview'][:250]}…")
            st.divider()

        # Metadata footer
        if meta:
            cols = st.columns(4)
            cols[0].metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
            cols[1].metric("Tokens in", meta.get("tokens_in", "—"))
            cols[2].metric("Tokens out", meta.get("tokens_out", "—"))
            cols[3].metric("Request ID", meta.get("request_id", "—")[:8] + "…")


# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # For assistant messages, render source citations + metadata
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"], msg.get("meta", {}))


# ---------------------------------------------------------------------------
# Chat input + query logic
# ---------------------------------------------------------------------------
if question := st.chat_input("Ask a question about the corpus…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            t0 = time.monotonic()
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "top_k": st.session_state.top_k},
                    headers=_get_auth_headers(),
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                sources = data["sources"]
                meta = {
                    "latency_ms": data["latency_ms"],
                    "tokens_in": data["tokens_in"],
                    "tokens_out": data["tokens_out"],
                    "request_id": data["request_id"],
                    "retrieval_hit_count": data["retrieval_hit_count"],
                }

                st.markdown(answer)
                _render_sources(sources, meta)

                # Persist to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "meta": meta,
                })

            except httpx.TimeoutException:
                err = f"⚠️ Request timed out after {REQUEST_TIMEOUT}s. The backend may be cold-starting."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except httpx.HTTPStatusError as e:
                err = f"⚠️ Backend error {e.response.status_code}: {e.response.text[:300]}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

            except Exception as e:
                err = f"⚠️ Unexpected error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
