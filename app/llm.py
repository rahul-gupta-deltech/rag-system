"""
app/llm.py — LLM calling logic
===============================
Wraps Vertex AI (Gemini / Gemma) behind a clean interface.
Swap LLM_MODEL env var to change models — zero code changes.

Interview talking point: this is the LLMProvider abstraction pattern.
In Phase 2 you'd extract an ABC and add a second provider (e.g., Claude)
so swapping is one config line.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import PROJECT_ID, LLM_MODEL, OFFLINE_LLM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI client (lazy init)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None and PROJECT_ID != "unknown-project":
        from google import genai
        _client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
        log.info(f"Vertex AI client initialised (model={LLM_MODEL})")
    return _client


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a knowledgeable technical assistant. Answer the user's question using ONLY
the provided context chunks. Cite sources inline as [1], [2], etc., matching the
chunk numbers given. If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents to answer this question."
Keep answers concise (2–4 paragraphs max) and technically precise.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(question: str, context_chunks: list[dict]) -> tuple[str, int, int]:
    """
    Build a grounded prompt and call the configured LLM via Vertex AI.

    Returns (answer_text, tokens_in, tokens_out).

    Prompt structure:
      - System instruction: role + citation rules + hedging
      - Numbered context chunks with source labels
      - Question last — reduces lost-in-the-middle degradation
    """
    if OFFLINE_LLM:
        stub = (
            f"[OFFLINE MODE] Question received: '{question}'. "
            f"Retrieved {len(context_chunks)} chunks. "
            "Set OFFLINE_LLM=0 and ensure Vertex AI ADC is configured for a real answer."
        )
        return stub, 0, 0

    # Build context block
    context_lines = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = f"{chunk['source']} (chunk {chunk['chunk_index']})"
        context_lines.append(f"[{i}] {source_label}\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_lines)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )

    client = _get_client()
    response = client.models.generate_content(model=LLM_MODEL, contents=full_prompt)

    answer = response.text or "(no response)"
    tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
    tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    return answer, tokens_in, tokens_out
