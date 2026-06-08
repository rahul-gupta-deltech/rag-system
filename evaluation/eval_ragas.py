"""
evaluation/eval_ragas.py — Baseline RAG evaluation
===================================================
Usage:
    python -m evaluation.eval_ragas
    python -m evaluation.eval_ragas --category easy --limit 5
    OFFLINE_LLM=1 python -m evaluation.eval_ragas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from app.config import (
    TOP_K, OFFLINE_LLM, PROJECT_ID, REGION, LLM_MODEL,
    GOLDEN_SET_PATH, PARQUET_PATH, PROJECT_ROOT,
)

RESULTS_PATH = PROJECT_ROOT / "eval_results.json"

# ---------------------------------------------------------------------------
# Retriever — two modes
# ---------------------------------------------------------------------------

def _load_parquet():
    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    return df


def bm25_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    BM25 retrieval — no embedding API needed.

    Interview note: BM25 is a term-frequency / inverse-document-frequency
    ranking function. It excels at exact-keyword recall and is cheap to run.
    Dense (embedding) retrieval handles paraphrase and semantic similarity better.
    Hybrid = BM25 + dense + reciprocal rank fusion → best of both worlds (Day 6).
    """
    from rank_bm25 import BM25Okapi

    df = _load_parquet()
    corpus_tokens = [str(t).lower().split() for t in df["text"]]
    bm25 = BM25Okapi(corpus_tokens)

    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        row = df.iloc[i]
        results.append({
            "source":      row["source"],
            "chunk_index": int(row["chunk_index"]),
            "text":        row["text"],
            "score":       float(scores[i]),
        })
    return results


def embedding_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Cosine-similarity retrieval over precomputed parquet embeddings.

    Interview note: we re-use the same text-embedding-005 model that was
    used at ingest time with task_type=RETRIEVAL_QUERY (vs RETRIEVAL_DOCUMENT
    at ingest). Vertex AI applies different prompt prefixes per task type to
    tune the embedding space — mixing task types degrades recall.
    """
    import numpy as np
    import pandas as pd
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    import vertexai

    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    q_emb = model.get_embeddings(
        [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    )[0].values
    q_vec = np.array(q_emb, dtype=np.float32)

    df = _load_parquet()
    doc_vecs = np.stack(df["embedding"].values).astype(np.float32)

    # Cosine similarity = dot product when vectors are L2-normalised
    # text-embedding-005 returns L2-normalised vectors by default.
    cosine_sims = doc_vecs @ q_vec

    top_indices = np.argsort(cosine_sims)[::-1][:top_k]
    results = []
    for i in top_indices:
        row = df.iloc[i]
        results.append({
            "source":      row["source"],
            "chunk_index": int(row["chunk_index"]),
            "text":        row["text"],
            "score":       float(cosine_sims[i]),
        })
    return results


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Dispatcher: BM25 in offline mode, embedding otherwise."""
    if OFFLINE_LLM:
        return bm25_retrieve(query, top_k)
    return embedding_retrieve(query, top_k)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a knowledgeable technical assistant. Answer the user's question using ONLY
the provided context chunks. Cite sources inline as [1], [2], etc. If the answer
cannot be found in the context, say exactly:
"I don't have enough information in the provided documents to answer this question."
Keep answers concise (2-4 paragraphs max) and technically precise."""


def call_llm(question: str, chunks: list[dict]) -> str:
    """
    Call Vertex AI (Gemini) or return a stub in offline mode.

    Interview note: for eval pipelines, OFFLINE_LLM=1 lets you iterate on
    retrieval quality without burning LLM quota. Stub answers always trigger
    the "unanswerable" path, so faithfulness will be N/A and you can still
    measure retrieval recall separately.
    """
    if OFFLINE_LLM:
        # Stub: pretend the LLM said it can't answer — this lets us at least
        # verify the pipeline runs end-to-end without Vertex AI credentials.
        return "[OFFLINE] I don't have enough information in the provided documents to answer this question."

    from google import genai as gai

    client = gai.Client(vertexai=True, project=PROJECT_ID, location="global")

    context_lines = []
    for i, c in enumerate(chunks, 1):
        context_lines.append(f"[{i}] {c['source']} (chunk {c['chunk_index']})\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_lines)

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT ===\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return response.text or "(no response)"


# ---------------------------------------------------------------------------
# Metrics helpers (offline / pre-Ragas)
# ---------------------------------------------------------------------------

def _is_refusal(text: str) -> bool:
    """Detect 'I don't have enough information' responses."""
    refusal_phrases = [
        "don't have enough information",
        "cannot be found in the",
        "not in the provided",
        "[offline]",
    ]
    lower = text.lower()
    return any(p in lower for p in refusal_phrases)


def correctness_for_unanswerables(answer: str, expected_answerable: bool) -> Optional[float]:
    """
    For unanswerable/PII questions, 1.0 if the model correctly refuses,
    0.0 if it hallucinates an answer. Returns None for answerable questions.

    Interview note: tracking this separately from faithfulness catches a
    failure mode Ragas misses — hallucination on out-of-corpus queries.
    A model with high faithfulness on answerable Qs can still hallucinate
    confidently on unanswerable ones.
    """
    if expected_answerable:
        return None
    return 1.0 if _is_refusal(answer) else 0.0


def token_overlap_score(answer: str, ground_truth: str) -> float:
    """
    Simple word-overlap (F1) between answer and ground truth.
    A rough offline proxy for answer correctness (no LLM judge needed).

    Interview note: this is what SQuAD used pre-LLM-judge era. It fails on
    paraphrase and synonym-heavy answers. Ragas answer_correctness with an
    LLM judge is strictly better for production evals.
    """
    if _is_refusal(answer) and _is_refusal(ground_truth):
        return 1.0
    a_tokens = set(answer.lower().split())
    g_tokens = set(ground_truth.lower().split())
    if not g_tokens:
        return 0.0
    precision = len(a_tokens & g_tokens) / len(a_tokens) if a_tokens else 0.0
    recall    = len(a_tokens & g_tokens) / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Ragas evaluation
# ---------------------------------------------------------------------------

def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run Ragas faithfulness + answer_relevance.

    Interview talking points:
      faithfulness:      LLM extracts atomic claims from the answer, then checks
                         each claim against the context chunks. Score = fraction
                         of claims supported. High faithfulness = low hallucination.
      answer_relevance:  LLM generates N back-questions from the answer, then
                         measures cosine similarity between each back-question and
                         the original question. High score = answer addresses the Q.

    Both require an LLM backend. By default Ragas uses OpenAI — we override
    to use Vertex AI (Gemini) to stay on GCP.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print("  [WARN] ragas or datasets not installed — skipping Ragas metrics.")
        print("  Install with: pip install ragas datasets --break-system-packages")
        return {}

    # Wire Ragas to Vertex AI (Gemini) instead of OpenAI
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_google_vertexai import ChatVertexAI
        llm = LangchainLLMWrapper(ChatVertexAI(model=LLM_MODEL, project=PROJECT_ID, location=REGION))
        faithfulness.llm = llm
        answer_relevancy.llm = llm
        print(f"  [INFO] Ragas using Vertex AI ({LLM_MODEL})")
    except Exception as e:
        print(f"  [WARN] Could not configure Ragas LLM backend: {e}")
        print("  Install langchain-google-vertexai for full Ragas support.")

    data = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    try:
        result = evaluate(data, metrics=[faithfulness, answer_relevancy])
        return {
            "faithfulness":     round(float(result["faithfulness"]), 4),
            "answer_relevance": round(float(result["answer_relevancy"]), 4),
        }
    except Exception as e:
        print(f"  [WARN] Ragas evaluation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def load_golden_set(
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    items = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            item = json.loads(line.strip())
            if category_filter and item["category"] != category_filter:
                continue
            items.append(item)
            if limit and len(items) >= limit:
                break
    return items


def print_table(rows: list[dict]) -> None:
    """Pretty-print results as a fixed-width table."""
    header = f"{'ID':<12} {'Category':<14} {'Overlap':>7} {'Refusal✓':>9} {'Retrieval':>9}"
    print("\n" + "=" * 60)
    print("  BASELINE EVAL TABLE")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for r in rows:
        refusal = r.get("refusal_correct")
        refusal_str = f"{'✓' if refusal else '✗'}" if refusal is not None else "  N/A"
        print(
            f"{r['id']:<12} "
            f"{r['category']:<14} "
            f"{r['token_overlap']:>7.3f} "
            f"{refusal_str:>9} "
            f"{r['top_retrieval_score']:>9.3f}"
        )
    print("-" * 60)

    # Category summaries
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)

    print("\nSUMMARY BY CATEGORY")
    print(f"  {'Category':<16} {'N':>3} {'Avg Overlap':>12} {'Refusal Acc':>12}")
    print("  " + "-" * 45)
    for cat, items in sorted(cats.items()):
        avg_overlap = sum(i["token_overlap"] for i in items) / len(items)
        refusal_items = [i for i in items if i["refusal_correct"] is not None]
        ref_acc = (
            sum(i["refusal_correct"] for i in refusal_items) / len(refusal_items)
            if refusal_items else None
        )
        ref_str = f"{ref_acc:.3f}" if ref_acc is not None else "   N/A"
        print(f"  {cat:<16} {len(items):>3} {avg_overlap:>12.3f} {ref_str:>12}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Day 5: RAG baseline evaluation")
    parser.add_argument("--category", choices=["easy", "multi_hop", "unanswerable", "pii_trap"],
                        help="Filter to a single category")
    parser.add_argument("--limit", type=int, help="Cap number of questions")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip Ragas metrics (just run retrieval + LLM)")
    args = parser.parse_args()

    mode = "OFFLINE (BM25 + stub LLM)" if OFFLINE_LLM else f"ONLINE (embedding + {LLM_MODEL})"
    print(f"\n🔍  Starting eval | mode={mode}")
    if args.category:
        print(f"    category filter: {args.category}")

    items = load_golden_set(category_filter=args.category, limit=args.limit)
    print(f"    questions loaded: {len(items)}\n")

    rows       = []
    q_list     = []
    a_list     = []
    ctx_list   = []
    gt_list    = []

    for item in items:
        q   = item["question"]
        gt  = item["ground_truth"]
        exp = item["expected_answerable"]

        t0 = time.monotonic()
        chunks = retrieve(q)
        retrieval_ms = round((time.monotonic() - t0) * 1000, 1)

        answer = call_llm(q, chunks)

        top_score   = chunks[0]["score"] if chunks else 0.0
        overlap     = token_overlap_score(answer, gt)
        refusal_ok  = correctness_for_unanswerables(answer, exp)

        row = {
            "id":                  item["id"],
            "category":            item["category"],
            "question":            q,
            "answer":              answer,
            "ground_truth":        gt,
            "token_overlap":       round(overlap, 4),
            "refusal_correct":     refusal_ok,
            "top_retrieval_score": round(top_score, 4),
            "retrieval_ms":        retrieval_ms,
            "top_source":          chunks[0]["source"] if chunks else None,
            "context_chunks":      [c["text"] for c in chunks],
        }
        rows.append(row)

        q_list.append(q)
        a_list.append(answer)
        ctx_list.append([c["text"] for c in chunks])
        gt_list.append(gt)

        status = "✓" if (exp and not _is_refusal(answer)) or (not exp and _is_refusal(answer)) else "✗"
        print(f"  [{status}] {item['id']:<12} retrieval={retrieval_ms}ms  score={top_score:.3f}")

    print_table(rows)

    # ── Ragas metrics (skipped in offline mode or if --skip-ragas) ───────────
    ragas_scores: dict = {}
    if not OFFLINE_LLM and not args.skip_ragas:
        print("Running Ragas faithfulness + answer_relevance...")
        # Only run Ragas on answerable questions (unanswerable/PII have no context support)
        answerable_idx = [i for i, item in enumerate(items) if item["expected_answerable"]]
        if answerable_idx:
            ragas_scores = run_ragas(
                questions   = [q_list[i] for i in answerable_idx],
                answers     = [a_list[i] for i in answerable_idx],
                contexts    = [ctx_list[i] for i in answerable_idx],
                ground_truths = [gt_list[i] for i in answerable_idx],
            )
            if ragas_scores:
                print("\nRAGAS METRICS (answerable questions only)")
                for k, v in ragas_scores.items():
                    print(f"  {k:<24}: {v:.4f}")
    else:
        print("  [INFO] Ragas metrics skipped (OFFLINE_LLM=1 or --skip-ragas).")
        print("         Run without OFFLINE_LLM=1 to get faithfulness + answer_relevance scores.")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "mode":          mode,
        "total_questions": len(rows),
        "ragas_scores":  ragas_scores,
        "rows":          rows,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅  Results saved → {RESULTS_PATH}")
    print()


if __name__ == "__main__":
    main()
