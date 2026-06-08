"""
app/retrieval — Retrieval package
=================================
Exports the two main search functions so callers can do:
    from app.retrieval import dense_search, hybrid_search
"""

from app.retrieval.dense import search as dense_search
from app.retrieval.hybrid import hybrid_search

__all__ = ["dense_search", "hybrid_search"]
