"""Layered repository retrieval: explicit mentions, symbols, dependency graph, BM25, embeddings."""

from fix_issue.retrieval.embeddings import (
    EmbeddingBackend,
    EmbeddingCache,
    LocalNgramEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from fix_issue.retrieval.index import RepoIndex, Symbol
from fix_issue.retrieval.mentions import Mentions, extract_mentions
from fix_issue.retrieval.retriever import (
    RepositoryRetriever,
    RetrievalResult,
    RetrievedFile,
    retrieval_metrics,
)

__all__ = [
    "EmbeddingBackend", "EmbeddingCache", "LocalNgramEmbedder", "OpenAICompatibleEmbedder", "build_embedder",
    "RepoIndex", "Symbol", "Mentions", "extract_mentions",
    "RepositoryRetriever", "RetrievalResult", "RetrievedFile", "retrieval_metrics",
]
