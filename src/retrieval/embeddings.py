"""
Embedding generation for the RAG pipeline.

Supports two backends:
  1. OpenAI embeddings (text-embedding-3-small) — better quality, needs API key
  2. Sentence-transformers (all-MiniLM-L6-v2) — runs locally, no API key needed

The choice is made automatically: if OPENAI_API_KEY is set we use OpenAI,
otherwise we fall back to sentence-transformers.
"""

import asyncio
from typing import List

import numpy as np

from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Unified interface for generating text embeddings. Picks the
    best available backend at init time.
    """

    def __init__(self):
        self._backend = None
        self._openai_client = None
        self._st_model = None

        if settings.openai_api_key:
            self._init_openai()
        else:
            self._init_sentence_transformers()

    def _init_openai(self):
        """Set up OpenAI embeddings."""
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
            self._backend = "openai"
            self._dimension = settings.embedding_dimension
            logger.info("embeddings.using_openai", model=settings.openai_embedding_model)
        except ImportError:
            logger.warning("openai package not available, trying sentence-transformers")
            self._init_sentence_transformers()

    def _init_sentence_transformers(self):
        """Set up local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"
            self._st_model = SentenceTransformer(model_name)
            self._backend = "sentence_transformers"
            # MiniLM outputs 384-dim vectors
            self._dimension = 384
            logger.info("embeddings.using_sentence_transformers", model=model_name)
        except ImportError:
            raise RuntimeError(
                "Neither openai nor sentence-transformers is available. "
                "Install one of them to use the retrieval pipeline."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts. Returns a numpy
        array of shape (n_texts, dimension).
        """
        if self._backend == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dimension,)."""
        result = self.embed_texts([query])
        return result[0]

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Call OpenAI's embedding endpoint."""
        # batch in groups of 100 to stay within rate limits
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                input=batch,
                model=settings.openai_embedding_model,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Use sentence-transformers for local embedding."""
        embeddings = self._st_model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


# module-level singleton so we don't re-load models unnecessarily
_engine = None


def get_embedding_engine() -> EmbeddingEngine:
    """Get (or create) the global embedding engine instance."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
