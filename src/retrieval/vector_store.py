"""
FAISS-backed vector store for retrieval-augmented generation.

This is the core of our RAG pipeline. Documents from the knowledge base
are embedded and stored in a FAISS index. At query time we embed the
user's question and do approximate nearest neighbor search to find
the most relevant documents.

The index is persisted to disk so we don't have to re-embed everything
on every restart. Use scripts/build_index.py to rebuild from scratch.
"""

import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.retrieval.embeddings import get_embedding_engine
from src.retrieval.knowledge_loader import Document, KnowledgeBaseLoader
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed — vector store will use brute force numpy search")


class VectorStore:
    """
    Manages the FAISS index and document metadata for retrieval.

    If FAISS isn't available we fall back to brute-force cosine
    similarity with numpy. Slower, but means the project runs
    out of the box without needing to install faiss-cpu.
    """

    def __init__(self):
        self._embedding_engine = get_embedding_engine()
        self._dimension = self._embedding_engine.dimension
        self._index = None
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None  # for numpy fallback

    def build_index(self, documents: List[Document]) -> None:
        """
        Embed all documents and build the search index from scratch.
        Call this when the knowledge base changes.
        """
        if not documents:
            logger.warning("vector_store.no_documents_to_index")
            return

        self._documents = documents
        texts = [doc.content for doc in documents]

        logger.info("vector_store.embedding_documents", count=len(texts))
        embeddings = self._embedding_engine.embed_texts(texts)
        logger.info("vector_store.embeddings_computed", shape=embeddings.shape)

        if FAISS_AVAILABLE:
            # L2 index with normalization — effectively cosine similarity
            faiss.normalize_L2(embeddings)
            self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(embeddings)
            logger.info("vector_store.faiss_index_built", vectors=self._index.ntotal)
        else:
            # numpy fallback
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = embeddings / norms
            logger.info("vector_store.numpy_index_built", vectors=len(self._embeddings))

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Find the most relevant documents for a query.
        Returns a list of (document, score) tuples sorted by relevance.
        """
        if not self._documents:
            logger.warning("vector_store.empty_index")
            return []

        query_embedding = self._embedding_engine.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        if FAISS_AVAILABLE and self._index is not None:
            faiss.normalize_L2(query_embedding)
            scores, indices = self._index.search(query_embedding, min(top_k, len(self._documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append((self._documents[idx], float(score)))
            return results
        elif self._embeddings is not None:
            # numpy cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(self._embeddings, query_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self._documents[idx], float(similarities[idx])))
            return results
        else:
            logger.warning("vector_store.no_index_available")
            return []

    def save(self, directory: Optional[str] = None) -> None:
        """Persist the index and document metadata to disk."""
        save_dir = directory or settings.vector_store_path
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # save document metadata
        meta_path = os.path.join(save_dir, "documents.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self._documents, f)

        if FAISS_AVAILABLE and self._index is not None:
            index_path = os.path.join(save_dir, "index.faiss")
            faiss.write_index(self._index, index_path)
            logger.info("vector_store.saved_faiss", path=index_path)
        elif self._embeddings is not None:
            emb_path = os.path.join(save_dir, "embeddings.npy")
            np.save(emb_path, self._embeddings)
            logger.info("vector_store.saved_numpy", path=emb_path)

        logger.info("vector_store.saved", directory=save_dir)

    def load(self, directory: Optional[str] = None) -> bool:
        """
        Load a previously saved index from disk.
        Returns True if successful, False otherwise.
        """
        load_dir = directory or settings.vector_store_path

        meta_path = os.path.join(load_dir, "documents.pkl")
        if not os.path.exists(meta_path):
            logger.info("vector_store.no_saved_index", directory=load_dir)
            return False

        with open(meta_path, "rb") as f:
            self._documents = pickle.load(f)

        if FAISS_AVAILABLE:
            index_path = os.path.join(load_dir, "index.faiss")
            if os.path.exists(index_path):
                self._index = faiss.read_index(index_path)
                logger.info("vector_store.loaded_faiss", vectors=self._index.ntotal)
                return True
        
        emb_path = os.path.join(load_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self._embeddings = np.load(emb_path)
            logger.info("vector_store.loaded_numpy", vectors=len(self._embeddings))
            return True

        logger.warning("vector_store.load_failed", directory=load_dir)
        return False

    def load_or_build(self) -> None:
        """
        Try to load a saved index; if that fails, build from the
        knowledge base files. This is the standard startup path.
        """
        if self.load():
            return

        logger.info("vector_store.building_from_scratch")
        loader = KnowledgeBaseLoader()
        documents = loader.load_all()

        if not documents:
            logger.warning("vector_store.no_documents_found")
            return

        self.build_index(documents)
        self.save()

    @property
    def document_count(self) -> int:
        return len(self._documents)


# module-level singleton
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or initialize the global vector store."""
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load_or_build()
    return _store
