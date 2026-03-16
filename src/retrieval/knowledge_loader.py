"""
Knowledge base loader — reads company docs, FAQs, and policies
from JSON files and prepares them for indexing.

The knowledge base lives in the knowledge_base/ directory as simple
JSON files. Each file contains an array of documents, where each
document has a "title", "content", and optionally "category" and
"metadata" fields.

This loader handles:
  - reading and parsing the JSON files
  - chunking long documents into smaller pieces for better retrieval
  - building the text + metadata pairs that go into the vector store
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """A single chunk of knowledge base content."""
    content: str
    title: str = ""
    category: str = ""
    source_file: str = ""
    chunk_index: int = 0
    metadata: Dict = field(default_factory=dict)

    @property
    def display_source(self) -> str:
        """Readable source identifier for citation in responses."""
        parts = [self.source_file]
        if self.title:
            parts.append(self.title)
        return " > ".join(parts)


class KnowledgeBaseLoader:
    """
    Loads and chunks documents from the knowledge base directory.

    Documents are split into overlapping chunks so that each chunk
    stays within a reasonable token count for the embedding model,
    and overlap ensures we don't lose context at chunk boundaries.
    """

    def __init__(
        self,
        kb_directory: str = "knowledge_base",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.kb_directory = kb_directory
        self.chunk_size = chunk_size        # in characters (rough approximation)
        self.chunk_overlap = chunk_overlap

    def load_all(self) -> List[Document]:
        """
        Load every JSON file in the knowledge base directory and
        return a flat list of chunked documents.
        """
        kb_path = Path(self.kb_directory)
        if not kb_path.exists():
            logger.warning("knowledge_base.directory_missing", path=str(kb_path))
            return []

        all_docs = []
        json_files = sorted(kb_path.glob("*.json"))

        if not json_files:
            logger.warning("knowledge_base.no_files_found", path=str(kb_path))
            return []

        for filepath in json_files:
            try:
                docs = self._load_file(filepath)
                all_docs.extend(docs)
                logger.info(
                    "knowledge_base.loaded_file",
                    file=filepath.name,
                    documents=len(docs),
                )
            except Exception as exc:
                logger.error(
                    "knowledge_base.load_error",
                    file=filepath.name,
                    error=str(exc),
                )

        logger.info("knowledge_base.total_documents", count=len(all_docs))
        return all_docs

    def _load_file(self, filepath: Path) -> List[Document]:
        """Parse a single JSON file and chunk its contents."""
        with open(filepath, "r") as fh:
            data = json.load(fh)

        # the file should contain a list of document objects
        if not isinstance(data, list):
            data = [data]

        documents = []
        for item in data:
            title = item.get("title", "")
            content = item.get("content", "")
            category = item.get("category", "")
            metadata = item.get("metadata", {})

            if not content.strip():
                continue

            chunks = self._chunk_text(content)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    title=title,
                    category=category,
                    source_file=filepath.name,
                    chunk_index=idx,
                    metadata=metadata,
                )
                documents.append(doc)

        return documents

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks. We try to break on sentence
        boundaries when possible to keep chunks coherent.
        """
        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        # split into sentences first (rough heuristic)
        sentences = self._split_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # save current chunk
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # keep the last few sentences for overlap
                overlap_chars = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    overlap_chars += len(s)
                    overlap_sentences.insert(0, s)
                    if overlap_chars >= self.chunk_overlap:
                        break

                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        # don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Basic sentence splitter. Handles common abbreviations poorly
        but works well enough for FAQ-style content. For a real product
        we'd use spacy or nltk.sent_tokenize.
        """
        import re
        # split on period/question mark/exclamation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]
