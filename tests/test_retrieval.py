"""
Tests for the retrieval pipeline — knowledge base loading, embedding, and search.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from src.retrieval.knowledge_loader import KnowledgeBaseLoader, Document


class TestKnowledgeBaseLoader:
    """Test document loading and chunking."""

    def _create_temp_kb(self, data):
        """Helper to create a temporary knowledge base directory."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test_kb.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        return tmpdir

    def test_load_simple_documents(self):
        data = [
            {
                "title": "Test FAQ",
                "content": "This is a test answer about password resets.",
                "category": "account",
            }
        ]
        tmpdir = self._create_temp_kb(data)

        loader = KnowledgeBaseLoader(kb_directory=tmpdir)
        docs = loader.load_all()

        assert len(docs) == 1
        assert docs[0].title == "Test FAQ"
        assert docs[0].category == "account"
        assert "password" in docs[0].content

    def test_chunking_long_document(self):
        # create a document that's longer than the default chunk size
        long_content = ". ".join([f"Sentence number {i} with some padding text" for i in range(100)])
        data = [{"title": "Long Doc", "content": long_content, "category": "test"}]
        tmpdir = self._create_temp_kb(data)

        loader = KnowledgeBaseLoader(kb_directory=tmpdir, chunk_size=200, chunk_overlap=30)
        docs = loader.load_all()

        assert len(docs) > 1  # should be split into multiple chunks
        # each chunk should be roughly under the limit
        for doc in docs:
            assert len(doc.content) < 400  # some slack for sentence boundaries

    def test_empty_content_skipped(self):
        data = [
            {"title": "Empty", "content": "", "category": "test"},
            {"title": "Valid", "content": "This one has content.", "category": "test"},
        ]
        tmpdir = self._create_temp_kb(data)

        loader = KnowledgeBaseLoader(kb_directory=tmpdir)
        docs = loader.load_all()
        assert len(docs) == 1  # the empty one should be skipped

    def test_missing_directory(self):
        loader = KnowledgeBaseLoader(kb_directory="/nonexistent/path")
        docs = loader.load_all()
        assert docs == []

    def test_sentence_splitting(self):
        result = KnowledgeBaseLoader._split_sentences(
            "First sentence. Second one! Third one? And a fourth."
        )
        assert len(result) == 4


class TestDocument:
    def test_display_source(self):
        doc = Document(
            content="test content",
            title="My FAQ",
            source_file="company_faq.json",
        )
        assert doc.display_source == "company_faq.json > My FAQ"

    def test_display_source_no_title(self):
        doc = Document(content="test", source_file="data.json")
        assert doc.display_source == "data.json"
