"""
Tests for the REST API endpoints.

Uses FastAPI's TestClient (which is backed by httpx) for synchronous
endpoint testing. WebSocket tests use the built-in WebSocket test client.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create a test client that doesn't run the lifespan events."""
    # we skip lifespan because it tries to init real services
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        # patch the vector store to avoid loading the real index
        with patch("src.api.routes.get_vector_store") as mock_store:
            mock_store.return_value.document_count = 42
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


class TestChatEndpoint:
    def test_chat_without_agent_returns_503(self, client):
        response = client.post("/chat", json={"message": "hello"})
        # should return 503 because the agent isn't initialized in test mode
        assert response.status_code == 503

    def test_chat_with_mocked_agent(self, client):
        mock_agent = MagicMock()
        mock_agent.process_message.return_value = {
            "agent_response": "How can I help you?",
            "intent": "general",
            "confidence": 0.88,
            "should_escalate": False,
            "total_latency_ms": 150.0,
            "retrieved_docs": [],
        }

        from src.api import routes
        original_agent = routes._agent
        original_metrics = routes._metrics

        try:
            routes._agent = mock_agent
            routes._metrics = MagicMock()

            response = client.post("/chat", json={"message": "hi there"})
            assert response.status_code == 200

            data = response.json()
            assert data["response"] == "How can I help you?"
            assert data["intent"] == "general"
            assert data["escalated"] is False
        finally:
            routes._agent = original_agent
            routes._metrics = original_metrics


class TestSearchEndpoint:
    def test_search_knowledge_base(self, client):
        mock_doc = MagicMock()
        mock_doc.content = "Password reset instructions"
        mock_doc.title = "Password FAQ"
        mock_doc.display_source = "company_faq.json > Password FAQ"

        with patch("src.api.routes.get_vector_store") as mock_store:
            mock_store.return_value.search.return_value = [(mock_doc, 0.95)]
            mock_store.return_value.document_count = 10

            response = client.post("/search", json={
                "query": "how to reset password",
                "top_k": 3,
            })

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.95


class TestMetricsEndpoint:
    def test_metrics_without_collector_returns_503(self, client):
        response = client.get("/metrics")
        assert response.status_code == 503
