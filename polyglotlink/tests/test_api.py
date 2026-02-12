"""
Tests for the REST API v1 endpoints.

Uses FastAPI TestClient with a mock server that has real pipeline
components (SchemaExtractor, SemanticTranslator, NormalizationEngine)
but no protocol listeners or output brokers.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from polyglotlink.app.server import create_app
from polyglotlink.modules.normalization_engine import NormalizationEngine
from polyglotlink.modules.schema_extractor import SchemaCache, SchemaExtractor
from polyglotlink.modules.semantic_translator_agent import SemanticTranslator


class MockServer:
    """Lightweight stand-in for PolyglotLinkServer with real pipeline components."""

    def __init__(self):
        self._running = True
        self._schema_extractor = SchemaExtractor(cache=SchemaCache(ttl_days=30))
        self._semantic_translator = SemanticTranslator()
        self._normalization_engine = NormalizationEngine()
        self._output_broker = None
        self._metrics = {
            "messages_received": 5,
            "messages_processed": 4,
            "messages_failed": 1,
            "start_time": None,
        }

    def get_metrics(self):
        return {
            **self._metrics,
            "uptime_seconds": 120.0,
            "running": self._running,
        }


@pytest.fixture
def app():
    """Create a FastAPI app with a mock server wired in."""
    application = create_app()
    application.state.server = MockServer()
    return application


@pytest.fixture
async def client(app):
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# POST /api/v1/test
# ---------------------------------------------------------------------------


class TestPostTest:
    """Tests for the dry-run test endpoint."""

    @pytest.mark.asyncio
    async def test_simple_json_payload(self, client):
        resp = await client.post(
            "/api/v1/test",
            json={
                "payload": {"temperature": 23.5, "humidity": 60},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message_id" in data
        assert "data" in data
        assert "schema_signature" in data
        assert data["published"] is False

    @pytest.mark.asyncio
    async def test_custom_device_id(self, client):
        resp = await client.post(
            "/api/v1/test",
            json={
                "payload": {"temp": 20},
                "device_id": "sensor-42",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["device_id"] == "sensor-42"

    @pytest.mark.asyncio
    async def test_never_publishes(self, client):
        resp = await client.post(
            "/api/v1/test",
            json={"payload": {"x": 1}},
        )
        assert resp.status_code == 200
        assert resp.json()["published"] is False

    @pytest.mark.asyncio
    async def test_returns_normalized_data(self, client):
        resp = await client.post(
            "/api/v1/test",
            json={"payload": {"temperature_c": 100, "pressure_hpa": 1013}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["data"], dict)
        assert isinstance(data["confidence"], float)

    @pytest.mark.asyncio
    async def test_empty_payload_still_works(self, client):
        resp = await client.post(
            "/api/v1/test",
            json={"payload": {}},
        )
        # Empty payloads should still process (with empty fields)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------


class TestPostIngest:
    """Tests for the ingest endpoint."""

    @pytest.mark.asyncio
    async def test_ingest_default_publishes(self, client):
        resp = await client.post(
            "/api/v1/ingest",
            json={"payload": {"temp": 25}},
        )
        assert resp.status_code == 200
        # No output broker is configured, so published=False even with default
        assert resp.json()["published"] is False

    @pytest.mark.asyncio
    async def test_ingest_publish_false(self, client):
        resp = await client.post(
            "/api/v1/ingest?publish=false",
            json={"payload": {"temp": 25}},
        )
        assert resp.status_code == 200
        assert resp.json()["published"] is False

    @pytest.mark.asyncio
    async def test_invalid_protocol_returns_400(self, client):
        resp = await client.post(
            "/api/v1/ingest",
            json={
                "payload": {"temp": 25},
                "protocol": "INVALID_PROTO",
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/v1/schemas
# ---------------------------------------------------------------------------


class TestGetSchemas:
    """Tests for schema listing."""

    @pytest.mark.asyncio
    async def test_empty_initially(self, client):
        resp = await client.get("/api/v1/schemas")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_populated_after_cache_write(self, client, app):
        """Schema listing works after a schema mapping is cached."""
        from polyglotlink.models.schemas import MappingSource

        server = app.state.server
        # Manually cache a schema mapping (like the pipeline would after learning)
        server._schema_extractor.cache_mapping(
            schema_signature="test-sig-abc",
            field_mappings=[],
            confidence=0.85,
            source=MappingSource.LLM,
        )

        resp = await client.get("/api/v1/schemas")
        assert resp.status_code == 200
        schemas = resp.json()
        assert len(schemas) >= 1

        item = schemas[0]
        assert "schema_signature" in item
        assert "field_count" in item
        assert "confidence" in item
        assert "source" in item
        assert "created_at" in item


# ---------------------------------------------------------------------------
# GET /api/v1/schemas/{signature}
# ---------------------------------------------------------------------------


class TestGetSchemaBySignature:
    """Tests for fetching a specific schema."""

    @pytest.mark.asyncio
    async def test_unknown_signature_returns_404(self, client):
        resp = await client.get("/api/v1/schemas/nonexistent123")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_known_signature_returns_mapping(self, client, app):
        """Fetching a cached schema by signature returns the full mapping."""
        from polyglotlink.models.schemas import MappingSource

        server = app.state.server
        sig = "known-sig-xyz"
        server._schema_extractor.cache_mapping(
            schema_signature=sig,
            field_mappings=[],
            confidence=0.92,
            source=MappingSource.LLM,
        )

        resp = await client.get(f"/api/v1/schemas/{sig}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_signature"] == sig
        assert "field_mappings" in data
        assert data["confidence"] == 0.92


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


class TestHealthDetail:
    """Tests for the detailed health endpoint."""

    @pytest.mark.asyncio
    async def test_returns_detailed_health(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["running"] is True
        assert data["messages_received"] == 5
        assert data["messages_processed"] == 4
        assert data["messages_failed"] == 1
        assert isinstance(data["cache_schemas"], int)

    @pytest.mark.asyncio
    async def test_health_without_server(self):
        """When server is None, health should return 'starting'."""
        app = create_app()
        # Don't set app.state.server — leave it as None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "starting"
            assert resp.json()["running"] is False


# ---------------------------------------------------------------------------
# Server not initialized (503)
# ---------------------------------------------------------------------------


class TestServerNotInitialized:
    """Endpoints should return 503 when server is not set."""

    @pytest.mark.asyncio
    async def test_ingest_503(self):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/ingest", json={"payload": {"x": 1}})
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_test_503(self):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/test", json={"payload": {"x": 1}})
            assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_schemas_503(self):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/schemas")
            assert resp.status_code == 503
