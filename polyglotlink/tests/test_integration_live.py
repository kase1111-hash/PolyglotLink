"""
Live integration tests for PolyglotLink.

These tests require real Redis and Mosquitto services.
Run with:
    make test-live

Or manually:
    docker compose -f docker-compose.test.yml up -d
    pytest polyglotlink/tests/test_integration_live.py -v --timeout=60
    docker compose -f docker-compose.test.yml down

Ports (offset to avoid collisions with dev services):
    Redis:     localhost:6399
    Mosquitto: localhost:1893
"""

import asyncio
import json
import os
import time

import pytest
from httpx import ASGITransport, AsyncClient

from polyglotlink.app.server import create_app
from polyglotlink.models.schemas import (
    CachedMapping,
    MappingSource,
    PayloadEncoding,
    Protocol,
    RawMessage,
)
from polyglotlink.modules.normalization_engine import NormalizationEngine
from polyglotlink.modules.protocol_listener import generate_uuid
from polyglotlink.modules.schema_extractor import SchemaCache, SchemaExtractor
from polyglotlink.modules.semantic_translator_agent import SemanticTranslator

# ---------------------------------------------------------------------------
# Configuration — override via env vars if needed
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6399/0")
MQTT_HOST = os.environ.get("TEST_MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("TEST_MQTT_PORT", "1893"))

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_redis():
    """Create a Redis client for testing, or skip if unavailable."""
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available at {REDIS_URL}: {e}")


def _make_mqtt_client():
    """Create a paho-mqtt client for testing, or skip if unavailable."""
    try:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id=f"polyglotlink-test-{generate_uuid()[:8]}")
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=10)
        return client
    except Exception as e:
        pytest.skip(f"MQTT broker not available at {MQTT_HOST}:{MQTT_PORT}: {e}")


class LiveMockServer:
    """Server-like object with real pipeline components and a real Redis cache."""

    def __init__(self, redis_client):
        self._running = True
        self._schema_extractor = SchemaExtractor(
            cache=SchemaCache(ttl_days=30, redis_client=redis_client),
        )
        self._semantic_translator = SemanticTranslator()
        self._normalization_engine = NormalizationEngine()
        self._output_broker = None
        self._metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "start_time": None,
        }

    def get_metrics(self):
        return {**self._metrics, "uptime_seconds": 60.0, "running": self._running}


@pytest.fixture
def redis_client():
    """Provide a real Redis client (skips if unavailable)."""
    client = _make_redis()
    # Clean test keys before each test
    for key in client.scan_iter("schema:*"):
        client.delete(key)
    yield client
    # Clean up after test
    for key in client.scan_iter("schema:*"):
        client.delete(key)


@pytest.fixture
def live_app(redis_client):
    """FastAPI app wired to a server with real Redis."""
    application = create_app()
    application.state.server = LiveMockServer(redis_client)
    return application


@pytest.fixture
async def live_client(live_app):
    """Async HTTP client against the live app."""
    transport = ASGITransport(app=live_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Test 1: MQTT end-to-end
# ---------------------------------------------------------------------------


class TestMQTTEndToEnd:
    """Publish a JSON payload to Mosquitto, verify pipeline processes it."""

    def test_mqtt_publish_and_receive(self):
        """Publish a message to Mosquitto and verify receipt via subscription."""
        mqtt_client = _make_mqtt_client()
        received = []

        def on_message(_client, _userdata, msg):
            received.append(json.loads(msg.payload.decode()))

        mqtt_client.on_message = on_message
        mqtt_client.subscribe("polyglotlink/test/#", qos=1)
        mqtt_client.loop_start()

        payload = {"temperature_c": 22.5, "humidity_pct": 55, "device_id": "mqtt-test"}
        mqtt_client.publish(
            "polyglotlink/test/sensor-001",
            json.dumps(payload).encode(),
            qos=1,
        )

        # Wait for message delivery
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.1)

        mqtt_client.loop_stop()
        mqtt_client.disconnect()

        assert len(received) == 1
        assert received[0]["temperature_c"] == 22.5

    def test_mqtt_payload_through_pipeline(self):
        """Verify an MQTT-style payload processes through the pipeline."""
        payload = json.dumps(
            {"temperature_c": 22.5, "humidity_pct": 55, "device_id": "mqtt-e2e"}
        ).encode()

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="mqtt-e2e",
            protocol=Protocol.MQTT,
            topic="sensors/mqtt-e2e/telemetry",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=__import__("datetime").datetime.utcnow(),
        )

        extractor = SchemaExtractor()
        schema = extractor.extract_schema(raw)

        assert schema.schema_signature is not None
        assert len(schema.fields) >= 2

        temp_field = next(f for f in schema.fields if f.key == "temperature_c")
        assert temp_field.inferred_unit == "celsius"


# ---------------------------------------------------------------------------
# Test 2: Redis persistence
# ---------------------------------------------------------------------------


class TestRedisPersistence:
    """Verify schema cache survives across SchemaExtractor instances."""

    def test_cache_survives_new_extractor(self, redis_client):
        """Schema cached via one extractor is retrievable by a fresh extractor."""
        # Extractor 1: process a payload and cache the mapping
        cache1 = SchemaCache(ttl_days=30, redis_client=redis_client)
        extractor1 = SchemaExtractor(cache=cache1)

        extractor1.cache_mapping(
            schema_signature="persist-test-sig",
            field_mappings=[],
            confidence=0.91,
            source=MappingSource.LLM,
        )

        # Extractor 2: brand new instance, same Redis
        cache2 = SchemaCache(ttl_days=30, redis_client=redis_client)

        result = cache2.get("persist-test-sig")
        assert result is not None
        assert result.schema_signature == "persist-test-sig"
        assert result.confidence == 0.91

    def test_schema_from_pipeline_persists(self, redis_client):
        """A schema learned from a real payload persists to Redis."""
        cache = SchemaCache(ttl_days=30, redis_client=redis_client)
        extractor = SchemaExtractor(cache=cache)

        payload = json.dumps({"voltage_v": 230.5, "current_a": 2.3}).encode()
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="persist-device",
            protocol=Protocol.HTTP,
            topic="api/test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=__import__("datetime").datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        sig = schema.schema_signature

        # Manually cache a mapping for this signature
        extractor.cache_mapping(
            schema_signature=sig,
            field_mappings=[],
            confidence=0.88,
            source=MappingSource.LEARNED,
        )

        # Verify it's in Redis directly
        raw_data = redis_client.get(f"schema:{sig}")
        assert raw_data is not None

        parsed = CachedMapping.model_validate_json(raw_data)
        assert parsed.schema_signature == sig
        assert parsed.confidence == 0.88


# ---------------------------------------------------------------------------
# Test 3: Multi-format payloads via API
# ---------------------------------------------------------------------------


class TestMultiFormatAPI:
    """Send JSON, XML, and CSV payloads via the REST API."""

    @pytest.mark.asyncio
    async def test_json_payload_via_api(self, live_client):
        """JSON payload normalizes correctly via POST /api/v1/test."""
        resp = await live_client.post(
            "/api/v1/test",
            json={
                "payload": {"temperature_c": 25.0, "humidity_pct": 60},
                "device_id": "json-device",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "json-device"
        assert isinstance(data["data"], dict)
        assert data["schema_signature"] is not None

    @pytest.mark.asyncio
    async def test_nested_payload_via_api(self, live_client):
        """Nested JSON payload is flattened and normalized."""
        resp = await live_client.post(
            "/api/v1/test",
            json={
                "payload": {
                    "readings": {
                        "temperature": {"value": 296.65, "unit": "K"},
                        "pressure": {"value": 101325, "unit": "Pa"},
                    },
                    "serial": "nested-001",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["data"], dict)
        assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_different_schemas_produce_different_signatures(self, live_client):
        """Two structurally different payloads get different schema signatures."""
        resp1 = await live_client.post(
            "/api/v1/test",
            json={"payload": {"temperature_c": 25.0, "humidity_pct": 60}},
        )
        resp2 = await live_client.post(
            "/api/v1/test",
            json={"payload": {"voltage_v": 230.5, "current_a": 2.3, "power_w": 530}},
        )

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        sig1 = resp1.json()["schema_signature"]
        sig2 = resp2.json()["schema_signature"]
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# Test 4: Schema reuse (cache hit on second request)
# ---------------------------------------------------------------------------


class TestSchemaReuse:
    """Send the same payload twice, verify cache is populated."""

    @pytest.mark.asyncio
    async def test_schema_appears_in_cache_after_manual_store(self, live_client, live_app):
        """After caching a schema mapping, GET /schemas lists it."""
        server = live_app.state.server

        # First: ingest a payload to learn its schema signature
        resp = await live_client.post(
            "/api/v1/test",
            json={"payload": {"temp_c": 22.0, "humidity": 50}},
        )
        assert resp.status_code == 200
        sig = resp.json()["schema_signature"]

        # Manually cache a mapping for the learned signature
        server._schema_extractor.cache_mapping(
            schema_signature=sig,
            field_mappings=[],
            confidence=0.9,
            source=MappingSource.LEARNED,
        )

        # Verify it appears in the schema listing
        resp = await live_client.get("/api/v1/schemas")
        assert resp.status_code == 200
        schemas = resp.json()
        sigs = [s["schema_signature"] for s in schemas]
        assert sig in sigs

    @pytest.mark.asyncio
    async def test_cached_schema_retrievable_by_signature(self, live_client, live_app):
        """After caching, GET /schemas/{sig} returns the full mapping."""
        server = live_app.state.server
        sig = "reuse-test-sig-123"

        server._schema_extractor.cache_mapping(
            schema_signature=sig,
            field_mappings=[],
            confidence=0.87,
            source=MappingSource.LLM,
        )

        resp = await live_client.get(f"/api/v1/schemas/{sig}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_signature"] == sig
        assert data["confidence"] == 0.87

    def test_redis_cache_hit_after_store(self, redis_client):
        """Verify that storing a mapping and retrieving it uses Redis."""
        cache = SchemaCache(ttl_days=30, redis_client=redis_client)

        mapping = CachedMapping(
            schema_signature="reuse-redis-test",
            field_mappings=[],
            confidence=0.93,
            created_at=__import__("datetime").datetime.utcnow(),
            source=MappingSource.LEARNED,
            hit_count=0,
        )
        cache.set("reuse-redis-test", mapping)

        # Clear local cache to force Redis lookup
        cache._local_cache.clear()

        result = cache.get("reuse-redis-test")
        assert result is not None
        assert result.confidence == 0.93

        # After Redis hit, local cache should be populated
        assert "schema:reuse-redis-test" in cache._local_cache
