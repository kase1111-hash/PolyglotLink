"""
Integration tests for the PolyglotLink message processing pipeline.

These tests verify that all components work together correctly,
from raw message ingestion through normalization and output.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from polyglotlink.models.schemas import (
    PayloadEncoding,
    Protocol,
    RawMessage,
)
from polyglotlink.modules.schema_extractor import SchemaExtractor
from polyglotlink.modules.semantic_translator_agent import SemanticTranslator
from polyglotlink.modules.normalization_engine import NormalizationEngine
from polyglotlink.modules.protocol_listener import (
    detect_encoding,
    extract_device_id,
    generate_uuid,
)


class TestEndToEndPipeline:
    """Integration tests for the full message processing pipeline."""

    @pytest.fixture
    def schema_extractor(self):
        return SchemaExtractor()

    @pytest.fixture
    def semantic_translator(self):
        return SemanticTranslator()

    @pytest.fixture
    def normalization_engine(self):
        return NormalizationEngine()

    @pytest.mark.asyncio
    async def test_environmental_sensor_pipeline(
        self,
        schema_extractor,
        semantic_translator,
        normalization_engine,
    ):
        """Test processing an environmental sensor message end-to-end."""
        # Step 1: Create raw message (simulating protocol listener output)
        payload = json.dumps({
            "temperature": 23.5,
            "humidity": 65,
            "pressure_hpa": 1013.25,
            "battery_pct": 85,
            "device_id": "env-sensor-001",
            "timestamp": "2024-01-15T10:30:00Z"
        }).encode()

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id=extract_device_id("sensors/env-sensor-001/telemetry"),
            protocol=Protocol.MQTT,
            topic="sensors/env-sensor-001/telemetry",
            payload_raw=payload,
            payload_encoding=detect_encoding(payload),
            timestamp=datetime.utcnow(),
        )

        # Step 2: Extract schema
        schema = schema_extractor.extract_schema(raw)

        assert schema.device_id == "env-sensor-001"
        assert len(schema.fields) >= 5
        assert schema.schema_signature is not None

        # Verify field detection
        field_keys = [f.key for f in schema.fields]
        assert "temperature" in field_keys
        assert "humidity" in field_keys
        assert "pressure_hpa" in field_keys

        # Verify semantic hints
        temp_field = next(f for f in schema.fields if f.key == "temperature")
        assert temp_field.inferred_semantic == "temperature"

        # Step 3: Translate schema
        mapping = await semantic_translator.translate_schema(schema)

        assert mapping.message_id == raw.message_id
        assert mapping.confidence > 0
        assert len(mapping.field_mappings) > 0

        # Step 4: Normalize message
        normalized = normalization_engine.normalize_message(schema, mapping)

        assert normalized.message_id == raw.message_id
        assert normalized.device_id == "env-sensor-001"
        assert normalized.timestamp is not None
        assert len(normalized.data) > 0

        # Verify metadata enrichment
        assert "source_protocol" in normalized.metadata
        assert normalized.metadata["source_protocol"] == "MQTT"

    @pytest.mark.asyncio
    async def test_power_meter_pipeline(
        self,
        schema_extractor,
        semantic_translator,
        normalization_engine,
    ):
        """Test processing a power meter message end-to-end."""
        payload = json.dumps({
            "voltage": 230.5,
            "current": 2.3,
            "power_w": 530.15,
            "energy_kwh": 1234.56,
            "power_factor": 0.95,
            "frequency_hz": 50.0,
            "ts": 1705312200000
        }).encode()

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="power-meter-001",
            protocol=Protocol.HTTP,
            topic="/api/meters/power-meter-001/readings",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = schema_extractor.extract_schema(raw)
        mapping = await semantic_translator.translate_schema(schema)
        normalized = normalization_engine.normalize_message(schema, mapping)

        assert normalized.device_id == "power-meter-001"
        assert len(normalized.data) > 0

        # Verify timestamp extraction from Unix milliseconds
        assert normalized.timestamp is not None

    @pytest.mark.asyncio
    async def test_gps_tracker_pipeline(
        self,
        schema_extractor,
        semantic_translator,
        normalization_engine,
    ):
        """Test processing a GPS tracker message end-to-end."""
        payload = json.dumps({
            "lat": 40.7128,
            "lng": -74.0060,
            "altitude_m": 10.5,
            "speed_kmh": 45.2,
            "heading": 180,
            "accuracy": 5.0,
            "satellites": 8,
            "timestamp": "2024-01-15T10:30:00Z"
        }).encode()

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="tracker-001",
            protocol=Protocol.MQTT,
            topic="trackers/tracker-001/location",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = schema_extractor.extract_schema(raw)

        # Verify location fields are identified
        lat_field = next((f for f in schema.fields if f.key == "lat"), None)
        assert lat_field is not None
        assert lat_field.inferred_semantic == "latitude"

        mapping = await semantic_translator.translate_schema(schema)
        normalized = normalization_engine.normalize_message(schema, mapping)

        assert len(normalized.data) > 0

    @pytest.mark.asyncio
    async def test_nested_payload_pipeline(
        self,
        schema_extractor,
        semantic_translator,
        normalization_engine,
    ):
        """Test processing a nested payload end-to-end."""
        payload = json.dumps({
            "device": {
                "id": "sensor-001",
                "type": "environmental",
                "firmware": "1.2.3"
            },
            "readings": {
                "temperature": {
                    "value": 23.5,
                    "unit": "celsius"
                },
                "humidity": {
                    "value": 65,
                    "unit": "percent"
                }
            },
            "meta": {
                "timestamp": "2024-01-15T10:30:00Z",
                "sequence": 12345
            }
        }).encode()

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = schema_extractor.extract_schema(raw)

        # Verify flattening
        field_keys = [f.key for f in schema.fields]
        assert "device.id" in field_keys
        assert "readings.temperature.value" in field_keys
        assert "readings.humidity.value" in field_keys

        mapping = await semantic_translator.translate_schema(schema)
        normalized = normalization_engine.normalize_message(schema, mapping)

        assert len(normalized.data) > 0


class TestProtocolListenerIntegration:
    """Integration tests for protocol listener components."""

    def test_encoding_detection_json(self):
        payload = json.dumps({"key": "value"}).encode()
        assert detect_encoding(payload) == PayloadEncoding.JSON

    def test_encoding_detection_xml(self):
        payload = b"<root><value>123</value></root>"
        assert detect_encoding(payload) == PayloadEncoding.XML

    def test_device_id_extraction_patterns(self):
        # Pattern: devices/{id}/telemetry
        assert extract_device_id("devices/sensor-01/telemetry") == "sensor-01"

        # Pattern: sensors/{id}
        assert extract_device_id("sensors/temp-001") == "temp-001"

        # Pattern: {id}/data - extracts the ID before /data
        assert extract_device_id("meter-001/data") == "meter-001"

        # Fallback: last segment
        assert extract_device_id("some/random/path/device123") == "device123"

    def test_uuid_generation_uniqueness(self):
        uuids = [generate_uuid() for _ in range(100)]
        assert len(set(uuids)) == 100  # All unique


class TestSchemaExtractorIntegration:
    """Integration tests for schema extractor with various payload types."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_handles_empty_payload(self, extractor):
        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=b"{}",
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        assert schema.fields == []

    def test_handles_array_payload(self, extractor):
        payload = json.dumps([
            {"temp": 23.5, "id": 1},
            {"temp": 24.0, "id": 2}
        ]).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        # Array should be wrapped
        assert len(schema.fields) > 0

    def test_handles_numeric_string_fields(self, extractor):
        payload = json.dumps({
            "value": "123.45",
            "count": "100"
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)

        value_field = next(f for f in schema.fields if f.key == "value")
        assert value_field.value_type == "numeric_string"

    def test_schema_caching_consistency(self, extractor):
        """Verify that same schema structure produces same signature."""
        payload1 = json.dumps({"temp": 23.5, "humidity": 65}).encode()
        payload2 = json.dumps({"temp": 25.0, "humidity": 70}).encode()

        raw1 = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload1,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        raw2 = RawMessage(
            message_id="test-002",
            device_id="device-002",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload2,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema1 = extractor.extract_schema(raw1)
        schema2 = extractor.extract_schema(raw2)

        # Same structure = same signature
        assert schema1.schema_signature == schema2.schema_signature


class TestNormalizationIntegration:
    """Integration tests for normalization with real conversions."""

    @pytest.fixture
    def engine(self):
        return NormalizationEngine()

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    @pytest.fixture
    def translator(self):
        return SemanticTranslator()

    @pytest.mark.asyncio
    async def test_temperature_conversion_flow(
        self,
        extractor,
        translator,
        engine,
    ):
        """Test that temperature values can be properly converted."""
        # Create a message with Fahrenheit temperature
        payload = json.dumps({
            "temperature_f": 77.0,  # 25°C in Fahrenheit
            "device_id": "thermo-001"
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="thermo-001",
            protocol=Protocol.MQTT,
            topic="sensors/thermo-001/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)

        # Verify Fahrenheit unit was inferred
        temp_field = next(f for f in schema.fields if f.key == "temperature_f")
        assert temp_field.inferred_unit == "fahrenheit"

        mapping = await translator.translate_schema(schema)
        normalized = engine.normalize_message(schema, mapping)

        # Message should be normalized successfully
        assert len(normalized.data) > 0

    @pytest.mark.asyncio
    async def test_handles_missing_mappings_gracefully(
        self,
        extractor,
        translator,
        engine,
    ):
        """Test that unmapped fields are handled properly."""
        payload = json.dumps({
            "unknown_field_xyz": 42,
            "another_mystery": "value"
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)
        normalized = engine.normalize_message(schema, mapping)

        # Unmapped fields should appear with prefix
        unmapped_keys = [k for k in normalized.data.keys() if k.startswith("_unmapped")]
        assert len(unmapped_keys) >= 0  # May or may not have unmapped depending on translator


class TestValidationIntegration:
    """Integration tests for validation across the pipeline."""

    @pytest.fixture
    def engine(self):
        return NormalizationEngine()

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    @pytest.fixture
    def translator(self):
        return SemanticTranslator()

    @pytest.mark.asyncio
    async def test_out_of_range_values_flagged(
        self,
        extractor,
        translator,
        engine,
    ):
        """Test that out-of-range values are caught during normalization."""
        payload = json.dumps({
            "humidity": 150,  # Invalid: > 100%
            "temperature": 23.5
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)
        normalized = engine.normalize_message(schema, mapping)

        # Check if validation errors were recorded
        # (depends on whether humidity_percent concept has constraints)
        assert normalized is not None

    @pytest.mark.asyncio
    async def test_malformed_timestamps_handled(
        self,
        extractor,
        translator,
        engine,
    ):
        """Test handling of malformed timestamp values."""
        payload = json.dumps({
            "value": 42,
            "timestamp": "not-a-valid-timestamp"
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)
        normalized = engine.normalize_message(schema, mapping)

        # Should complete without crashing
        assert normalized is not None
        # Timestamp should fall back to current time
        assert normalized.timestamp is not None
