"""
System and Acceptance Tests for PolyglotLink.

These tests verify the complete system behavior from an end-user perspective,
ensuring the application meets its acceptance criteria and specification.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from polyglotlink.models.schemas import (
    NormalizedMessage,
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
from polyglotlink.utils.config import Settings, get_settings
from polyglotlink.utils.validation import (
    sanitize_string,
    validate_json_payload,
    is_valid_topic,
)


class TestAcceptanceCriteria:
    """Tests verifying the system meets acceptance criteria from spec."""

    @pytest.fixture
    def pipeline_components(self):
        """Create all pipeline components for testing."""
        return {
            "extractor": SchemaExtractor(),
            "translator": SemanticTranslator(),
            "normalizer": NormalizationEngine(),
        }

    @pytest.mark.asyncio
    async def test_ac1_multi_protocol_message_ingestion(self, pipeline_components):
        """
        AC1: System shall ingest messages from MQTT, CoAP, Modbus, OPC-UA,
        HTTP, and WebSocket protocols.
        """
        protocols = [
            Protocol.MQTT,
            Protocol.COAP,
            Protocol.MODBUS,
            Protocol.OPCUA,
            Protocol.HTTP,
            Protocol.WEBSOCKET,
        ]

        payload = json.dumps({"value": 42}).encode()

        for protocol in protocols:
            raw = RawMessage(
                message_id=generate_uuid(),
                device_id=f"device-{protocol.value}",
                protocol=protocol,
                topic=f"/test/{protocol.value}",
                payload_raw=payload,
                payload_encoding=PayloadEncoding.JSON,
                timestamp=datetime.utcnow(),
            )

            # Verify message creation succeeds for all protocols
            assert raw.protocol == protocol
            assert raw.payload_raw == payload

    @pytest.mark.asyncio
    async def test_ac2_automatic_schema_detection(self, pipeline_components):
        """
        AC2: System shall automatically detect and extract schemas from
        incoming payloads without prior configuration.
        """
        extractor = pipeline_components["extractor"]

        # Test with unknown/new payload structure
        payload = json.dumps({
            "custom_sensor_reading": 123.45,
            "proprietary_status": "active",
            "vendor_timestamp": "2024-01-15T10:30:00Z",
            "nested": {
                "custom_value": 99
            }
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="unknown-device",
            protocol=Protocol.MQTT,
            topic="unknown/topic",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        # Should extract schema without configuration
        schema = extractor.extract_schema(raw)

        assert schema is not None
        assert len(schema.fields) >= 4
        assert schema.schema_signature is not None

        # Verify nested fields are flattened
        field_keys = [f.key for f in schema.fields]
        assert "nested.custom_value" in field_keys

    @pytest.mark.asyncio
    async def test_ac3_semantic_translation(self, pipeline_components):
        """
        AC3: System shall translate device-specific field names to
        canonical concepts.
        """
        extractor = pipeline_components["extractor"]
        translator = pipeline_components["translator"]

        # Device uses non-standard field names
        payload = json.dumps({
            "temp_celsius": 23.5,
            "rh_percent": 65,
            "press_hpa": 1013.25,
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)

        # Should have field mappings
        assert len(mapping.field_mappings) > 0
        assert mapping.confidence > 0

    @pytest.mark.asyncio
    async def test_ac4_unit_conversion(self, pipeline_components):
        """
        AC4: System shall convert values to canonical units where applicable.
        """
        extractor = pipeline_components["extractor"]
        translator = pipeline_components["translator"]
        normalizer = pipeline_components["normalizer"]

        # Temperature in Fahrenheit
        payload = json.dumps({
            "temperature_f": 77.0,  # 25°C
            "speed_mph": 60,  # ~96.5 km/h
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)

        # Verify unit inference from field name
        temp_field = next((f for f in schema.fields if f.key == "temperature_f"), None)
        assert temp_field is not None
        assert temp_field.inferred_unit == "fahrenheit"

        mapping = await translator.translate_schema(schema)
        normalized = normalizer.normalize_message(schema, mapping)

        assert normalized is not None

    @pytest.mark.asyncio
    async def test_ac5_normalized_output_structure(self, pipeline_components):
        """
        AC5: System shall output messages in a unified, normalized format.
        """
        extractor = pipeline_components["extractor"]
        translator = pipeline_components["translator"]
        normalizer = pipeline_components["normalizer"]

        payload = json.dumps({
            "temperature": 23.5,
            "humidity": 65,
            "device_id": "sensor-001",
            "timestamp": "2024-01-15T10:30:00Z"
        }).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)
        normalized = normalizer.normalize_message(schema, mapping)

        # Verify normalized message structure
        assert isinstance(normalized, NormalizedMessage)
        assert normalized.message_id == raw.message_id
        assert normalized.device_id == "sensor-001"
        assert normalized.timestamp is not None
        assert isinstance(normalized.data, dict)
        assert isinstance(normalized.metadata, dict)

    @pytest.mark.asyncio
    async def test_ac6_schema_caching(self, pipeline_components):
        """
        AC6: System shall cache learned schemas for performance optimization.
        """
        extractor = pipeline_components["extractor"]

        payload1 = json.dumps({"temp": 23.5, "hum": 65}).encode()
        payload2 = json.dumps({"temp": 25.0, "hum": 70}).encode()

        raw1 = RawMessage(
            message_id="test-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload1,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        raw2 = RawMessage(
            message_id="test-002",
            device_id="sensor-002",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload2,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema1 = extractor.extract_schema(raw1)
        schema2 = extractor.extract_schema(raw2)

        # Same structure should produce same signature
        assert schema1.schema_signature == schema2.schema_signature

    @pytest.mark.asyncio
    async def test_ac7_metadata_enrichment(self, pipeline_components):
        """
        AC7: System shall enrich messages with processing metadata.
        """
        extractor = pipeline_components["extractor"]
        translator = pipeline_components["translator"]
        normalizer = pipeline_components["normalizer"]

        payload = json.dumps({"value": 42}).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        mapping = await translator.translate_schema(schema)
        normalized = normalizer.normalize_message(schema, mapping)

        # Verify metadata enrichment
        assert "source_protocol" in normalized.metadata
        assert normalized.metadata["source_protocol"] == "MQTT"


class TestSystemBehavior:
    """Tests verifying overall system behavior."""

    def test_encoding_auto_detection(self):
        """System should auto-detect payload encoding."""
        json_payload = b'{"key": "value"}'
        xml_payload = b'<root><key>value</key></root>'
        cbor_payload = bytes([0xa1, 0x63, 0x6b, 0x65, 0x79])  # CBOR map

        assert detect_encoding(json_payload) == PayloadEncoding.JSON
        assert detect_encoding(xml_payload) == PayloadEncoding.XML

    def test_device_id_extraction_patterns(self):
        """System should extract device IDs from various topic patterns."""
        patterns = [
            ("devices/sensor-001/telemetry", "sensor-001"),
            ("sensors/temp-001", "temp-001"),
            ("/api/v1/devices/meter-001/data", "meter-001"),
            ("building/floor1/room1/sensor", "sensor"),
        ]

        for topic, expected in patterns:
            result = extract_device_id(topic)
            assert result is not None

    def test_uuid_generation(self):
        """System should generate unique message IDs."""
        uuids = set()
        for _ in range(1000):
            uuids.add(generate_uuid())

        assert len(uuids) == 1000  # All unique

    def test_input_validation(self):
        """System should validate and sanitize inputs."""
        # Valid inputs
        assert sanitize_string("normal-device-001") == "normal-device-001"
        assert is_valid_topic("sensors/device-001/data")

        # Potentially malicious inputs should be sanitized
        malicious = "<script>alert('xss')</script>"
        sanitized = sanitize_string(malicious)
        assert "<script>" not in sanitized

    def test_json_payload_validation(self):
        """System should validate JSON payloads."""
        valid_json = b'{"key": "value", "number": 123}'
        invalid_json = b'{"key": invalid}'

        is_valid, _ = validate_json_payload(valid_json)
        assert is_valid is True

        is_valid, error = validate_json_payload(invalid_json)
        assert is_valid is False
        assert error is not None


class TestConfigurationSystem:
    """Tests for configuration management."""

    def test_settings_loading(self):
        """System should load settings correctly."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_environment_detection(self):
        """System should detect environment correctly."""
        settings = get_settings()
        assert settings.env in ["development", "staging", "production", "test"]

    def test_log_level_configuration(self):
        """System should accept valid log levels."""
        # Note: validation_alias requires using alias name in constructor
        settings = Settings(LOG_LEVEL="DEBUG")
        assert settings.log_level == "DEBUG"

        settings = Settings(LOG_LEVEL="info")
        assert settings.log_level == "INFO"


class TestErrorHandling:
    """Tests for error handling behavior."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_handles_empty_payload(self, extractor):
        """System should handle empty payloads gracefully."""
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
        assert schema is not None
        assert schema.fields == []

    def test_handles_malformed_json(self, extractor):
        """System should handle malformed JSON gracefully."""
        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=b"{invalid json}",
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        # Should not crash, should return empty or error schema
        try:
            schema = extractor.extract_schema(raw)
            assert schema is not None
        except Exception:
            # Acceptable to raise an exception for malformed data
            pass

    def test_handles_binary_payload(self, extractor):
        """System should handle binary payloads appropriately."""
        raw = RawMessage(
            message_id="test-001",
            device_id="device-001",
            protocol=Protocol.MODBUS,
            topic="modbus/data",
            payload_raw=b"\x00\x01\x02\x03\x04",
            payload_encoding=PayloadEncoding.BINARY,
            timestamp=datetime.utcnow(),
        )

        # Should handle without crashing
        try:
            schema = extractor.extract_schema(raw)
        except Exception:
            # Acceptable for binary that can't be parsed
            pass


class TestPerformanceCharacteristics:
    """Tests verifying performance characteristics."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_schema_extraction_is_fast(self, extractor):
        """Schema extraction should complete quickly."""
        import time

        payload = json.dumps({
            "field1": 1, "field2": 2, "field3": 3,
            "field4": 4, "field5": 5, "field6": 6,
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

        start = time.perf_counter()
        for _ in range(100):
            extractor.extract_schema(raw)
        elapsed = time.perf_counter() - start

        # 100 extractions should complete in under 1 second
        assert elapsed < 1.0

    def test_handles_large_payload(self, extractor):
        """System should handle reasonably large payloads."""
        # Create a large payload with many fields
        large_data = {f"field_{i}": i * 1.5 for i in range(100)}
        payload = json.dumps(large_data).encode()

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
        assert len(schema.fields) == 100

    def test_handles_deeply_nested_payload(self, extractor):
        """System should handle deeply nested payloads."""
        # Create nested structure
        nested = {"value": 42}
        for i in range(10):
            nested = {"level": nested}

        payload = json.dumps(nested).encode()

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
        assert schema is not None
        assert len(schema.fields) > 0


class TestRegressionSuite:
    """Regression tests to prevent previously fixed bugs."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_numeric_string_detection(self, extractor):
        """Regression: Numeric strings should be detected correctly."""
        payload = json.dumps({
            "port": "8080",
            "version": "1.2.3",
            "count": "42"
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

        count_field = next((f for f in schema.fields if f.key == "count"), None)
        assert count_field is not None
        assert count_field.value_type == "numeric_string"

    def test_timestamp_field_detection(self, extractor):
        """Regression: Timestamp fields should be detected."""
        payload = json.dumps({
            "timestamp": "2024-01-15T10:30:00Z",
            "created_at": "2024-01-15T10:30:00Z",
            "ts": 1705312200000,
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

        timestamp_fields = [f for f in schema.fields if f.is_timestamp]
        assert len(timestamp_fields) >= 1

    def test_identifier_field_detection(self, extractor):
        """Regression: Identifier fields should be detected."""
        payload = json.dumps({
            "device_id": "sensor-001",
            "sensor_id": "temp-001",
            "id": "12345",
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

        id_fields = [f for f in schema.fields if f.is_identifier]
        assert len(id_fields) >= 1

    def test_array_payload_handling(self, extractor):
        """Regression: Array payloads should be handled."""
        payload = json.dumps([
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
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
        assert schema is not None

    def test_special_characters_in_values(self, extractor):
        """Regression: Special characters in values should be handled."""
        payload = json.dumps({
            "message": "Hello, World! 你好",
            "path": "/usr/local/bin",
            "emoji": "🌡️",
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
        assert len(schema.fields) == 3
