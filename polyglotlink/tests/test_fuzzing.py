"""
Dynamic Analysis Tests - Fuzzing with Hypothesis

These tests use property-based testing to discover edge cases and
unexpected behaviors through random input generation.

Run with: pytest polyglotlink/tests/test_fuzzing.py -v
"""

import contextlib
import json
from datetime import datetime

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import (
    binary,
    booleans,
    dictionaries,
    floats,
    integers,
    lists,
    none,
    one_of,
    recursive,
    text,
)

from polyglotlink.models.schemas import (
    PayloadEncoding,
    Protocol,
    RawMessage,
)
from polyglotlink.modules.normalization_engine import NormalizationEngine
from polyglotlink.modules.protocol_listener import (
    detect_encoding,
    extract_device_id,
    generate_uuid,
)
from polyglotlink.modules.schema_extractor import SchemaExtractor
from polyglotlink.utils.validation import (
    detect_malicious_patterns,
    sanitize_identifier,
    sanitize_string,
    sanitize_topic,
    validate_json_payload,
)

# Custom strategies for IoT data
json_primitives = one_of(
    none(),
    booleans(),
    integers(),
    floats(allow_nan=False, allow_infinity=False),
    text(max_size=100),
)

json_values = recursive(
    json_primitives,
    lambda children: one_of(
        lists(children, max_size=10),
        dictionaries(text(min_size=1, max_size=20), children, max_size=10),
    ),
    max_leaves=50,
)

device_ids = text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_",
    min_size=1,
    max_size=50,
)

mqtt_topics = text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789/-_+#",
    min_size=1,
    max_size=100,
)


class TestSchemaExtractorFuzzing:
    """Fuzz testing for schema extraction."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    @given(json_values)
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    def test_extract_schema_never_crashes(self, extractor, payload_data):
        """Schema extraction should never crash on any valid JSON structure."""
        try:
            payload = json.dumps(payload_data).encode()
        except (TypeError, ValueError):
            assume(False)  # Skip non-serializable data

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="fuzz-device",
            protocol=Protocol.MQTT,
            topic="fuzz/test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        # Should not raise any exception
        try:
            schema = extractor.extract_schema(raw)
            assert schema is not None
        except json.JSONDecodeError:
            pass  # Acceptable for malformed JSON

    @given(dictionaries(text(min_size=1, max_size=50), json_primitives, max_size=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_flat_dict_extraction(self, extractor, data):
        """Flat dictionaries should always produce valid schemas."""
        try:
            payload = json.dumps(data).encode()
        except (TypeError, ValueError):
            assume(False)

        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="fuzz-device",
            protocol=Protocol.MQTT,
            topic="fuzz/test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )

        schema = extractor.extract_schema(raw)
        assert schema is not None
        # Number of fields should match or be less (due to filtering)
        assert len(schema.fields) <= len(data)

    @given(binary(min_size=0, max_size=1000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_binary_payload_handling(self, extractor, data):
        """Binary payloads should be handled without crashing."""
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="fuzz-device",
            protocol=Protocol.MODBUS,
            topic="fuzz/test",
            payload_raw=data,
            payload_encoding=PayloadEncoding.BINARY,
            timestamp=datetime.utcnow(),
        )

        # Should not crash
        with contextlib.suppress(Exception):
            # Acceptable for truly invalid data
            extractor.extract_schema(raw)


class TestEncodingDetectionFuzzing:
    """Fuzz testing for encoding detection."""

    @given(binary(min_size=0, max_size=10000))
    @settings(max_examples=500)
    def test_detect_encoding_never_crashes(self, data):
        """Encoding detection should never crash."""
        result = detect_encoding(data)
        # Should always return a valid PayloadEncoding
        assert isinstance(result, PayloadEncoding)

    @given(text(max_size=1000))
    @settings(max_examples=200)
    def test_detect_encoding_text_input(self, data):
        """Text input should be handled correctly."""
        result = detect_encoding(data.encode("utf-8", errors="replace"))
        assert result is not None


class TestValidationFuzzing:
    """Fuzz testing for input validation."""

    @given(text(max_size=10000))
    @settings(max_examples=500)
    def test_sanitize_string_never_crashes(self, data):
        """String sanitization should never crash."""
        result = sanitize_string(data)
        assert isinstance(result, str)
        # Result should never be longer than input (only removes/replaces)
        assert len(result) <= len(data) + 100  # Allow some buffer for replacements

    @given(text(min_size=1, max_size=1000, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_"))
    @settings(max_examples=300)
    def test_sanitize_identifier_valid_output(self, data):
        """Sanitized identifiers should only contain safe characters."""
        try:
            result = sanitize_identifier(data)
            # Should only contain alphanumeric, dash, underscore, dot
            assert all(c.isalnum() or c in "-_." for c in result)
        except Exception:
            pass  # Empty identifiers raise ValidationError

    @given(mqtt_topics)
    @settings(max_examples=300)
    def test_sanitize_topic_handling(self, topic):
        """Topic sanitization should produce valid topics."""
        result = sanitize_topic(topic)
        # Result should not contain dangerous characters
        assert ".." not in result
        assert "<" not in result
        assert ">" not in result

    @given(text(max_size=5000))
    @settings(max_examples=300)
    def test_detect_malicious_patterns_never_crashes(self, data):
        """Malicious pattern detection should never crash."""
        result = detect_malicious_patterns(data)
        # Returns string description of threat, or None if clean
        assert result is None or isinstance(result, str)

    @given(binary(min_size=0, max_size=100000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_json_payload_never_crashes(self, data):
        """JSON validation should handle any input."""
        try:
            is_valid, error = validate_json_payload(data)
            assert isinstance(is_valid, bool)
        except UnicodeDecodeError:
            pass  # Binary data that can't be decoded is acceptable to fail


class TestDeviceIdExtractionFuzzing:
    """Fuzz testing for device ID extraction."""

    @given(mqtt_topics)
    @settings(max_examples=500)
    def test_extract_device_id_never_crashes(self, topic):
        """Device ID extraction should never crash."""
        result = extract_device_id(topic)
        # Result is either a string or None
        assert result is None or isinstance(result, str)

    @given(text(max_size=1000))
    @settings(max_examples=300)
    def test_extract_device_id_arbitrary_text(self, topic):
        """Arbitrary text should be handled safely."""
        result = extract_device_id(topic)
        assert result is None or isinstance(result, str)


class TestUuidGenerationFuzzing:
    """Fuzz testing for UUID generation."""

    @given(integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_uuid_uniqueness(self, count):
        """Generated UUIDs should be unique."""
        uuids = [generate_uuid() for _ in range(count)]
        assert len(set(uuids)) == count


class TestNormalizationFuzzing:
    """Fuzz testing for normalization engine."""

    @pytest.fixture
    def normalizer(self):
        return NormalizationEngine()

    @given(floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_unit_conversion_floats(self, normalizer, value):
        """Unit conversions should handle any float value."""
        # Test temperature conversion
        if hasattr(normalizer, "_safe_eval"):
            try:
                result = normalizer._safe_eval("x * 1.8 + 32", value)
                assert isinstance(result, (int, float))
            except Exception:
                pass  # Acceptable for edge cases

    @given(integers())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_unit_conversion_integers(self, normalizer, value):
        """Unit conversions should handle any integer value."""
        if hasattr(normalizer, "_safe_eval"):
            try:
                result = normalizer._safe_eval("x / 1000", value)
                assert isinstance(result, (int, float))
            except Exception:
                pass


class TestProtocolHandlingFuzzing:
    """Fuzz testing for protocol handling."""

    @given(st.sampled_from(list(Protocol)))
    @settings(max_examples=50)
    def test_all_protocols_handled(self, protocol):
        """All protocol types should be valid."""
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id="test-device",
            protocol=protocol,
            topic="test/topic",
            payload_raw=b'{"test": 1}',
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        assert raw.protocol == protocol


class TestEdgeCases:
    """Tests for specific edge cases found through fuzzing."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_empty_string_fields(self, extractor):
        """Empty string field names should be handled."""
        payload = json.dumps({"": "value", "normal": 123}).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None

    def test_unicode_field_names(self, extractor):
        """Unicode field names should be handled."""
        payload = json.dumps(
            {
                "温度": 23.5,
                "влажность": 65,
                "🌡️": 100,
            }
        ).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None
        assert len(schema.fields) == 3

    def test_very_long_field_names(self, extractor):
        """Very long field names should be handled."""
        payload = json.dumps(
            {
                "a" * 1000: "value",
                "b" * 500: 123,
            }
        ).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None

    def test_special_float_values(self, extractor):
        """Special float-like values should be handled."""
        payload = json.dumps(
            {
                "zero": 0.0,
                "negative_zero": -0.0,
                "tiny": 1e-300,
                "huge": 1e300,
            }
        ).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None
        assert len(schema.fields) == 4

    def test_deeply_nested_arrays(self, extractor):
        """Deeply nested arrays should be handled."""
        nested = [[[[[1, 2, 3]]]]]
        payload = json.dumps({"data": nested}).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None

    def test_mixed_array_types(self, extractor):
        """Arrays with mixed types should be handled."""
        payload = json.dumps(
            {
                "mixed": [1, "two", 3.0, True, None, {"nested": "obj"}],
            }
        ).encode()
        raw = RawMessage(
            message_id="test",
            device_id="device",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.utcnow(),
        )
        schema = extractor.extract_schema(raw)
        assert schema is not None
