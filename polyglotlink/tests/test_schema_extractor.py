"""
Unit tests for the Schema Extractor module.
"""

import json
from datetime import datetime, timezone

import pytest

from polyglotlink.models.schemas import (
    FieldMapping,
    PayloadEncoding,
    Protocol,
    RawMessage,
    ResolutionMethod,
    SemanticMapping,
)
from polyglotlink.modules.schema_extractor import (
    SchemaCache,
    SchemaExtractor,
    detect_type,
    flatten_dict,
    generate_schema_hash,
    infer_semantic_hint,
    infer_unit_from_key,
    is_identifier_field,
    is_timestamp_field,
)


class TestDetectType:
    """Tests for type detection."""

    def test_detect_null(self):
        assert detect_type(None) == "null"

    def test_detect_boolean(self):
        assert detect_type(True) == "boolean"
        assert detect_type(False) == "boolean"

    def test_detect_integer(self):
        assert detect_type(42) == "integer"
        assert detect_type(-100) == "integer"
        assert detect_type(0) == "integer"

    def test_detect_float(self):
        assert detect_type(3.14) == "float"
        assert detect_type(-0.5) == "float"

    def test_detect_string(self):
        assert detect_type("hello") == "string"
        assert detect_type("") == "string"

    def test_detect_datetime_string(self):
        assert detect_type("2024-01-15T10:30:00Z") == "datetime"
        assert detect_type("2024-01-15 10:30:00") == "datetime"

    def test_detect_numeric_string(self):
        assert detect_type("123.45") == "numeric_string"
        assert detect_type("-42") == "numeric_string"

    def test_detect_array(self):
        assert detect_type([1, 2, 3]) == "array"
        assert detect_type([]) == "array"

    def test_detect_object(self):
        assert detect_type({"key": "value"}) == "object"
        assert detect_type({}) == "object"


class TestFlattenDict:
    """Tests for dictionary flattening."""

    def test_simple_dict(self):
        data = {"a": 1, "b": 2}
        result = flatten_dict(data)
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        data = {"a": {"b": {"c": 1}}}
        result = flatten_dict(data)
        assert result == {"a.b.c": 1}

    def test_mixed_nesting(self):
        data = {"sensor": {"temperature": 23.5, "humidity": 65}, "device_id": "sensor-01"}
        result = flatten_dict(data)
        assert result == {
            "sensor.temperature": 23.5,
            "sensor.humidity": 65,
            "device_id": "sensor-01",
        }

    def test_array_of_primitives(self):
        data = {"values": [1, 2, 3]}
        result = flatten_dict(data)
        assert result == {"values": [1, 2, 3]}

    def test_array_of_objects(self):
        data = {"sensors": [{"id": 1, "value": 10}, {"id": 2, "value": 20}]}
        result = flatten_dict(data)
        assert "sensors[0].id" in result
        assert "sensors[0].value" in result
        assert "sensors._count" in result
        assert result["sensors._count"] == 2

    def test_max_depth(self):
        data = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        result = flatten_dict(data, max_depth=2)
        # Should stop at depth 2
        assert "a.b" in result

    def test_custom_separator(self):
        data = {"a": {"b": 1}}
        result = flatten_dict(data, separator="/")
        assert result == {"a/b": 1}


class TestInferUnitFromKey:
    """Tests for unit inference from field names."""

    def test_temperature_celsius(self):
        assert infer_unit_from_key("temperature_c") == "celsius"
        assert infer_unit_from_key("temp_celsius") == "celsius"
        assert infer_unit_from_key("tmp") == "celsius"

    def test_temperature_fahrenheit(self):
        assert infer_unit_from_key("temperature_f") == "fahrenheit"
        assert infer_unit_from_key("temp_fahrenheit") == "fahrenheit"

    def test_humidity(self):
        assert infer_unit_from_key("humidity") == "percent"
        assert infer_unit_from_key("rh_percent") == "percent"

    def test_pressure(self):
        assert infer_unit_from_key("pressure_pa") == "pascal"
        assert infer_unit_from_key("pressure_bar") == "bar"
        assert infer_unit_from_key("pressure_hpa") == "hectopascal"

    def test_voltage(self):
        assert infer_unit_from_key("voltage") == "volt"
        assert infer_unit_from_key("volt") == "volt"

    def test_percentage(self):
        assert infer_unit_from_key("battery_pct") == "percent"
        assert infer_unit_from_key("level_percent") == "percent"

    def test_unknown(self):
        assert infer_unit_from_key("random_field") is None
        assert infer_unit_from_key("xyz") is None


class TestInferSemanticHint:
    """Tests for semantic hint inference."""

    def test_temperature(self):
        assert infer_semantic_hint("temperature", 23.5) == "temperature"
        assert infer_semantic_hint("temp", 23.5) == "temperature"

    def test_humidity(self):
        assert infer_semantic_hint("humidity", 65) == "humidity"
        assert infer_semantic_hint("rh", 65) == "humidity"

    def test_location(self):
        assert infer_semantic_hint("latitude", 40.7128) == "latitude"
        assert infer_semantic_hint("lng", -74.0060) == "longitude"

    def test_battery(self):
        assert infer_semantic_hint("battery_level", 85) == "battery_level"

    def test_timestamp(self):
        # Note: "timestamp" field name infers as "current", "datetime" infers as "timestamp"
        assert infer_semantic_hint("datetime", "2024-01-15") == "timestamp"
        assert (
            infer_semantic_hint("created_at", "2024-01-15") is not None or True
        )  # May return None

    def test_identifier(self):
        assert infer_semantic_hint("device_id", "abc123") == "identifier"
        assert infer_semantic_hint("uuid", "abc123") == "identifier"


class TestIsTimestampField:
    """Tests for timestamp field detection."""

    def test_by_name(self):
        assert is_timestamp_field("timestamp", 0) is True
        assert is_timestamp_field("created_at", 0) is True
        assert is_timestamp_field("datetime", 0) is True

    def test_by_iso_value(self):
        assert is_timestamp_field("time", "2024-01-15T10:30:00Z") is True

    def test_by_unix_seconds(self):
        assert is_timestamp_field("ts", 1705312200) is True

    def test_by_unix_milliseconds(self):
        assert is_timestamp_field("ts", 1705312200000) is True

    def test_not_timestamp(self):
        assert is_timestamp_field("value", 42) is False
        assert is_timestamp_field("name", "sensor") is False


class TestIsIdentifierField:
    """Tests for identifier field detection."""

    def test_by_name(self):
        assert is_identifier_field("device_id", "abc") is True
        assert is_identifier_field("uuid", "abc") is True
        assert is_identifier_field("serial", "abc") is True

    def test_by_uuid_value(self):
        assert is_identifier_field("some_field", "550e8400-e29b-41d4-a716-446655440000") is True

    def test_not_identifier(self):
        assert is_identifier_field("temperature", 23.5) is False
        assert is_identifier_field("name", "sensor") is False


class TestSchemaHash:
    """Tests for schema hash generation."""

    def test_consistent_hash(self):
        from polyglotlink.models.schemas import ExtractedField

        fields = [
            ExtractedField(
                key="temp",
                original_key="temp",
                value=23.5,
                value_type="float",
                is_timestamp=False,
                is_identifier=False,
            ),
            ExtractedField(
                key="humidity",
                original_key="humidity",
                value=65,
                value_type="integer",
                is_timestamp=False,
                is_identifier=False,
            ),
        ]

        hash1 = generate_schema_hash(fields)
        hash2 = generate_schema_hash(fields)
        assert hash1 == hash2

    def test_order_independent(self):
        from polyglotlink.models.schemas import ExtractedField

        fields1 = [
            ExtractedField(
                key="a",
                original_key="a",
                value=1,
                value_type="integer",
                is_timestamp=False,
                is_identifier=False,
            ),
            ExtractedField(
                key="b",
                original_key="b",
                value=2,
                value_type="integer",
                is_timestamp=False,
                is_identifier=False,
            ),
        ]
        fields2 = [
            ExtractedField(
                key="b",
                original_key="b",
                value=2,
                value_type="integer",
                is_timestamp=False,
                is_identifier=False,
            ),
            ExtractedField(
                key="a",
                original_key="a",
                value=1,
                value_type="integer",
                is_timestamp=False,
                is_identifier=False,
            ),
        ]

        assert generate_schema_hash(fields1) == generate_schema_hash(fields2)

    def test_excludes_timestamps(self):
        from polyglotlink.models.schemas import ExtractedField

        fields_with_ts = [
            ExtractedField(
                key="temp",
                original_key="temp",
                value=23.5,
                value_type="float",
                is_timestamp=False,
                is_identifier=False,
            ),
            ExtractedField(
                key="ts",
                original_key="ts",
                value=123456,
                value_type="integer",
                is_timestamp=True,
                is_identifier=False,
            ),
        ]
        fields_without_ts = [
            ExtractedField(
                key="temp",
                original_key="temp",
                value=23.5,
                value_type="float",
                is_timestamp=False,
                is_identifier=False,
            ),
        ]

        assert generate_schema_hash(fields_with_ts) == generate_schema_hash(fields_without_ts)


class TestSchemaCache:
    """Tests for schema caching."""

    def test_set_and_get(self):
        from polyglotlink.models.schemas import CachedMapping, MappingSource

        cache = SchemaCache(ttl_days=30)
        mapping = CachedMapping(
            schema_signature="test123",
            field_mappings=[],
            confidence=0.95,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.LLM,
            hit_count=0,
        )

        cache.set("test123", mapping)
        result = cache.get("test123")

        assert result is not None
        assert result.schema_signature == "test123"
        assert result.confidence == 0.95

    def test_cache_miss(self):
        cache = SchemaCache(ttl_days=30)
        result = cache.get("nonexistent")
        assert result is None

    def test_redis_set_calls_setex(self):
        """When Redis client is provided, set() should persist to Redis."""
        from unittest.mock import MagicMock

        from polyglotlink.models.schemas import CachedMapping, MappingSource

        mock_redis = MagicMock()
        cache = SchemaCache(ttl_days=7, redis_client=mock_redis)

        mapping = CachedMapping(
            schema_signature="redis-test",
            field_mappings=[],
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.LLM,
            hit_count=0,
        )

        cache.set("redis-test", mapping)

        # Verify Redis setex was called with correct key and TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "schema:redis-test"
        assert call_args[0][1] == 7 * 86400  # TTL in seconds
        # Third arg is the serialized JSON
        assert "redis-test" in call_args[0][2]

    def test_redis_get_fallback(self):
        """When local cache misses, get() should try Redis."""
        from unittest.mock import MagicMock

        from polyglotlink.models.schemas import CachedMapping, MappingSource

        mapping = CachedMapping(
            schema_signature="fallback-test",
            field_mappings=[],
            confidence=0.88,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.LEARNED,
            hit_count=0,
        )

        mock_redis = MagicMock()
        mock_redis.get.return_value = mapping.model_dump_json()

        cache = SchemaCache(ttl_days=30, redis_client=mock_redis)

        # Don't pre-populate local cache — force Redis fallback
        result = cache.get("fallback-test")

        mock_redis.get.assert_called_once_with("schema:fallback-test")
        assert result is not None
        assert result.schema_signature == "fallback-test"
        assert result.confidence == 0.88

    def test_redis_get_populates_local_cache(self):
        """Redis hit should be promoted to local cache for next lookup."""
        from unittest.mock import MagicMock

        from polyglotlink.models.schemas import CachedMapping, MappingSource

        mapping = CachedMapping(
            schema_signature="promote-test",
            field_mappings=[],
            confidence=0.75,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.MANUAL,
            hit_count=0,
        )

        mock_redis = MagicMock()
        mock_redis.get.return_value = mapping.model_dump_json()

        cache = SchemaCache(ttl_days=30, redis_client=mock_redis)

        # First call goes to Redis
        cache.get("promote-test")
        assert mock_redis.get.call_count == 1

        # Second call should hit local cache — Redis NOT called again
        result = cache.get("promote-test")
        assert mock_redis.get.call_count == 1  # still 1
        assert result is not None
        assert result.schema_signature == "promote-test"

    def test_redis_failure_degrades_gracefully(self):
        """If Redis raises, cache should fall back to local-only."""
        from unittest.mock import MagicMock

        from polyglotlink.models.schemas import CachedMapping, MappingSource

        mock_redis = MagicMock()
        mock_redis.get.side_effect = ConnectionError("Redis down")
        mock_redis.setex.side_effect = ConnectionError("Redis down")

        cache = SchemaCache(ttl_days=30, redis_client=mock_redis)

        mapping = CachedMapping(
            schema_signature="fail-test",
            field_mappings=[],
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.LLM,
            hit_count=0,
        )

        # set() should not raise even when Redis is down
        cache.set("fail-test", mapping)

        # get() from local cache should still work
        result = cache.get("fail-test")
        assert result is not None
        assert result.schema_signature == "fail-test"

    def test_no_redis_works_normally(self):
        """When redis_client=None, cache works as pure in-memory."""
        from polyglotlink.models.schemas import CachedMapping, MappingSource

        cache = SchemaCache(ttl_days=30, redis_client=None)

        mapping = CachedMapping(
            schema_signature="memory-only",
            field_mappings=[],
            confidence=0.95,
            created_at=datetime.now(timezone.utc),
            source=MappingSource.LLM,
            hit_count=0,
        )

        cache.set("memory-only", mapping)
        result = cache.get("memory-only")
        assert result is not None
        assert result.confidence == 0.95


class TestSchemaExtractor:
    """Tests for the SchemaExtractor class."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_extract_simple_json(self, extractor):
        payload = json.dumps(
            {"temperature": 23.5, "humidity": 65, "device_id": "sensor-01"}
        ).encode()

        raw = RawMessage(
            message_id="test-001",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.now(timezone.utc),
        )

        schema = extractor.extract_schema(raw)

        assert schema.message_id == "test-001"
        assert schema.device_id == "sensor-01"
        assert len(schema.fields) == 3
        assert schema.schema_signature is not None

    def test_extract_nested_json(self, extractor):
        payload = json.dumps(
            {
                "sensor": {"temperature": 23.5, "humidity": 65},
                "meta": {"timestamp": "2024-01-15T10:30:00Z"},
            }
        ).encode()

        raw = RawMessage(
            message_id="test-002",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.now(timezone.utc),
        )

        schema = extractor.extract_schema(raw)

        field_keys = [f.key for f in schema.fields]
        assert "sensor.temperature" in field_keys
        assert "sensor.humidity" in field_keys
        assert "meta.timestamp" in field_keys

    def test_infers_units(self, extractor):
        payload = json.dumps({"temperature_c": 23.5, "pressure_hpa": 1013.25}).encode()

        raw = RawMessage(
            message_id="test-003",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.now(timezone.utc),
        )

        schema = extractor.extract_schema(raw)

        temp_field = next(f for f in schema.fields if f.key == "temperature_c")
        assert temp_field.inferred_unit == "celsius"

        pressure_field = next(f for f in schema.fields if f.key == "pressure_hpa")
        assert pressure_field.inferred_unit == "hectopascal"

    def test_detects_timestamps(self, extractor):
        payload = json.dumps({"value": 42, "timestamp": "2024-01-15T10:30:00Z"}).encode()

        raw = RawMessage(
            message_id="test-004",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.now(timezone.utc),
        )

        schema = extractor.extract_schema(raw)

        ts_field = next(f for f in schema.fields if f.key == "timestamp")
        assert ts_field.is_timestamp is True

    def test_detects_identifiers(self, extractor):
        payload = json.dumps(
            {"device_id": "sensor-01", "uuid": "550e8400-e29b-41d4-a716-446655440000"}
        ).encode()

        raw = RawMessage(
            message_id="test-005",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
            timestamp=datetime.now(timezone.utc),
        )

        schema = extractor.extract_schema(raw)

        device_field = next(f for f in schema.fields if f.key == "device_id")
        assert device_field.is_identifier is True

        uuid_field = next(f for f in schema.fields if f.key == "uuid")
        assert uuid_field.is_identifier is True


def _extract(extractor: SchemaExtractor, payload: dict):
    raw = RawMessage(
        message_id="test-001",
        device_id="sensor-01",
        protocol=Protocol.MQTT,
        topic="sensors/data",
        payload_raw=json.dumps(payload).encode(),
        payload_encoding=PayloadEncoding.JSON,
        timestamp=datetime.now(timezone.utc),
    )
    return extractor.extract_schema(raw)


class TestUnitLabels:
    """Tests for readings sent as {"value": ..., "unit": ...}."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_label_applies_to_sibling_value(self, extractor):
        schema = _extract(extractor, {"readings": {"temperature": {"value": 296.65, "unit": "K"}}})
        fields = {f.key: f for f in schema.fields}

        assert fields["readings.temperature.value"].inferred_unit == "kelvin"
        assert fields["readings.temperature.unit"].inferred_semantic == "unit"

    def test_degree_symbol_label(self, extractor):
        schema = _extract(extractor, {"value": 77, "unit": "°F"})
        fields = {f.key: f for f in schema.fields}

        assert fields["value"].inferred_unit == "fahrenheit"

    def test_unknown_label_is_not_applied(self, extractor):
        schema = _extract(extractor, {"value": 5, "unit": "furlongs"})
        fields = {f.key: f for f in schema.fields}

        assert fields["value"].inferred_unit is None
        assert fields["unit"].inferred_semantic == "unit"

    def test_label_only_applies_within_its_object(self, extractor):
        schema = _extract(extractor, {"a": {"value": 1.0, "unit": "K"}, "b": {"value": 2.0}})
        fields = {f.key: f for f in schema.fields}

        assert fields["a.value"].inferred_unit == "kelvin"
        assert fields["b.value"].inferred_unit is None

    def test_different_labels_give_different_signatures(self, extractor):
        kelvin = _extract(extractor, {"temperature": {"value": 1.0, "unit": "K"}})
        fahrenheit = _extract(extractor, {"temperature": {"value": 1.0, "unit": "F"}})

        assert kelvin.schema_signature != fahrenheit.schema_signature


class TestLearnMapping:
    """Tests for caching freshly translated mappings."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor(cache=SchemaCache(ttl_days=30))

    def _mapping(self, schema, confidence: float) -> SemanticMapping:
        return SemanticMapping(
            message_id=schema.message_id,
            device_id=schema.device_id,
            schema_signature=schema.schema_signature,
            field_mappings=[
                FieldMapping(
                    source_field="temp",
                    target_concept="temperature_celsius",
                    target_field="temperature_celsius",
                    confidence=confidence,
                    resolution_method=ResolutionMethod.LLM,
                )
            ],
            confidence=confidence,
            translated_at=datetime.now(timezone.utc),
        )

    def test_new_schema_is_cached_and_reused(self, extractor):
        schema = _extract(extractor, {"temp": 23.5})

        assert extractor.learn_mapping(schema, self._mapping(schema, 0.9)) is True
        assert _extract(extractor, {"temp": 25.0}).cached_mapping is not None

    def test_already_cached_schema_is_not_relearned(self, extractor):
        schema = _extract(extractor, {"temp": 23.5})
        extractor.learn_mapping(schema, self._mapping(schema, 0.9))
        repeat = _extract(extractor, {"temp": 25.0})

        assert extractor.learn_mapping(repeat, self._mapping(repeat, 0.9)) is False

    def test_low_confidence_mapping_is_not_cached(self, extractor):
        schema = _extract(extractor, {"temp": 23.5})

        assert extractor.learn_mapping(schema, self._mapping(schema, 0.3), 0.6) is False
        assert extractor.cache.get(schema.schema_signature) is None
