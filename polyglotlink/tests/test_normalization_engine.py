"""
Unit tests for the Normalization Engine module.
"""

import pytest
from datetime import datetime

from polyglotlink.models.schemas import (
    ExtractedField,
    ExtractedSchema,
    FieldMapping,
    Protocol,
    ResolutionMethod,
    SemanticMapping,
)
from polyglotlink.modules.normalization_engine import (
    NormalizationEngine,
    apply_conversion,
    enforce_type,
    get_unit_conversion,
    validate_value,
    Concept,
    ConversionError,
    UnsafeFormulaError,
)


class TestGetUnitConversion:
    """Tests for unit conversion formula lookup."""

    def test_same_unit(self):
        assert get_unit_conversion("celsius", "celsius") is None

    def test_celsius_to_fahrenheit(self):
        formula = get_unit_conversion("celsius", "fahrenheit")
        assert formula is not None
        assert "9/5" in formula or "1.8" in formula

    def test_fahrenheit_to_celsius(self):
        formula = get_unit_conversion("fahrenheit", "celsius")
        assert formula is not None

    def test_pascal_to_bar(self):
        formula = get_unit_conversion("pascal", "bar")
        assert formula is not None

    def test_meters_per_second_to_kmh(self):
        formula = get_unit_conversion("meters_per_second", "kilometers_per_hour")
        assert formula is not None
        assert "3.6" in formula

    def test_unknown_conversion(self):
        # No direct conversion between unrelated units
        formula = get_unit_conversion("celsius", "pascal")
        assert formula is None


class TestApplyConversion:
    """Tests for conversion formula execution."""

    def test_celsius_to_fahrenheit(self):
        # 0°C = 32°F
        result = apply_conversion(0, "(value * 9/5) + 32")
        assert result == 32.0

        # 100°C = 212°F
        result = apply_conversion(100, "(value * 9/5) + 32")
        assert result == 212.0

    def test_fahrenheit_to_celsius(self):
        # 32°F = 0°C
        result = apply_conversion(32, "(value - 32) * 5/9")
        assert abs(result - 0.0) < 0.001

        # 212°F = 100°C
        result = apply_conversion(212, "(value - 32) * 5/9")
        assert abs(result - 100.0) < 0.001

    def test_none_value(self):
        result = apply_conversion(None, "(value * 2)")
        assert result is None

    def test_simple_multiplication(self):
        result = apply_conversion(10, "value * 3.6")
        assert result == 36.0

    def test_precision_limiting(self):
        result = apply_conversion(1, "value / 3")
        # Should be rounded to 6 decimal places
        assert len(str(result).split(".")[-1]) <= 6

    def test_unsafe_formula_letters(self):
        with pytest.raises(UnsafeFormulaError):
            apply_conversion(10, "value + import('os')")

    def test_unsafe_formula_functions(self):
        with pytest.raises(UnsafeFormulaError):
            apply_conversion(10, "eval(value)")


class TestEnforceType:
    """Tests for type enforcement."""

    def test_to_float(self):
        assert enforce_type(42, "float") == 42.0
        assert enforce_type("3.14", "float") == 3.14
        assert enforce_type(True, "float") == 1.0

    def test_to_integer(self):
        assert enforce_type(42.9, "integer") == 42
        assert enforce_type("42", "integer") == 42

    def test_to_string(self):
        assert enforce_type(42, "string") == "42"
        assert enforce_type(3.14, "string") == "3.14"

    def test_to_boolean(self):
        assert enforce_type(True, "boolean") is True
        assert enforce_type("true", "boolean") is True
        assert enforce_type("yes", "boolean") is True
        assert enforce_type("1", "boolean") is True
        assert enforce_type("false", "boolean") is False
        assert enforce_type("no", "boolean") is False
        assert enforce_type(1, "boolean") is True
        assert enforce_type(0, "boolean") is False

    def test_none_value(self):
        assert enforce_type(None, "float") is None
        assert enforce_type(None, "integer") is None

    def test_to_datetime_iso(self):
        result = enforce_type("2024-01-15T10:30:00+00:00", "datetime")
        assert isinstance(result, datetime)

    def test_to_datetime_unix(self):
        result = enforce_type(1705312200, "datetime")
        assert isinstance(result, datetime)

    def test_to_datetime_unix_ms(self):
        result = enforce_type(1705312200000, "datetime")
        assert isinstance(result, datetime)


class TestValidateValue:
    """Tests for value validation."""

    def test_within_range(self):
        concept = Concept("temp", "float", min_value=-40, max_value=100)
        assert validate_value(25, concept) is True
        assert validate_value(-40, concept) is True
        assert validate_value(100, concept) is True

    def test_below_min(self):
        concept = Concept("temp", "float", min_value=0, max_value=100)
        assert validate_value(-1, concept) is False

    def test_above_max(self):
        concept = Concept("temp", "float", min_value=0, max_value=100)
        assert validate_value(101, concept) is False

    def test_none_value(self):
        concept = Concept("temp", "float", min_value=0, max_value=100)
        assert validate_value(None, concept) is True

    def test_no_constraints(self):
        concept = Concept("value", "float")
        assert validate_value(1000000, concept) is True
        assert validate_value(-1000000, concept) is True


class TestNormalizationEngine:
    """Tests for the NormalizationEngine class."""

    @pytest.fixture
    def engine(self):
        return NormalizationEngine()

    @pytest.fixture
    def sample_schema(self):
        return ExtractedSchema(
            message_id="test-001",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            fields=[
                ExtractedField(
                    key="temperature",
                    original_key="temperature",
                    value=23.5,
                    value_type="float",
                    inferred_unit="celsius",
                    inferred_semantic="temperature",
                    is_timestamp=False,
                    is_identifier=False,
                ),
                ExtractedField(
                    key="humidity",
                    original_key="humidity",
                    value=65,
                    value_type="integer",
                    inferred_unit="percent",
                    inferred_semantic="humidity",
                    is_timestamp=False,
                    is_identifier=False,
                ),
                ExtractedField(
                    key="timestamp",
                    original_key="timestamp",
                    value="2024-01-15T10:30:00Z",
                    value_type="datetime",
                    is_timestamp=True,
                    is_identifier=False,
                ),
            ],
            schema_signature="abc123",
            payload_decoded={"temperature": 23.5, "humidity": 65},
            extracted_at=datetime.utcnow(),
        )

    @pytest.fixture
    def sample_mapping(self):
        return SemanticMapping(
            message_id="test-001",
            device_id="sensor-01",
            schema_signature="abc123",
            field_mappings=[
                FieldMapping(
                    source_field="temperature",
                    target_concept="temperature_celsius",
                    target_field="temperature_celsius",
                    source_unit="celsius",
                    target_unit="celsius",
                    confidence=0.95,
                    resolution_method=ResolutionMethod.EMBEDDING,
                ),
                FieldMapping(
                    source_field="humidity",
                    target_concept="humidity_percent",
                    target_field="humidity_percent",
                    source_unit="percent",
                    target_unit="percent",
                    confidence=0.95,
                    resolution_method=ResolutionMethod.EMBEDDING,
                ),
                FieldMapping(
                    source_field="timestamp",
                    target_concept="_timestamp",
                    target_field="timestamp",
                    confidence=1.0,
                    resolution_method=ResolutionMethod.PASSTHROUGH,
                ),
            ],
            confidence=0.95,
            llm_generated=False,
            translated_at=datetime.utcnow(),
        )

    def test_basic_normalization(self, engine, sample_schema, sample_mapping):
        result = engine.normalize_message(sample_schema, sample_mapping)

        assert result.message_id == "test-001"
        assert result.device_id == "sensor-01"
        assert "temperature_celsius" in result.data
        assert "humidity_percent" in result.data
        assert result.data["temperature_celsius"] == 23.5
        assert result.data["humidity_percent"] == 65

    def test_unit_conversion(self, engine, sample_schema):
        # Modify mapping to require conversion
        mapping = SemanticMapping(
            message_id="test-001",
            device_id="sensor-01",
            schema_signature="abc123",
            field_mappings=[
                FieldMapping(
                    source_field="temperature",
                    target_concept="temperature_fahrenheit",
                    target_field="temperature_fahrenheit",
                    source_unit="celsius",
                    target_unit="fahrenheit",
                    conversion_formula="(value * 9/5) + 32",
                    confidence=0.95,
                    resolution_method=ResolutionMethod.LLM,
                ),
            ],
            confidence=0.95,
            llm_generated=True,
            translated_at=datetime.utcnow(),
        )

        result = engine.normalize_message(sample_schema, mapping)

        # 23.5°C = 74.3°F
        assert "temperature_fahrenheit" in result.data
        assert abs(result.data["temperature_fahrenheit"] - 74.3) < 0.1

        # Check conversion was recorded
        assert len(result.conversions) == 1
        assert result.conversions[0].from_unit == "celsius"
        assert result.conversions[0].to_unit == "fahrenheit"

    def test_unmapped_fields(self, engine, sample_schema):
        # Mapping without humidity
        mapping = SemanticMapping(
            message_id="test-001",
            device_id="sensor-01",
            schema_signature="abc123",
            field_mappings=[
                FieldMapping(
                    source_field="temperature",
                    target_concept="temperature_celsius",
                    target_field="temperature_celsius",
                    confidence=0.95,
                    resolution_method=ResolutionMethod.EMBEDDING,
                ),
            ],
            confidence=0.95,
            llm_generated=False,
            translated_at=datetime.utcnow(),
        )

        result = engine.normalize_message(sample_schema, mapping)

        # Humidity should be in unmapped fields
        assert "_unmapped.humidity" in result.data

    def test_metadata_enrichment(self, engine, sample_schema, sample_mapping):
        result = engine.normalize_message(sample_schema, sample_mapping)

        assert "source_protocol" in result.metadata
        assert result.metadata["source_protocol"] == "MQTT"
        assert "source_topic" in result.metadata
        assert result.metadata["source_topic"] == "sensors/data"

    def test_timestamp_extraction(self, engine, sample_schema, sample_mapping):
        result = engine.normalize_message(sample_schema, sample_mapping)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_validation_errors(self, engine):
        # Create schema with out-of-range value
        schema = ExtractedSchema(
            message_id="test-001",
            device_id="sensor-01",
            protocol=Protocol.MQTT,
            topic="sensors/data",
            fields=[
                ExtractedField(
                    key="humidity",
                    original_key="humidity",
                    value=150,  # Invalid: > 100%
                    value_type="integer",
                    is_timestamp=False,
                    is_identifier=False,
                ),
            ],
            schema_signature="abc123",
            payload_decoded={"humidity": 150},
            extracted_at=datetime.utcnow(),
        )

        mapping = SemanticMapping(
            message_id="test-001",
            device_id="sensor-01",
            schema_signature="abc123",
            field_mappings=[
                FieldMapping(
                    source_field="humidity",
                    target_concept="humidity_percent",
                    target_field="humidity_percent",
                    confidence=0.95,
                    resolution_method=ResolutionMethod.EMBEDDING,
                ),
            ],
            confidence=0.95,
            llm_generated=False,
            translated_at=datetime.utcnow(),
        )

        result = engine.normalize_message(schema, mapping)

        # Should have validation error
        assert len(result.validation_errors) > 0
        assert "_invalid.humidity_percent" in result.data

    def test_device_registry(self, engine):
        engine.register_device(
            device_id="sensor-01",
            device_type="environmental",
            name="Office Sensor",
            location="Building A",
            tags=["indoor", "office"],
        )

        device = engine.device_registry.get("sensor-01")
        assert device is not None
        assert device.type == "environmental"
        assert device.name == "Office Sensor"
