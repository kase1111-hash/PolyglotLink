"""
Unit tests for the Validation utilities module.
"""

import pytest

from polyglotlink.utils.validation import (
    detect_malicious_patterns,
    sanitize_dict_keys,
    sanitize_identifier,
    sanitize_string,
    sanitize_topic,
    validate_confidence,
    validate_field_type,
    validate_json_depth,
    validate_json_size,
    validate_number,
    validate_payload_size,
    validate_protocol,
)
from polyglotlink.utils.exceptions import ValidationError


class TestSanitizeString:
    """Tests for string sanitization."""

    def test_basic_string(self):
        result = sanitize_string("hello world")
        assert result == "hello world"

    def test_strips_whitespace(self):
        result = sanitize_string("  hello  ")
        assert result == "hello"

    def test_truncates_long_string(self):
        long_string = "a" * 20000
        result = sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_escapes_html(self):
        result = sanitize_string("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_removes_control_chars(self):
        result = sanitize_string("hello\x00\x01world")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserves_newlines_tabs(self):
        result = sanitize_string("line1\nline2\ttab", strip_control_chars=True)
        assert "\n" in result
        assert "\t" in result

    def test_non_string_input(self):
        result = sanitize_string(42)
        assert result == "42"


class TestSanitizeIdentifier:
    """Tests for identifier sanitization."""

    def test_basic_identifier(self):
        result = sanitize_identifier("device_01")
        assert result == "device_01"

    def test_removes_special_chars(self):
        result = sanitize_identifier("device@#$%01")
        assert result == "device01"

    def test_allows_dots(self):
        result = sanitize_identifier("sensor.temp", allow_dots=True)
        assert result == "sensor.temp"

    def test_disallows_dots(self):
        result = sanitize_identifier("sensor.temp", allow_dots=False)
        assert result == "sensortemp"

    def test_allows_dashes(self):
        result = sanitize_identifier("device-01", allow_dashes=True)
        assert result == "device-01"

    def test_truncates_length(self):
        long_id = "a" * 500
        result = sanitize_identifier(long_id, max_length=100)
        assert len(result) == 100

    def test_strips_leading_special(self):
        result = sanitize_identifier("_device01")
        assert result == "device01"

    def test_empty_identifier_raises(self):
        with pytest.raises(ValidationError):
            sanitize_identifier("")

    def test_all_special_chars_raises(self):
        with pytest.raises(ValidationError):
            sanitize_identifier("@#$%^&")


class TestSanitizeTopic:
    """Tests for MQTT topic sanitization."""

    def test_basic_topic(self):
        result = sanitize_topic("devices/sensor01/telemetry")
        assert result == "devices/sensor01/telemetry"

    def test_removes_dangerous_chars(self):
        result = sanitize_topic("devices/sensor$01/data")
        assert result == "devices/sensor01/data"

    def test_preserves_wildcards(self):
        result = sanitize_topic("devices/+/telemetry")
        assert result == "devices/+/telemetry"

        result = sanitize_topic("sensors/#")
        assert result == "sensors/#"

    def test_collapses_slashes(self):
        result = sanitize_topic("devices///sensor//data")
        assert result == "devices/sensor/data"

    def test_strips_leading_trailing_slashes(self):
        result = sanitize_topic("/devices/sensor/")
        assert result == "devices/sensor"

    def test_truncates_length(self):
        long_topic = "a/" * 1000
        result = sanitize_topic(long_topic, max_length=100)
        assert len(result) <= 100


class TestValidateJsonDepth:
    """Tests for JSON depth validation."""

    def test_shallow_dict(self):
        data = {"a": 1, "b": 2}
        assert validate_json_depth(data, max_depth=5) is True

    def test_nested_dict_within_limit(self):
        data = {"a": {"b": {"c": 1}}}
        assert validate_json_depth(data, max_depth=5) is True

    def test_nested_dict_exceeds_limit(self):
        data = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
        with pytest.raises(ValidationError):
            validate_json_depth(data, max_depth=3)

    def test_array_depth(self):
        data = {"items": [{"nested": {"deep": 1}}]}
        assert validate_json_depth(data, max_depth=5) is True

    def test_deeply_nested_array(self):
        data = [[[[[[1]]]]]]
        with pytest.raises(ValidationError):
            validate_json_depth(data, max_depth=3)


class TestValidateJsonSize:
    """Tests for JSON size validation."""

    def test_small_dict(self):
        data = {"a": 1, "b": 2}
        assert validate_json_size(data, max_keys=100) is True

    def test_too_many_keys(self):
        data = {f"key{i}": i for i in range(100)}
        with pytest.raises(ValidationError):
            validate_json_size(data, max_keys=50)

    def test_string_length_ok(self):
        data = {"text": "a" * 100}
        assert validate_json_size(data, max_string_length=1000) is True

    def test_string_too_long(self):
        data = {"text": "a" * 10000}
        with pytest.raises(ValidationError):
            validate_json_size(data, max_string_length=1000)


class TestSanitizeDictKeys:
    """Tests for dictionary key sanitization."""

    def test_basic_keys(self):
        data = {"temperature": 23.5, "humidity": 65}
        result = sanitize_dict_keys(data)
        assert "temperature" in result
        assert "humidity" in result

    def test_removes_special_chars_from_keys(self):
        data = {"temp@erature": 23.5}
        result = sanitize_dict_keys(data)
        assert "temperature" in result

    def test_nested_dict(self):
        data = {"sensor@data": {"temp@c": 23.5}}
        result = sanitize_dict_keys(data)
        assert "sensordata" in result
        assert "tempc" in result["sensordata"]

    def test_array_values(self):
        data = {"items@list": [{"val@ue": 1}]}
        result = sanitize_dict_keys(data)
        assert "itemslist" in result
        assert result["itemslist"][0]["value"] == 1


class TestValidateNumber:
    """Tests for numeric value validation."""

    def test_valid_integer(self):
        result = validate_number(42, "value")
        assert result == 42

    def test_valid_float(self):
        result = validate_number(3.14, "value")
        assert result == 3.14

    def test_string_to_number(self):
        result = validate_number("42", "value")
        assert result == 42

    def test_min_value(self):
        result = validate_number(10, "value", min_value=0)
        assert result == 10

        with pytest.raises(ValidationError):
            validate_number(-1, "value", min_value=0)

    def test_max_value(self):
        result = validate_number(50, "value", max_value=100)
        assert result == 50

        with pytest.raises(ValidationError):
            validate_number(101, "value", max_value=100)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            validate_number(float("nan"), "value", allow_nan=False)

    def test_nan_allowed(self):
        import math
        result = validate_number(float("nan"), "value", allow_nan=True)
        assert math.isnan(result)

    def test_inf_rejected(self):
        with pytest.raises(ValidationError):
            validate_number(float("inf"), "value", allow_inf=False)

    def test_inf_allowed(self):
        import math
        result = validate_number(float("inf"), "value", allow_inf=True)
        assert math.isinf(result)

    def test_boolean_rejected(self):
        with pytest.raises(ValidationError):
            validate_number(True, "value")

    def test_invalid_string(self):
        with pytest.raises(ValidationError):
            validate_number("not a number", "value")


class TestValidatePayloadSize:
    """Tests for payload size validation."""

    def test_small_payload(self):
        payload = b"small data"
        assert validate_payload_size(payload, max_size=1000) is True

    def test_large_payload(self):
        payload = b"x" * 10000
        with pytest.raises(ValidationError):
            validate_payload_size(payload, max_size=1000)

    def test_exact_limit(self):
        payload = b"x" * 100
        assert validate_payload_size(payload, max_size=100) is True


class TestDetectMaliciousPatterns:
    """Tests for malicious pattern detection."""

    def test_clean_data(self):
        result = detect_malicious_patterns("normal sensor data 123")
        assert result is None

    def test_sql_injection_or_1_1(self):
        result = detect_malicious_patterns("user' OR 1=1 --")
        assert result is not None
        assert "SQL" in result

    def test_sql_injection_union(self):
        result = detect_malicious_patterns("' UNION SELECT * FROM users")
        assert result is not None
        assert "SQL" in result

    def test_xss_script_tag(self):
        result = detect_malicious_patterns("<script>alert('xss')</script>")
        assert result is not None
        assert "Script" in result

    def test_xss_event_handler(self):
        result = detect_malicious_patterns('<img onerror="alert(1)">')
        assert result is not None
        assert "Script" in result

    def test_path_traversal(self):
        result = detect_malicious_patterns("../../etc/passwd")
        assert result is not None
        assert "Path" in result

    def test_path_traversal_encoded(self):
        result = detect_malicious_patterns("%2e%2e/etc/passwd")
        assert result is not None
        assert "Path" in result

    def test_disable_specific_checks(self):
        # SQL check disabled, so SQL injection not detected
        result = detect_malicious_patterns(
            "OR 1=1",
            check_sql=False,
            check_script=True,
            check_path_traversal=True
        )
        assert result is None


class TestValidateFieldType:
    """Tests for field type validation."""

    def test_valid_types(self):
        assert validate_field_type("string") is True
        assert validate_field_type("integer") is True
        assert validate_field_type("float") is True
        assert validate_field_type("boolean") is True
        assert validate_field_type("datetime") is True
        assert validate_field_type("array") is True
        assert validate_field_type("object") is True
        assert validate_field_type("null") is True

    def test_case_insensitive(self):
        assert validate_field_type("STRING") is True
        assert validate_field_type("Integer") is True

    def test_invalid_type(self):
        assert validate_field_type("invalid_type") is False


class TestValidateProtocol:
    """Tests for protocol validation."""

    def test_valid_protocols(self):
        assert validate_protocol("MQTT") is True
        assert validate_protocol("HTTP") is True
        assert validate_protocol("CoAP") is True
        assert validate_protocol("Modbus") is True
        assert validate_protocol("OPC-UA") is True
        assert validate_protocol("WebSocket") is True

    def test_invalid_protocol(self):
        assert validate_protocol("INVALID") is False
        assert validate_protocol("mqtt") is False  # Case sensitive


class TestValidateConfidence:
    """Tests for confidence score validation."""

    def test_valid_confidence(self):
        assert validate_confidence(0.0) == 0.0
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(1.0) == 1.0

    def test_invalid_confidence_below(self):
        with pytest.raises(ValidationError):
            validate_confidence(-0.1)

    def test_invalid_confidence_above(self):
        with pytest.raises(ValidationError):
            validate_confidence(1.1)
