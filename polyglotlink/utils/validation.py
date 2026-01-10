"""
PolyglotLink Input Validation Module

Provides validation and sanitization utilities for incoming data.
Protects against injection attacks and malformed input.
"""

import html
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

from polyglotlink.utils.exceptions import ValidationError

T = TypeVar("T")


# =============================================================================
# String Sanitization
# =============================================================================

# Characters that should be stripped or escaped
CONTROL_CHARS = set(chr(i) for i in range(32) if chr(i) not in '\t\n\r')
SQL_DANGEROUS_CHARS = {"'", '"', ";", "--", "/*", "*/", "xp_"}
SHELL_DANGEROUS_CHARS = {"|", "&", ";", "$", "`", "(", ")", "{", "}", "[", "]", "<", ">", "\\"}


def sanitize_string(
    value: str,
    max_length: int = 10000,
    strip_control_chars: bool = True,
    strip_html: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """
    Sanitize a string value for safe processing.

    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        strip_control_chars: Remove control characters
        strip_html: Escape HTML entities
        normalize_unicode: Normalize unicode characters

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)

    # Truncate to max length
    if len(value) > max_length:
        value = value[:max_length]

    # Normalize unicode
    if normalize_unicode:
        value = unicodedata.normalize("NFKC", value)

    # Strip control characters
    if strip_control_chars:
        value = "".join(c for c in value if c not in CONTROL_CHARS)

    # Escape HTML
    if strip_html:
        value = html.escape(value)

    return value.strip()


def sanitize_identifier(
    value: str,
    max_length: int = 255,
    allow_dots: bool = True,
    allow_dashes: bool = True,
    allow_underscores: bool = True,
) -> str:
    """
    Sanitize an identifier (device ID, field name, etc.).

    Args:
        value: Identifier to sanitize
        max_length: Maximum allowed length
        allow_dots: Allow dots in identifier
        allow_dashes: Allow dashes in identifier
        allow_underscores: Allow underscores in identifier

    Returns:
        Sanitized identifier

    Raises:
        ValidationError: If identifier is invalid
    """
    if not value:
        raise ValidationError(
            field="identifier",
            value=value,
            reason="Identifier cannot be empty"
        )

    # Build allowed characters pattern
    allowed = r"a-zA-Z0-9"
    if allow_dots:
        allowed += r"\."
    if allow_dashes:
        allowed += r"\-"
    if allow_underscores:
        allowed += r"_"

    # Remove disallowed characters
    pattern = f"[^{allowed}]"
    sanitized = re.sub(pattern, "", value)

    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Must start with alphanumeric
    if sanitized and not sanitized[0].isalnum():
        sanitized = sanitized.lstrip("._-")

    if not sanitized:
        raise ValidationError(
            field="identifier",
            value=value,
            reason="Identifier contains no valid characters"
        )

    return sanitized


def sanitize_topic(value: str, max_length: int = 1000) -> str:
    """
    Sanitize an MQTT topic or path.

    Args:
        value: Topic to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized topic
    """
    # Allow alphanumeric, slashes, plus, hash, dots, dashes, underscores
    sanitized = re.sub(r"[^a-zA-Z0-9/+#._\-]", "", value)

    # Collapse multiple slashes
    sanitized = re.sub(r"/+", "/", sanitized)

    # Remove leading/trailing slashes
    sanitized = sanitized.strip("/")

    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


# =============================================================================
# JSON/Dict Validation
# =============================================================================

def validate_json_depth(
    data: Union[Dict, List],
    max_depth: int = 20,
    current_depth: int = 0
) -> bool:
    """
    Validate that JSON nesting depth doesn't exceed maximum.

    Args:
        data: JSON data to validate
        max_depth: Maximum allowed nesting depth
        current_depth: Current depth (used for recursion)

    Returns:
        True if depth is valid

    Raises:
        ValidationError: If depth exceeds maximum
    """
    if current_depth > max_depth:
        raise ValidationError(
            field="json",
            value=f"depth={current_depth}",
            reason=f"JSON nesting depth exceeds maximum of {max_depth}"
        )

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, (dict, list)):
                validate_json_depth(value, max_depth, current_depth + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                validate_json_depth(item, max_depth, current_depth + 1)

    return True


def validate_json_size(
    data: Union[Dict, List],
    max_keys: int = 1000,
    max_string_length: int = 100000,
) -> bool:
    """
    Validate JSON data size constraints.

    Args:
        data: JSON data to validate
        max_keys: Maximum number of keys across all objects
        max_string_length: Maximum string value length

    Returns:
        True if size is valid

    Raises:
        ValidationError: If size exceeds limits
    """
    key_count = 0

    def count_and_validate(obj: Any) -> None:
        nonlocal key_count

        if isinstance(obj, dict):
            key_count += len(obj)
            if key_count > max_keys:
                raise ValidationError(
                    field="json",
                    value=f"keys={key_count}",
                    reason=f"JSON has too many keys (max: {max_keys})"
                )
            for value in obj.values():
                count_and_validate(value)
        elif isinstance(obj, list):
            for item in obj:
                count_and_validate(item)
        elif isinstance(obj, str):
            if len(obj) > max_string_length:
                raise ValidationError(
                    field="json_string",
                    value=f"length={len(obj)}",
                    reason=f"String value exceeds maximum length of {max_string_length}"
                )

    count_and_validate(data)
    return True


def sanitize_dict_keys(
    data: Dict[str, Any],
    max_key_length: int = 255,
) -> Dict[str, Any]:
    """
    Sanitize dictionary keys recursively.

    Args:
        data: Dictionary to sanitize
        max_key_length: Maximum key length

    Returns:
        Dictionary with sanitized keys
    """
    result = {}

    for key, value in data.items():
        # Sanitize key
        sanitized_key = sanitize_identifier(
            str(key)[:max_key_length],
            max_length=max_key_length
        )

        # Recursively sanitize nested dicts
        if isinstance(value, dict):
            result[sanitized_key] = sanitize_dict_keys(value, max_key_length)
        elif isinstance(value, list):
            result[sanitized_key] = [
                sanitize_dict_keys(item, max_key_length) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[sanitized_key] = value

    return result


# =============================================================================
# Numeric Validation
# =============================================================================

def validate_number(
    value: Any,
    field: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Union[int, float]:
    """
    Validate and coerce a numeric value.

    Args:
        value: Value to validate
        field: Field name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_nan: Allow NaN values
        allow_inf: Allow infinity values

    Returns:
        Validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    import math

    # Try to coerce to number
    try:
        if isinstance(value, bool):
            raise ValueError("Boolean not allowed")
        num = float(value)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            field=field,
            value=value,
            reason=f"Cannot convert to number: {e}"
        )

    # Check for NaN
    if math.isnan(num) and not allow_nan:
        raise ValidationError(
            field=field,
            value=value,
            reason="NaN values not allowed"
        )

    # Check for infinity
    if math.isinf(num) and not allow_inf:
        raise ValidationError(
            field=field,
            value=value,
            reason="Infinity values not allowed"
        )

    # Check bounds
    if min_value is not None and num < min_value:
        raise ValidationError(
            field=field,
            value=value,
            reason=f"Value below minimum of {min_value}",
            constraint={"min": min_value}
        )

    if max_value is not None and num > max_value:
        raise ValidationError(
            field=field,
            value=value,
            reason=f"Value above maximum of {max_value}",
            constraint={"max": max_value}
        )

    # Return as int if it's a whole number (but not NaN or Inf)
    if not math.isnan(num) and not math.isinf(num) and num == int(num):
        return int(num)

    return num


# =============================================================================
# Payload Validation
# =============================================================================

def validate_payload_size(
    payload: bytes,
    max_size: int = 10 * 1024 * 1024,  # 10 MB default
) -> bool:
    """
    Validate payload size.

    Args:
        payload: Raw payload bytes
        max_size: Maximum allowed size in bytes

    Returns:
        True if size is valid

    Raises:
        ValidationError: If size exceeds limit
    """
    if len(payload) > max_size:
        raise ValidationError(
            field="payload",
            value=f"size={len(payload)}",
            reason=f"Payload size ({len(payload)} bytes) exceeds maximum ({max_size} bytes)"
        )
    return True


def detect_malicious_patterns(
    data: str,
    check_sql: bool = True,
    check_script: bool = True,
    check_path_traversal: bool = True,
) -> Optional[str]:
    """
    Detect potentially malicious patterns in input.

    Args:
        data: String data to check
        check_sql: Check for SQL injection patterns
        check_script: Check for script injection patterns
        check_path_traversal: Check for path traversal patterns

    Returns:
        Description of detected threat, or None if clean
    """
    data_lower = data.lower()

    if check_sql:
        sql_patterns = [
            r"(\bor\b|\band\b)\s+\d+\s*=\s*\d+",  # OR 1=1, AND 1=1
            r"union\s+(all\s+)?select",  # UNION SELECT
            r";\s*drop\s+",  # ; DROP
            r";\s*delete\s+",  # ; DELETE
            r";\s*update\s+.*\s+set\s+",  # ; UPDATE ... SET
            r"exec(\s+|\().*xp_",  # EXEC xp_
            r"--\s*$",  # SQL comment at end
        ]
        for pattern in sql_patterns:
            if re.search(pattern, data_lower):
                return f"SQL injection pattern detected: {pattern}"

    if check_script:
        script_patterns = [
            r"<script",  # Script tag
            r"javascript:",  # JavaScript URL
            r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
            r"eval\s*\(",  # eval()
            r"document\.(cookie|location|write)",  # DOM manipulation
        ]
        for pattern in script_patterns:
            if re.search(pattern, data_lower):
                return f"Script injection pattern detected: {pattern}"

    if check_path_traversal:
        path_patterns = [
            r"\.\./",  # ../
            r"\.\.\\",  # ..\
            r"%2e%2e[/\\]",  # URL encoded
            r"\.\.%2f",  # Mixed encoding
        ]
        for pattern in path_patterns:
            if re.search(pattern, data_lower):
                return f"Path traversal pattern detected: {pattern}"

    return None


# =============================================================================
# Schema Validation Helpers
# =============================================================================

VALID_FIELD_TYPES = {"null", "boolean", "integer", "float", "string", "datetime", "array", "object", "unknown"}
VALID_PROTOCOLS = {"MQTT", "CoAP", "Modbus", "OPC-UA", "HTTP", "WebSocket"}


def validate_field_type(field_type: str) -> bool:
    """Validate that a field type is recognized."""
    return field_type.lower() in {t.lower() for t in VALID_FIELD_TYPES}


def validate_protocol(protocol: str) -> bool:
    """Validate that a protocol is supported."""
    return protocol in VALID_PROTOCOLS


def validate_confidence(value: float, field: str = "confidence") -> float:
    """Validate confidence score is between 0 and 1."""
    return validate_number(value, field, min_value=0.0, max_value=1.0)


def validate_json_payload(
    payload: bytes,
    max_size: int = 10 * 1024 * 1024,
    max_depth: int = 50,
) -> tuple:
    """
    Validate a JSON payload.

    Args:
        payload: Raw bytes payload
        max_size: Maximum payload size in bytes
        max_depth: Maximum nesting depth

    Returns:
        Tuple of (is_valid, error_message)
    """
    import json

    # Check size
    if len(payload) > max_size:
        return False, f"Payload size ({len(payload)}) exceeds maximum ({max_size})"

    # Try to parse JSON
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    # Check depth
    try:
        validate_json_depth(data, max_depth=max_depth)
    except ValidationError as e:
        return False, f"JSON depth exceeded: {str(e)}"

    return True, None


def is_valid_topic(topic: str) -> bool:
    """
    Check if an MQTT topic is valid.

    Args:
        topic: Topic string to validate

    Returns:
        True if valid, False otherwise
    """
    if not topic:
        return False

    # Check for path traversal
    if ".." in topic:
        return False

    # Check for dangerous patterns
    if "<" in topic or ">" in topic:
        return False

    # Check for null bytes
    if "\x00" in topic:
        return False

    # Basic format check - alphanumeric, slashes, wildcards, dots, dashes, underscores
    if not re.match(r'^[a-zA-Z0-9/+#._\-]+$', topic):
        return False

    return True
