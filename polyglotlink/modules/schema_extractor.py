"""
PolyglotLink Schema Extractor Module

This module extracts structured schemas from raw IoT payloads,
detecting field types, units, and semantic hints.
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import structlog

from polyglotlink.models.schemas import (
    CachedMapping,
    ExtractedField,
    ExtractedSchema,
    MappingSource,
    PayloadEncoding,
    RawMessage,
    SchemaExtractorConfig,
)
from polyglotlink.modules.protocol_listener import ENCODING_PARSERS

logger = structlog.get_logger(__name__)


# ============================================================================
# Unit Pattern Detection
# ============================================================================

UNIT_PATTERNS: Dict[str, str] = {
    # Temperature
    r"(temp|temperature).*(_c|_celsius|_centigrade)$": "celsius",
    r"(temp|temperature).*(_f|_fahrenheit)$": "fahrenheit",
    r"(temp|temperature).*(_k|_kelvin)$": "kelvin",
    r"^(tmp|temp|t)$": "celsius",  # Default assumption

    # Humidity
    r"(hum|humidity|rh).*(_pct|_percent|%)?$": "percent",

    # Pressure
    r"(press|pressure).*(_pa|_pascal)$": "pascal",
    r"(press|pressure).*(_bar)$": "bar",
    r"(press|pressure).*(_psi)$": "psi",
    r"(press|pressure).*(_hpa)$": "hectopascal",

    # Voltage/Current
    r"(volt|voltage|v)$": "volt",
    r"(current|amp|ampere|i)$": "ampere",
    r"(power|watt|w)$": "watt",

    # Speed/Flow
    r"(speed|velocity).*(_mps|_ms)$": "meters_per_second",
    r"(speed|velocity).*(_kmh|_kph)$": "kilometers_per_hour",
    r"(flow).*(_lpm|_lm)$": "liters_per_minute",

    # Distance/Length
    r"(dist|distance|length).*(_m|_meter)$": "meter",
    r"(dist|distance|length).*(_cm)$": "centimeter",
    r"(dist|distance|length).*(_mm)$": "millimeter",

    # Time
    r"(time|duration).*(_s|_sec|_seconds)$": "seconds",
    r"(time|duration).*(_ms|_milliseconds)$": "milliseconds",

    # Mass
    r"(weight|mass).*(_kg|_kilogram)$": "kilogram",
    r"(weight|mass).*(_g|_gram)$": "gram",

    # Percentage
    r".*(_pct|_percent|%)$": "percent",
    r"^(pct|percent|ratio)$": "percent"
}


# ============================================================================
# Semantic Hint Detection
# ============================================================================

SEMANTIC_HINTS: Dict[str, str] = {
    # Environment
    r"(temp|temperature|tmp)": "temperature",
    r"(hum|humidity|rh)": "humidity",
    r"(press|pressure|baro)": "pressure",
    r"(light|lux|luminosity)": "illuminance",
    r"(co2|carbon_dioxide)": "co2_concentration",
    r"(pm25|pm2\.5)": "particulate_matter_2_5",
    r"(noise|sound|db|decibel)": "noise_level",

    # Motion/Position
    r"(accel|acceleration)": "acceleration",
    r"(gyro|angular)": "angular_velocity",
    r"(lat|latitude)": "latitude",
    r"(lon|lng|longitude)": "longitude",
    r"(alt|altitude|elevation)": "altitude",
    r"(speed|velocity)": "speed",

    # Electrical
    r"(volt|voltage)": "voltage",
    r"(current|ampere|amp)": "current",
    r"(power|watt)": "power",
    r"(energy|kwh)": "energy",
    r"(freq|frequency|hz)": "frequency",

    # State/Status
    r"(state|status)": "state",
    r"(online|connected|alive)": "connectivity",
    r"(battery|batt)": "battery_level",
    r"(rssi|signal)": "signal_strength",

    # Identifiers
    r"(id|uuid|guid)$": "identifier",
    r"(name|label)$": "name",
    r"(timestamp|time|ts|datetime)": "timestamp"
}


# ============================================================================
# Type Detection
# ============================================================================

def is_iso_datetime(value: str) -> bool:
    """Check if string is ISO datetime format."""
    patterns = [
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO 8601
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # Common format
        r"^\d{4}/\d{2}/\d{2}",  # Date only
    ]
    return any(re.match(p, value) for p in patterns)


def is_numeric_string(value: str) -> bool:
    """Check if string represents a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def detect_type(value: Any) -> str:
    """Detect the type of a value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        if is_iso_datetime(value):
            return "datetime"
        if is_numeric_string(value):
            return "numeric_string"
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def infer_unit_from_key(key: str) -> Optional[str]:
    """Infer unit from field name using patterns."""
    key_lower = key.lower()

    for pattern, unit in UNIT_PATTERNS.items():
        if re.match(pattern, key_lower):
            return unit

    return None


def infer_semantic_hint(key: str, value: Any) -> Optional[str]:
    """Infer semantic category from field name and value."""
    key_lower = key.lower()

    for pattern, semantic in SEMANTIC_HINTS.items():
        if re.search(pattern, key_lower):
            return semantic

    return None


def is_timestamp_field(key: str, value: Any) -> bool:
    """Check if field is likely a timestamp."""
    key_lower = key.lower()
    timestamp_keywords = ['timestamp', 'time', 'ts', 'datetime', 'created_at', 'updated_at', 'date']

    if any(kw in key_lower for kw in timestamp_keywords):
        return True

    if isinstance(value, str) and is_iso_datetime(value):
        return True

    # Unix timestamp detection (10 or 13 digits)
    if isinstance(value, (int, float)):
        if 1_000_000_000 < value < 10_000_000_000:  # Seconds
            return True
        if 1_000_000_000_000 < value < 10_000_000_000_000:  # Milliseconds
            return True

    return False


def is_identifier_field(key: str, value: Any) -> bool:
    """Check if field is likely an identifier."""
    key_lower = key.lower()
    id_keywords = ['id', 'uuid', 'guid', 'key', 'code', 'serial', 'mac', 'imei']

    if any(key_lower.endswith(kw) or key_lower.startswith(kw) for kw in id_keywords):
        return True

    # UUID pattern check
    if isinstance(value, str):
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if re.match(uuid_pattern, value, re.IGNORECASE):
            return True

    return False


# ============================================================================
# Flattening Utilities
# ============================================================================

def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
    max_depth: int = 10,
    current_depth: int = 0
) -> Dict[str, Any]:
    """
    Flatten nested dictionary into dot-notation keys.
    {"a": {"b": 1}} -> {"a.b": 1}
    """
    if current_depth >= max_depth:
        return {parent_key: d} if parent_key else d

    items: List[tuple] = []

    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(
                flatten_dict(
                    value, new_key, separator, max_depth, current_depth + 1
                ).items()
            )
        elif isinstance(value, list):
            # Handle arrays
            if len(value) > 0 and isinstance(value[0], dict):
                # Array of objects - flatten first element as template
                items.extend(
                    flatten_dict(
                        value[0], f"{new_key}[0]", separator, max_depth, current_depth + 1
                    ).items()
                )
                items.append((f"{new_key}._count", len(value)))
            else:
                items.append((new_key, value))
        else:
            items.append((new_key, value))

    return dict(items)


# ============================================================================
# Schema Hashing and Caching
# ============================================================================

def generate_schema_hash(fields: List[ExtractedField]) -> str:
    """
    Generate a fingerprint for the schema based on field names and types.
    Used for caching semantic mappings.
    """
    # Create canonical representation
    canonical = sorted([
        f"{f.key}:{f.value_type}"
        for f in fields
        if not f.is_timestamp and not f.is_identifier
    ])

    schema_string = "|".join(canonical)
    return hashlib.sha256(schema_string.encode()).hexdigest()[:16]


class SchemaCache:
    """In-memory schema cache with optional Redis backing."""

    def __init__(self, ttl_days: int = 30, redis_client=None):
        self._local_cache: Dict[str, CachedMapping] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
        self._ttl = timedelta(days=ttl_days)
        self._redis = redis_client

    def get(self, schema_signature: str) -> Optional[CachedMapping]:
        """Check if we've seen this schema before."""
        cache_key = f"schema:{schema_signature}"

        # Check local cache first
        if cache_key in self._local_cache:
            mapping = self._local_cache[cache_key]
            # Check if expired
            if datetime.utcnow() - mapping.created_at < self._ttl:
                self._update_stats(schema_signature)
                return mapping
            else:
                del self._local_cache[cache_key]

        # Try Redis if available
        if self._redis:
            try:
                cached = self._redis.get(cache_key)
                if cached:
                    mapping = CachedMapping.model_validate_json(cached)
                    self._local_cache[cache_key] = mapping
                    self._update_stats(schema_signature)
                    return mapping
            except Exception as e:
                logger.warning("Redis cache lookup failed", error=str(e))

        return None

    def set(self, schema_signature: str, mapping: CachedMapping) -> None:
        """Store a schema mapping."""
        cache_key = f"schema:{schema_signature}"
        self._local_cache[cache_key] = mapping

        # Persist to Redis if available
        if self._redis:
            try:
                self._redis.setex(
                    cache_key,
                    self._ttl,
                    mapping.model_dump_json()
                )
            except Exception as e:
                logger.warning("Redis cache write failed", error=str(e))

    def _update_stats(self, schema_signature: str) -> None:
        """Track cache statistics."""
        if schema_signature not in self._stats:
            self._stats[schema_signature] = {'hits': 0}
        self._stats[schema_signature]['hits'] += 1


# ============================================================================
# Schema Extractor
# ============================================================================

class UnsupportedEncodingError(Exception):
    """Raised when payload encoding is not supported."""
    pass


class SchemaExtractor:
    """
    Extracts schema information from raw IoT messages.
    """

    def __init__(
        self,
        config: Optional[SchemaExtractorConfig] = None,
        cache: Optional[SchemaCache] = None
    ):
        self.config = config or SchemaExtractorConfig()
        self.cache = cache or SchemaCache(ttl_days=self.config.cache_ttl_days)

    def extract_schema(self, raw: RawMessage) -> ExtractedSchema:
        """
        Extract schema from a raw message.
        """
        # Decode payload based on detected encoding
        parser = ENCODING_PARSERS.get(raw.payload_encoding)
        if not parser:
            raise UnsupportedEncodingError(f"Unsupported encoding: {raw.payload_encoding}")

        try:
            decoded = parser(raw.payload_raw)
        except Exception as e:
            logger.error(
                "Failed to decode payload",
                encoding=raw.payload_encoding,
                error=str(e)
            )
            decoded = {'_raw': raw.payload_raw.hex()}

        # Ensure decoded is a dict
        if not isinstance(decoded, dict):
            decoded = {'value': decoded}

        # Flatten nested structures
        flat_fields = flatten_dict(
            decoded,
            separator=".",
            max_depth=self.config.max_nesting_depth
        )

        # Extract field information
        fields: List[ExtractedField] = []
        for key, value in flat_fields.items():
            # Skip null fields if configured
            if value is None and not self.config.preserve_null_fields:
                continue

            inferred_unit = None
            inferred_semantic = None

            if self.config.enable_unit_inference:
                inferred_unit = infer_unit_from_key(key)

            if self.config.enable_semantic_hints:
                inferred_semantic = infer_semantic_hint(key, value)

            field = ExtractedField(
                key=key,
                original_key=key,
                value=value,
                value_type=detect_type(value),
                inferred_unit=inferred_unit,
                inferred_semantic=inferred_semantic,
                is_timestamp=is_timestamp_field(key, value),
                is_identifier=is_identifier_field(key, value)
            )
            fields.append(field)

        # Generate schema fingerprint
        schema_signature = generate_schema_hash(fields)

        # Check cache for known mapping
        cached_mapping = self.cache.get(schema_signature)

        return ExtractedSchema(
            message_id=raw.message_id,
            device_id=raw.device_id,
            protocol=raw.protocol,
            topic=raw.topic,
            fields=fields,
            schema_signature=schema_signature,
            cached_mapping=cached_mapping,
            payload_decoded=decoded,
            extracted_at=datetime.utcnow()
        )

    def cache_mapping(
        self,
        schema_signature: str,
        field_mappings: List,
        confidence: float,
        source: MappingSource = MappingSource.LLM
    ) -> None:
        """Store a learned schema mapping for future reuse."""
        cached = CachedMapping(
            schema_signature=schema_signature,
            field_mappings=field_mappings,
            confidence=confidence,
            created_at=datetime.utcnow(),
            source=source,
            hit_count=0
        )
        self.cache.set(schema_signature, cached)
        logger.info(
            "Cached new schema mapping",
            signature=schema_signature,
            source=source.value
        )

    def get_field_summary(self, schema: ExtractedSchema) -> str:
        """Generate a human-readable summary of extracted fields."""
        lines = []
        for field in schema.fields:
            line = f"  - {field.key}: {field.value_type}"
            if field.inferred_unit:
                line += f" ({field.inferred_unit})"
            if field.inferred_semantic:
                line += f" -> {field.inferred_semantic}"
            lines.append(line)

        return f"Schema ({schema.schema_signature}):\n" + "\n".join(lines)
