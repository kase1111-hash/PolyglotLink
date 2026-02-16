"""
PolyglotLink Normalization Engine Module

This module normalizes semantic mappings into standardized messages,
handling unit conversions, type enforcement, and value validation.
"""

import ast
import operator
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from polyglotlink.models.schemas import (
    ConversionRecord,
    ExtractedSchema,
    NormalizationConfig,
    NormalizedMessage,
    SemanticMapping,
    ValidationError,
    ValidationErrorType,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Unit Conversion Registry
# ============================================================================

UNIT_CONVERSIONS: dict[str, dict[str, str]] = {
    # Temperature
    "celsius": {"fahrenheit": "(value * 9/5) + 32", "kelvin": "value + 273.15"},
    "fahrenheit": {"celsius": "(value - 32) * 5/9", "kelvin": "(value - 32) * 5/9 + 273.15"},
    "kelvin": {"celsius": "value - 273.15", "fahrenheit": "(value - 273.15) * 9/5 + 32"},
    # Pressure
    "pascal": {"bar": "value / 100000", "psi": "value * 0.000145038", "hectopascal": "value / 100"},
    "bar": {"pascal": "value * 100000", "psi": "value * 14.5038"},
    "psi": {"pascal": "value / 0.000145038", "bar": "value / 14.5038"},
    "hectopascal": {"pascal": "value * 100"},
    # Speed
    "meters_per_second": {
        "kilometers_per_hour": "value * 3.6",
        "miles_per_hour": "value * 2.23694",
    },
    "kilometers_per_hour": {
        "meters_per_second": "value / 3.6",
        "miles_per_hour": "value * 0.621371",
    },
    "miles_per_hour": {
        "meters_per_second": "value / 2.23694",
        "kilometers_per_hour": "value / 0.621371",
    },
    # Length
    "meter": {
        "centimeter": "value * 100",
        "millimeter": "value * 1000",
        "kilometer": "value / 1000",
        "feet": "value * 3.28084",
    },
    "centimeter": {"meter": "value / 100", "millimeter": "value * 10"},
    "millimeter": {"meter": "value / 1000", "centimeter": "value / 10"},
    # Mass
    "kilogram": {"gram": "value * 1000", "pound": "value * 2.20462"},
    "gram": {"kilogram": "value / 1000"},
    # Volume
    "liter": {"milliliter": "value * 1000", "gallon": "value * 0.264172"},
    "milliliter": {"liter": "value / 1000"},
    # Percentage/Ratio
    "percent": {"ratio": "value / 100"},
    "ratio": {"percent": "value * 100"},
    # Time
    "seconds": {"milliseconds": "value * 1000", "minutes": "value / 60"},
    "milliseconds": {"seconds": "value / 1000"},
    "minutes": {"seconds": "value * 60"},
}


def get_unit_conversion(from_unit: str, to_unit: str) -> str | None:
    """Get conversion formula between units."""
    if from_unit == to_unit:
        return None

    if from_unit in UNIT_CONVERSIONS and to_unit in UNIT_CONVERSIONS[from_unit]:
        return UNIT_CONVERSIONS[from_unit][to_unit]

    # Try transitive conversion (limited to 2 hops)
    if from_unit in UNIT_CONVERSIONS:
        for intermediate, formula1 in UNIT_CONVERSIONS[from_unit].items():
            if intermediate in UNIT_CONVERSIONS and to_unit in UNIT_CONVERSIONS[intermediate]:
                formula2 = UNIT_CONVERSIONS[intermediate][to_unit]
                # Compose formulas
                return f"({formula2.replace('value', f'({formula1})')})"

    return None


# ============================================================================
# Safe Formula Execution
# ============================================================================


class UnsafeFormulaError(Exception):
    """Raised when a formula contains unsafe operations."""

    pass


class ConversionError(Exception):
    """Raised when unit conversion fails."""

    pass


class SafeExpressionEvaluator:
    """
    Safe expression evaluator using AST that only allows basic arithmetic.
    Replaces unsafe eval() with a restricted evaluator.
    """

    # Supported operators for safe evaluation
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def __init__(self, variables: dict[str, float]):
        self.variables = variables

    def evaluate(self, expression: str) -> float:
        """Safely evaluate an arithmetic expression."""
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except (SyntaxError, TypeError, KeyError) as e:
            raise ConversionError(f"Invalid expression: {e}")

    def _eval_node(self, node: ast.expr) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses ast.Constant for numbers
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise UnsafeFormulaError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility (deprecated but may still appear)
            return float(node.n)

        elif isinstance(node, ast.Name):
            # Variable reference (e.g., 'value')
            if node.id in self.variables:
                return self.variables[node.id]
            raise UnsafeFormulaError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.BinOp):
            # Binary operations (+, -, *, /)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise UnsafeFormulaError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            # Unary operations (+, -)
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise UnsafeFormulaError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.OPERATORS[op_type](operand)

        elif isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        else:
            raise UnsafeFormulaError(f"Unsupported expression type: {type(node).__name__}")


def apply_conversion(value: Any, formula: str) -> float | None:
    """
    Safely execute unit conversion formula using AST-based evaluation.
    Only allows basic arithmetic operations (+, -, *, /) and the 'value' variable.
    Returns None when value is None.
    """
    if value is None:
        return None

    # Validate formula (only allow safe characters)
    allowed_tokens = set("value0123456789.+-*/() ")
    if not all(c in allowed_tokens for c in formula):
        raise UnsafeFormulaError(f"Unsafe characters in formula: {formula}")

    # Additional validation: no function calls or assignments
    if re.search(r"[a-z_][a-z0-9_]*\s*\(", formula.replace("value", "")):
        raise UnsafeFormulaError(f"Function calls not allowed in formula: {formula}")

    try:
        evaluator = SafeExpressionEvaluator({"value": float(value)})
        result = evaluator.evaluate(formula)
        return round(result, 6)  # Limit precision
    except (UnsafeFormulaError, ConversionError):
        raise
    except Exception as e:
        raise ConversionError(f"Conversion failed: {e}")


# ============================================================================
# Type Enforcement
# ============================================================================


def enforce_type(value: Any, target_type: str) -> Any:
    """Cast value to expected type."""
    if value is None:
        return None

    try:
        if target_type == "float" or target_type == "numeric_string":
            return float(value)
        elif target_type == "integer":
            return int(float(value))
        elif target_type == "string":
            return str(value)
        elif target_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif target_type == "datetime":
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, (int, float)):
                # Unix timestamp
                if value > 1e12:
                    value = value / 1000  # Milliseconds to seconds
                return datetime.fromtimestamp(value, tz=timezone.utc)
        return value
    except (ValueError, TypeError, OSError) as e:
        raise TypeError(f"Cannot convert {type(value).__name__} to {target_type}: {e}")


# ============================================================================
# Value Validation
# ============================================================================


class Concept:
    """Simplified concept for validation (when ontology not available)."""

    def __init__(
        self,
        concept_id: str,
        datatype: str = "float",
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        self.concept_id = concept_id
        self.datatype = datatype
        self.min_value = min_value
        self.max_value = max_value


# Default validation constraints
DEFAULT_CONSTRAINTS: dict[str, Concept] = {
    "temperature_celsius": Concept("temperature_celsius", "float", -273.15, 1000),
    "humidity_percent": Concept("humidity_percent", "float", 0, 100),
    "battery_percent": Concept("battery_percent", "float", 0, 100),
    "latitude_degrees": Concept("latitude_degrees", "float", -90, 90),
    "longitude_degrees": Concept("longitude_degrees", "float", -180, 180),
    "pressure_pascal": Concept("pressure_pascal", "float", 0, 1e9),
    "voltage_volt": Concept("voltage_volt", "float", -1e6, 1e6),
    "current_ampere": Concept("current_ampere", "float", -1e6, 1e6),
    "power_watt": Concept("power_watt", "float", 0, 1e9),
}


def validate_value(value: Any, concept: Concept) -> bool:
    """Check value against ontology constraints."""
    if value is None:
        return True  # Nulls handled separately

    if concept.min_value is not None and value < concept.min_value:
        return False

    return not (concept.max_value is not None and value > concept.max_value)


# ============================================================================
# Timestamp Extraction
# ============================================================================


def extract_timestamp(
    schema: ExtractedSchema, timestamp_fields: list[str] = None
) -> datetime | None:
    """Extract timestamp from schema fields."""
    if timestamp_fields is None:
        timestamp_fields = ["timestamp", "ts", "time", "datetime", "created_at"]

    for field in schema.fields:
        if field.is_timestamp:
            try:
                return enforce_type(field.value, "datetime")
            except (TypeError, ValueError):
                continue

    # Check by field name
    for field in schema.fields:
        if any(tf in field.key.lower() for tf in timestamp_fields):
            try:
                return enforce_type(field.value, "datetime")
            except (TypeError, ValueError):
                continue

    return None


# ============================================================================
# Metadata Enrichment
# ============================================================================


class DeviceInfo:
    """Device information for metadata enrichment."""

    def __init__(
        self,
        device_id: str,
        device_type: str | None = None,
        name: str | None = None,
        location: str | None = None,
        tags: list[str] | None = None,
    ):
        self.device_id = device_id
        self.type = device_type
        self.name = name
        self.location = location
        self.tags = tags or []


class DeviceRegistry:
    """Simple in-memory device registry."""

    def __init__(self):
        self._devices: dict[str, DeviceInfo] = {}

    def register(self, device: DeviceInfo) -> None:
        """Register a device."""
        self._devices[device.device_id] = device

    def get(self, device_id: str) -> DeviceInfo | None:
        """Get device info."""
        return self._devices.get(device_id)

    def update(self, device_id: str, **kwargs) -> None:
        """Update device info."""
        if device_id in self._devices:
            device = self._devices[device_id]
            for key, value in kwargs.items():
                if hasattr(device, key):
                    setattr(device, key, value)


def enrich_metadata(
    schema: ExtractedSchema, mapping: SemanticMapping, device_registry: DeviceRegistry | None = None
) -> dict[str, Any]:
    """Add contextual metadata to the normalized message."""
    metadata = {
        "source_protocol": schema.protocol.value,
        "source_topic": schema.topic,
        "translation_confidence": mapping.confidence,
        "schema_signature": schema.schema_signature,
    }

    # Add device registry info if available
    if device_registry:
        device_info = device_registry.get(schema.device_id)
        if device_info:
            metadata["device_type"] = device_info.type
            metadata["device_name"] = device_info.name
            metadata["location"] = device_info.location
            metadata["tags"] = device_info.tags

    # Add inferred context
    if mapping.device_context:
        metadata["inferred_context"] = mapping.device_context

    # Add LLM flag
    if mapping.llm_generated:
        metadata["llm_assisted"] = True

    return metadata


# ============================================================================
# Normalization Engine
# ============================================================================


class NormalizationEngine:
    """
    Normalizes semantic mappings into standardized messages.
    """

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        ontology_registry=None,
        device_registry: DeviceRegistry | None = None,
    ):
        self.config = config or NormalizationConfig()
        self.ontology_registry = ontology_registry
        self.device_registry = device_registry or DeviceRegistry()

    def normalize_message(
        self, schema: ExtractedSchema, mapping: SemanticMapping
    ) -> NormalizedMessage:
        """
        Apply semantic mapping to raw values and produce normalized message.
        """
        normalized_fields: dict[str, Any] = {}
        validation_errors: list[ValidationError] = []
        conversions_applied: list[ConversionRecord] = []

        # Build lookup from source field to mapping
        mapping_lookup = {m.source_field: m for m in mapping.field_mappings}

        for field in schema.fields:
            field_mapping = mapping_lookup.get(field.key)

            if not field_mapping:
                # Passthrough unmapped fields with prefix if configured
                if self.config.include_unmapped:
                    normalized_fields[f"_unmapped.{field.key}"] = field.value
                continue

            value = field.value
            target_field = field_mapping.target_field

            # Handle null values
            if value is None:
                if self.config.null_strategy == "omit":
                    continue
                normalized_fields[target_field] = self._handle_null(field_mapping.target_concept)
                continue

            # Step 1: Unit conversion
            if field_mapping.conversion_formula:
                try:
                    original_value = value
                    value = apply_conversion(value, field_mapping.conversion_formula)
                    conversions_applied.append(
                        ConversionRecord(
                            field=field.key,
                            from_unit=field_mapping.source_unit or "unknown",
                            to_unit=field_mapping.target_unit or "unknown",
                            original_value=original_value,
                            converted_value=value,
                        )
                    )
                except (ConversionError, UnsafeFormulaError) as e:
                    validation_errors.append(
                        ValidationError(
                            field=field.key,
                            error=ValidationErrorType.CONVERSION_FAILED,
                            details=str(e),
                        )
                    )
                    if self.config.include_invalid:
                        normalized_fields[f"_error.{target_field}"] = field.value
                    continue
            elif field_mapping.source_unit and field_mapping.target_unit:
                # Try to find conversion formula
                formula = get_unit_conversion(field_mapping.source_unit, field_mapping.target_unit)
                if formula:
                    try:
                        original_value = value
                        value = apply_conversion(value, formula)
                        conversions_applied.append(
                            ConversionRecord(
                                field=field.key,
                                from_unit=field_mapping.source_unit,
                                to_unit=field_mapping.target_unit,
                                original_value=original_value,
                                converted_value=value,
                            )
                        )
                    except (ConversionError, UnsafeFormulaError) as e:
                        logger.warning("Auto-conversion failed", field=field.key, error=str(e))

            # Step 2: Type enforcement
            concept = self._get_concept(field_mapping.target_concept)
            if concept:
                try:
                    value = enforce_type(value, concept.datatype)
                except TypeError as e:
                    validation_errors.append(
                        ValidationError(
                            field=field.key,
                            error=ValidationErrorType.TYPE_MISMATCH,
                            expected=concept.datatype,
                            actual=type(field.value).__name__,
                            details=str(e),
                        )
                    )
                    if self.config.include_invalid:
                        normalized_fields[f"_error.{target_field}"] = field.value
                    continue

                # Step 3: Value validation
                if not validate_value(value, concept):
                    validation_errors.append(
                        ValidationError(
                            field=field.key,
                            error=ValidationErrorType.OUT_OF_RANGE,
                            value=value,
                            min=concept.min_value,
                            max=concept.max_value,
                        )
                    )
                    if self.config.include_invalid:
                        normalized_fields[f"_invalid.{target_field}"] = value
                    continue

            # Apply precision
            if isinstance(value, float):
                value = self._apply_precision(value, target_field)

            normalized_fields[target_field] = value

        # Step 4: Metadata enrichment
        metadata = enrich_metadata(schema, mapping, self.device_registry)

        # Extract timestamp
        timestamp = extract_timestamp(schema, self.config.timestamp_field_names)

        return NormalizedMessage(
            message_id=schema.message_id,
            device_id=schema.device_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            data=normalized_fields,
            metadata=metadata,
            context=mapping.device_context,
            schema_signature=schema.schema_signature,
            confidence=mapping.confidence,
            conversions=conversions_applied,
            validation_errors=validation_errors,
            normalized_at=datetime.now(timezone.utc),
        )

    def _get_concept(self, concept_id: str) -> Concept | None:
        """Get concept from registry or defaults."""
        if self.ontology_registry:
            concept = self.ontology_registry.get_concept(concept_id)
            if concept:
                return concept

        # Check default constraints
        return DEFAULT_CONSTRAINTS.get(concept_id)

    def _handle_null(self, target_concept: str) -> Any:
        """Handle null values according to configuration.

        Note: The "omit" strategy is handled by the caller (normalize_message)
        via an early ``continue`` before this method is reached.  Only
        "preserve" and "default" should reach here.
        """
        if self.config.null_strategy == "preserve":
            return None
        elif self.config.null_strategy == "default":
            concept = self._get_concept(target_concept)
            if concept:
                return self.config.default_values.get(concept.datatype)
            return None

        return None

    def _apply_precision(self, value: float, field_name: str) -> float:
        """Apply precision based on field type."""
        # Check specific precisions first
        for pattern, precision in self.config.precision.items():
            if pattern in field_name.lower():
                return round(value, precision)

        # Default float precision
        default_precision = self.config.precision.get("float", 6)
        return round(value, default_precision)

    def register_device(
        self,
        device_id: str,
        device_type: str | None = None,
        name: str | None = None,
        location: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register a device for metadata enrichment."""
        device = DeviceInfo(
            device_id=device_id, device_type=device_type, name=name, location=location, tags=tags
        )
        self.device_registry.register(device)
        logger.info("Device registered", device_id=device_id)
