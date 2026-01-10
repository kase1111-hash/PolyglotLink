"""
PolyglotLink Custom Exceptions

Centralized exception definitions for consistent error handling across the application.
All exceptions inherit from PolyglotLinkError for unified error handling.
"""

from typing import Any, Dict, Optional


class PolyglotLinkError(Exception):
    """
    Base exception for all PolyglotLink errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error context
        recoverable: Whether the error is recoverable
    """

    def __init__(
        self,
        message: str,
        code: str = "POLYGLOTLINK_ERROR",
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.recoverable = recoverable

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
        }

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} - {self.details}"
        return f"[{self.code}] {self.message}"


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(PolyglotLinkError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        super().__init__(
            message=message,
            code="CONFIG_ERROR",
            details=details,
            recoverable=False,
            **kwargs
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when a required configuration value is missing."""

    def __init__(self, field: str, **kwargs):
        super().__init__(
            message=f"Required configuration '{field}' is missing",
            field=field,
            **kwargs
        )
        self.code = "CONFIG_MISSING"


# =============================================================================
# Protocol Errors
# =============================================================================

class ProtocolError(PolyglotLinkError):
    """Base class for protocol-related errors."""

    def __init__(self, message: str, protocol: str, **kwargs):
        details = kwargs.pop("details", {})
        details["protocol"] = protocol
        super().__init__(
            message=message,
            code="PROTOCOL_ERROR",
            details=details,
            **kwargs
        )


class ConnectionError(ProtocolError):
    """Raised when a protocol connection fails."""

    def __init__(self, protocol: str, host: str, port: int, reason: str = "", **kwargs):
        details = kwargs.pop("details", {})
        details.update({"host": host, "port": port, "reason": reason})
        super().__init__(
            message=f"Failed to connect to {protocol} at {host}:{port}",
            protocol=protocol,
            details=details,
            **kwargs
        )
        self.code = "CONNECTION_FAILED"


class MessageParseError(ProtocolError):
    """Raised when a message cannot be parsed."""

    def __init__(self, protocol: str, reason: str, raw_data: Optional[bytes] = None, **kwargs):
        details = kwargs.pop("details", {})
        details["reason"] = reason
        if raw_data:
            details["raw_data_preview"] = raw_data[:100].hex() if len(raw_data) > 100 else raw_data.hex()
        super().__init__(
            message=f"Failed to parse {protocol} message: {reason}",
            protocol=protocol,
            details=details,
            **kwargs
        )
        self.code = "MESSAGE_PARSE_ERROR"


# =============================================================================
# Schema Errors
# =============================================================================

class SchemaError(PolyglotLinkError):
    """Base class for schema-related errors."""

    def __init__(self, message: str, schema_signature: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if schema_signature:
            details["schema_signature"] = schema_signature
        super().__init__(
            message=message,
            code="SCHEMA_ERROR",
            details=details,
            **kwargs
        )


class UnsupportedEncodingError(SchemaError):
    """Raised when payload encoding is not supported."""

    def __init__(self, encoding: str, **kwargs):
        details = kwargs.pop("details", {})
        details["encoding"] = encoding
        super().__init__(
            message=f"Unsupported payload encoding: {encoding}",
            details=details,
            **kwargs
        )
        self.code = "UNSUPPORTED_ENCODING"


class SchemaExtractionError(SchemaError):
    """Raised when schema extraction fails."""

    def __init__(self, reason: str, **kwargs):
        super().__init__(
            message=f"Schema extraction failed: {reason}",
            **kwargs
        )
        self.code = "SCHEMA_EXTRACTION_FAILED"


# =============================================================================
# Translation Errors
# =============================================================================

class TranslationError(PolyglotLinkError):
    """Base class for semantic translation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="TRANSLATION_ERROR",
            **kwargs
        )


class LLMError(TranslationError):
    """Raised when LLM call fails."""

    def __init__(self, reason: str, model: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if model:
            details["model"] = model
        details["reason"] = reason
        super().__init__(
            message=f"LLM translation failed: {reason}",
            details=details,
            **kwargs
        )
        self.code = "LLM_ERROR"


class EmbeddingError(TranslationError):
    """Raised when embedding generation or lookup fails."""

    def __init__(self, reason: str, **kwargs):
        details = kwargs.pop("details", {})
        details["reason"] = reason
        super().__init__(
            message=f"Embedding operation failed: {reason}",
            details=details,
            **kwargs
        )
        self.code = "EMBEDDING_ERROR"


class MappingNotFoundError(TranslationError):
    """Raised when no mapping can be found for a field."""

    def __init__(self, field: str, **kwargs):
        details = kwargs.pop("details", {})
        details["field"] = field
        super().__init__(
            message=f"No mapping found for field: {field}",
            details=details,
            **kwargs
        )
        self.code = "MAPPING_NOT_FOUND"


# =============================================================================
# Normalization Errors
# =============================================================================

class NormalizationError(PolyglotLinkError):
    """Base class for normalization errors."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        super().__init__(
            message=message,
            code="NORMALIZATION_ERROR",
            details=details,
            **kwargs
        )


class ConversionError(NormalizationError):
    """Raised when unit conversion fails."""

    def __init__(
        self,
        field: str,
        from_unit: str,
        to_unit: str,
        reason: str = "",
        **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "from_unit": from_unit,
            "to_unit": to_unit,
            "reason": reason
        })
        super().__init__(
            message=f"Unit conversion failed for '{field}': {from_unit} -> {to_unit}",
            field=field,
            details=details,
            **kwargs
        )
        self.code = "CONVERSION_ERROR"


class UnsafeFormulaError(NormalizationError):
    """Raised when a conversion formula contains unsafe operations."""

    def __init__(self, formula: str, **kwargs):
        details = kwargs.pop("details", {})
        details["formula"] = formula
        super().__init__(
            message=f"Unsafe conversion formula detected: {formula}",
            details=details,
            recoverable=False,
            **kwargs
        )
        self.code = "UNSAFE_FORMULA"


class ValidationError(NormalizationError):
    """Raised when value validation fails."""

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        constraint: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "value": str(value),
            "reason": reason
        })
        if constraint:
            details["constraint"] = constraint
        super().__init__(
            message=f"Validation failed for '{field}': {reason}",
            field=field,
            details=details,
            **kwargs
        )
        self.code = "VALIDATION_ERROR"


class TypeCoercionError(NormalizationError):
    """Raised when type coercion fails."""

    def __init__(
        self,
        field: str,
        value: Any,
        target_type: str,
        **kwargs
    ):
        details = kwargs.pop("details", {})
        details.update({
            "value": str(value),
            "source_type": type(value).__name__,
            "target_type": target_type
        })
        super().__init__(
            message=f"Cannot convert '{field}' from {type(value).__name__} to {target_type}",
            field=field,
            details=details,
            **kwargs
        )
        self.code = "TYPE_COERCION_ERROR"


# =============================================================================
# Output Errors
# =============================================================================

class OutputError(PolyglotLinkError):
    """Base class for output/publishing errors."""

    def __init__(self, message: str, output_type: str, **kwargs):
        details = kwargs.pop("details", {})
        details["output_type"] = output_type
        super().__init__(
            message=message,
            code="OUTPUT_ERROR",
            details=details,
            **kwargs
        )


class PublishError(OutputError):
    """Raised when publishing a message fails."""

    def __init__(self, output_type: str, destination: str, reason: str, **kwargs):
        details = kwargs.pop("details", {})
        details.update({
            "destination": destination,
            "reason": reason
        })
        super().__init__(
            message=f"Failed to publish to {output_type}: {reason}",
            output_type=output_type,
            details=details,
            **kwargs
        )
        self.code = "PUBLISH_ERROR"


# =============================================================================
# Storage Errors
# =============================================================================

class StorageError(PolyglotLinkError):
    """Base class for storage-related errors."""

    def __init__(self, message: str, storage_type: str, **kwargs):
        details = kwargs.pop("details", {})
        details["storage_type"] = storage_type
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            details=details,
            **kwargs
        )


class CacheError(StorageError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, reason: str, **kwargs):
        details = kwargs.pop("details", {})
        details.update({
            "operation": operation,
            "reason": reason
        })
        super().__init__(
            message=f"Cache {operation} failed: {reason}",
            storage_type="cache",
            details=details,
            **kwargs
        )
        self.code = "CACHE_ERROR"


class DatabaseError(StorageError):
    """Raised when database operations fail."""

    def __init__(self, database: str, operation: str, reason: str, **kwargs):
        details = kwargs.pop("details", {})
        details.update({
            "operation": operation,
            "reason": reason
        })
        super().__init__(
            message=f"Database operation '{operation}' failed on {database}: {reason}",
            storage_type=database,
            details=details,
            **kwargs
        )
        self.code = "DATABASE_ERROR"


# =============================================================================
# Ontology Errors
# =============================================================================

class OntologyError(PolyglotLinkError):
    """Base class for ontology-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="ONTOLOGY_ERROR",
            **kwargs
        )


class ConceptNotFoundError(OntologyError):
    """Raised when an ontology concept is not found."""

    def __init__(self, concept_id: str, **kwargs):
        details = kwargs.pop("details", {})
        details["concept_id"] = concept_id
        super().__init__(
            message=f"Ontology concept not found: {concept_id}",
            details=details,
            **kwargs
        )
        self.code = "CONCEPT_NOT_FOUND"


class ConceptExistsError(OntologyError):
    """Raised when trying to create a concept that already exists."""

    def __init__(self, concept_id: str, **kwargs):
        details = kwargs.pop("details", {})
        details["concept_id"] = concept_id
        super().__init__(
            message=f"Ontology concept already exists: {concept_id}",
            details=details,
            **kwargs
        )
        self.code = "CONCEPT_EXISTS"
