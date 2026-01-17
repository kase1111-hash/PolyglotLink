"""
PolyglotLink Utilities

Common utilities for configuration, logging, validation, error handling,
secrets management, and error tracking.
"""

from polyglotlink.utils.config import (
    Settings,
    get_settings,
    reload_settings,
)
from polyglotlink.utils.error_logging import (
    add_breadcrumb,
    capture_errors,
    capture_exception,
    capture_message,
    error_context,
    init_sentry,
    set_user,
)
from polyglotlink.utils.error_logging import (
    flush as flush_errors,
)
from polyglotlink.utils.exceptions import (
    CacheError,
    ConceptExistsError,
    ConceptNotFoundError,
    ConfigurationError,
    ConversionError,
    DatabaseError,
    EmbeddingError,
    LLMError,
    MappingNotFoundError,
    MessageParseError,
    MissingConfigurationError,
    NormalizationError,
    OntologyError,
    OutputError,
    PolyglotLinkError,
    ProtocolError,
    PublishError,
    SchemaError,
    SchemaExtractionError,
    StorageError,
    TranslationError,
    TypeCoercionError,
    UnsafeFormulaError,
    UnsupportedEncodingError,
    ValidationError,
)
from polyglotlink.utils.logging import (
    LogContext,
    MetricsLogger,
    configure_logging,
    get_logger,
    log_performance,
)
from polyglotlink.utils.metrics import (
    PolyglotLinkMetrics,
    get_metrics,
    metrics,
)
from polyglotlink.utils.secrets import (
    SecretsManager,
    get_secret,
    get_secrets_manager,
    require_secret,
)
from polyglotlink.utils.validation import (
    detect_malicious_patterns,
    is_valid_topic,
    sanitize_dict_keys,
    sanitize_identifier,
    sanitize_string,
    sanitize_topic,
    validate_confidence,
    validate_field_type,
    validate_json_depth,
    validate_json_payload,
    validate_json_size,
    validate_number,
    validate_payload_size,
    validate_protocol,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "reload_settings",
    # Error Logging
    "add_breadcrumb",
    "capture_errors",
    "capture_exception",
    "capture_message",
    "error_context",
    "flush_errors",
    "init_sentry",
    "set_user",
    # Secrets
    "SecretsManager",
    "get_secret",
    "get_secrets_manager",
    "require_secret",
    # Exceptions
    "CacheError",
    "ConceptExistsError",
    "ConceptNotFoundError",
    "ConfigurationError",
    "ConversionError",
    "DatabaseError",
    "EmbeddingError",
    "LLMError",
    "MappingNotFoundError",
    "MessageParseError",
    "MissingConfigurationError",
    "NormalizationError",
    "OntologyError",
    "OutputError",
    "PolyglotLinkError",
    "ProtocolError",
    "PublishError",
    "SchemaError",
    "SchemaExtractionError",
    "StorageError",
    "TranslationError",
    "TypeCoercionError",
    "UnsafeFormulaError",
    "UnsupportedEncodingError",
    "ValidationError",
    # Logging
    "LogContext",
    "MetricsLogger",
    "configure_logging",
    "get_logger",
    "log_performance",
    # Validation
    "detect_malicious_patterns",
    "is_valid_topic",
    "sanitize_dict_keys",
    "sanitize_identifier",
    "sanitize_string",
    "sanitize_topic",
    "validate_confidence",
    "validate_field_type",
    "validate_json_depth",
    "validate_json_payload",
    "validate_json_size",
    "validate_number",
    "validate_payload_size",
    "validate_protocol",
    # Metrics
    "PolyglotLinkMetrics",
    "get_metrics",
    "metrics",
]
