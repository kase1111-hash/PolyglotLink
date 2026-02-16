"""
PolyglotLink Logging Module

Structured logging configuration using structlog.
Provides consistent, machine-readable logs with context propagation.
"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import structlog
from structlog.types import EventDict, Processor

# Context variables for request-scoped logging
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
device_id_var: ContextVar[str | None] = ContextVar("device_id", default=None)
message_id_var: ContextVar[str | None] = ContextVar("message_id", default=None)


def add_timestamp(_logger: logging.Logger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO format timestamp to log event."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
    return event_dict


def add_context(_logger: logging.Logger, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add context variables to log event."""
    request_id = request_id_var.get()
    device_id = device_id_var.get()
    message_id = message_id_var.get()

    if request_id:
        event_dict["request_id"] = request_id
    if device_id:
        event_dict["device_id"] = device_id
    if message_id:
        event_dict["message_id"] = message_id

    return event_dict


def add_service_info(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Add service information to log event."""
    event_dict["service"] = "polyglotlink"
    return event_dict


def format_exception(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Format exception information if present."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        if isinstance(exc_info, BaseException):
            event_dict["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
            }
            # Add details if it's a PolyglotLinkError
            if hasattr(exc_info, "to_dict"):
                event_dict["exception"]["details"] = exc_info.to_dict()
        elif exc_info is True:
            import traceback

            event_dict["exception"] = {"traceback": traceback.format_exc()}
    return event_dict


def censor_sensitive_data(
    _logger: logging.Logger, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Censor sensitive data in logs."""
    sensitive_keys = {
        "password",
        "api_key",
        "token",
        "secret",
        "authorization",
        "apikey",
        "access_token",
        "refresh_token",
        "private_key",
    }

    def censor_dict(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in d.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                result[key] = "***REDACTED***"
            elif isinstance(value, dict):
                result[key] = censor_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    censor_dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                result[key] = value
        return result

    return censor_dict(event_dict)


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    development: bool = False,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON formatted logs
        development: Whether to use development-friendly formatting
    """
    # Shared processors for all loggers
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_timestamp,
        add_context,
        add_service_info,
        format_exception,
        censor_sensitive_data,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if development:
        # Development: colored console output
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]
    elif json_logs:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Plain text output
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=False)]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Set levels for noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("kafka").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (defaults to module name)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary logging context.

    Usage:
        with LogContext(request_id="abc123", device_id="sensor-01"):
            logger.info("Processing message")
    """

    def __init__(
        self,
        request_id: str | None = None,
        device_id: str | None = None,
        message_id: str | None = None,
    ):
        self.request_id = request_id
        self.device_id = device_id
        self.message_id = message_id
        self._tokens: list = []

    def __enter__(self) -> "LogContext":
        if self.request_id:
            self._tokens.append(request_id_var.set(self.request_id))
        if self.device_id:
            self._tokens.append(device_id_var.set(self.device_id))
        if self.message_id:
            self._tokens.append(message_id_var.set(self.message_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for token in reversed(self._tokens):
            # Reset context vars
            if hasattr(token, "var"):
                token.var.reset(token)


def log_performance(
    logger: structlog.stdlib.BoundLogger, operation: str, start_time: datetime, **extra: Any
) -> None:
    """
    Log performance metrics for an operation.

    Args:
        logger: Logger instance
        operation: Name of the operation
        start_time: When the operation started
        **extra: Additional context to log
    """
    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    logger.info(
        f"{operation} completed", operation=operation, duration_ms=round(duration_ms, 2), **extra
    )


class MetricsLogger:
    """
    Logger for metrics and statistics.
    Provides methods for logging common metric types.
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger | None = None):
        self.logger = logger or get_logger("metrics")

    def count(self, metric: str, value: int = 1, **tags: Any) -> None:
        """Log a count metric."""
        self.logger.info("metric.count", metric=metric, value=value, metric_type="counter", **tags)

    def gauge(self, metric: str, value: float, **tags: Any) -> None:
        """Log a gauge metric."""
        self.logger.info("metric.gauge", metric=metric, value=value, metric_type="gauge", **tags)

    def histogram(self, metric: str, value: float, **tags: Any) -> None:
        """Log a histogram metric."""
        self.logger.info(
            "metric.histogram", metric=metric, value=value, metric_type="histogram", **tags
        )

    def timing(self, metric: str, duration_ms: float, **tags: Any) -> None:
        """Log a timing metric."""
        self.logger.info(
            "metric.timing", metric=metric, duration_ms=duration_ms, metric_type="timing", **tags
        )
