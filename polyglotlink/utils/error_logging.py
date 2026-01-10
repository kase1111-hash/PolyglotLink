"""
PolyglotLink Error Logging Module

Integrates with Sentry for error tracking and monitoring.
Provides utilities for capturing errors with context.
"""

import sys
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from polyglotlink.utils.config import get_settings
from polyglotlink.utils.exceptions import PolyglotLinkError
from polyglotlink.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic function decorator
F = TypeVar("F", bound=Callable[..., Any])

# Global flag to track if Sentry is initialized
_sentry_initialized = False


def init_sentry() -> bool:
    """
    Initialize Sentry error tracking.

    Returns:
        True if Sentry was initialized successfully, False otherwise.
    """
    global _sentry_initialized

    if _sentry_initialized:
        return True

    settings = get_settings()

    if not settings.sentry.dsn:
        logger.info("Sentry DSN not configured, error tracking disabled")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.asyncio import AsyncioIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        # Configure logging integration
        logging_integration = LoggingIntegration(
            level=None,  # Capture all logs as breadcrumbs
            event_level=None,  # Don't send logs as events (we do that manually)
        )

        sentry_sdk.init(
            dsn=settings.sentry.dsn,
            environment=settings.sentry.environment,
            traces_sample_rate=settings.sentry.traces_sample_rate,
            integrations=[
                logging_integration,
                AsyncioIntegration(),
            ],
            # Don't send PII
            send_default_pii=False,
            # Attach stacktrace to messages
            attach_stacktrace=True,
            # Set release version
            release=f"polyglotlink@0.1.0",
            # Before send hook for filtering
            before_send=_before_send,
        )

        _sentry_initialized = True
        logger.info(
            "Sentry initialized",
            environment=settings.sentry.environment,
            traces_sample_rate=settings.sentry.traces_sample_rate,
        )
        return True

    except ImportError:
        logger.warning("sentry-sdk not installed, error tracking disabled")
        return False
    except Exception as e:
        logger.error("Failed to initialize Sentry", error=str(e))
        return False


def _before_send(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter events before sending to Sentry.

    - Scrub sensitive data
    - Filter out expected errors
    - Add additional context
    """
    # Get exception info if present
    exc_info = hint.get("exc_info")
    if exc_info:
        exc_type, exc_value, _ = exc_info

        # Don't send expected/recoverable errors
        if isinstance(exc_value, PolyglotLinkError):
            if exc_value.recoverable:
                # Log locally but don't send to Sentry
                return None

    # Scrub sensitive data from request
    if "request" in event:
        request = event["request"]
        if "headers" in request:
            sensitive_headers = {"authorization", "x-api-key", "cookie"}
            request["headers"] = {
                k: "[Filtered]" if k.lower() in sensitive_headers else v
                for k, v in request.get("headers", {}).items()
            }

    # Scrub sensitive data from extra context
    if "extra" in event:
        event["extra"] = _scrub_sensitive(event["extra"])

    return event


def _scrub_sensitive(data: Any) -> Any:
    """Recursively scrub sensitive data from dictionaries."""
    sensitive_keys = {
        "password", "api_key", "token", "secret", "authorization",
        "apikey", "access_token", "refresh_token", "private_key", "dsn"
    }

    if isinstance(data, dict):
        return {
            k: "[Filtered]" if any(s in k.lower() for s in sensitive_keys) else _scrub_sensitive(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_scrub_sensitive(item) for item in data]
    return data


def capture_exception(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    level: str = "error",
) -> Optional[str]:
    """
    Capture an exception to Sentry.

    Args:
        error: The exception to capture
        context: Additional context data
        tags: Tags for categorization
        level: Error level (error, warning, info)

    Returns:
        Event ID if captured, None otherwise
    """
    if not _sentry_initialized:
        # Log locally if Sentry not available
        logger.error(
            "Exception occurred",
            exc_info=error,
            context=context,
            tags=tags,
        )
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            # Set level
            scope.level = level

            # Add tags
            if tags:
                for key, value in tags.items():
                    scope.set_tag(key, value)

            # Add context
            if context:
                scope.set_context("additional", context)

            # Add PolyglotLinkError details
            if isinstance(error, PolyglotLinkError):
                scope.set_tag("error_code", error.code)
                scope.set_tag("recoverable", str(error.recoverable))
                scope.set_context("error_details", error.details)

            # Capture the exception
            event_id = sentry_sdk.capture_exception(error)
            return event_id

    except Exception as e:
        logger.error("Failed to capture exception to Sentry", error=str(e))
        return None


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Capture a message to Sentry.

    Args:
        message: The message to capture
        level: Message level (error, warning, info, debug)
        context: Additional context data
        tags: Tags for categorization

    Returns:
        Event ID if captured, None otherwise
    """
    if not _sentry_initialized:
        log_func = getattr(logger, level, logger.info)
        log_func(message, context=context, tags=tags)
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            scope.level = level

            if tags:
                for key, value in tags.items():
                    scope.set_tag(key, value)

            if context:
                scope.set_context("additional", context)

            event_id = sentry_sdk.capture_message(message)
            return event_id

    except Exception as e:
        logger.error("Failed to capture message to Sentry", error=str(e))
        return None


def set_user(
    user_id: Optional[str] = None,
    device_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """
    Set user context for Sentry events.

    Args:
        user_id: User identifier
        device_id: Device identifier
        **extra: Additional user attributes
    """
    if not _sentry_initialized:
        return

    try:
        import sentry_sdk

        user_data = {}
        if user_id:
            user_data["id"] = user_id
        if device_id:
            user_data["device_id"] = device_id
        user_data.update(extra)

        sentry_sdk.set_user(user_data if user_data else None)

    except Exception as e:
        logger.warning("Failed to set Sentry user", error=str(e))


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add a breadcrumb for Sentry event context.

    Args:
        message: Breadcrumb message
        category: Breadcrumb category
        level: Breadcrumb level
        data: Additional data
    """
    if not _sentry_initialized:
        return

    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data,
        )

    except Exception:
        pass  # Don't log breadcrumb failures


@contextmanager
def error_context(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    reraise: bool = True,
):
    """
    Context manager for capturing errors with context.

    Args:
        operation: Name of the operation
        context: Additional context data
        tags: Tags for categorization
        reraise: Whether to re-raise the exception

    Usage:
        with error_context("process_message", {"device_id": "sensor-01"}):
            process_message(msg)
    """
    add_breadcrumb(f"Starting {operation}", category="operation", level="info")

    try:
        yield
        add_breadcrumb(f"Completed {operation}", category="operation", level="info")
    except Exception as e:
        capture_exception(
            e,
            context={"operation": operation, **(context or {})},
            tags={"operation": operation, **(tags or {})},
        )
        if reraise:
            raise


def capture_errors(
    operation: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for capturing errors from functions.

    Args:
        operation: Operation name (defaults to function name)
        tags: Tags for categorization
        reraise: Whether to re-raise the exception

    Usage:
        @capture_errors(operation="message_processing")
        async def process_message(msg):
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with error_context(op_name, tags=tags, reraise=reraise):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with error_context(op_name, tags=tags, reraise=reraise):
                return await func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def flush(timeout: float = 2.0) -> None:
    """
    Flush pending Sentry events.

    Args:
        timeout: Maximum time to wait for flush
    """
    if not _sentry_initialized:
        return

    try:
        import sentry_sdk
        sentry_sdk.flush(timeout=timeout)
    except Exception:
        pass
