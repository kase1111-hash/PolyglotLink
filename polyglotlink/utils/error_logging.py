"""
PolyglotLink Error Logging Module

Provides error capture utilities. Sentry integration removed;
functions remain as no-ops/local-logging stubs so call sites don't break.
"""

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from polyglotlink.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def init_sentry() -> bool:
    """No-op. Sentry integration removed."""
    return False


def capture_exception(
    error: Exception,
    context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    level: str = "error",
) -> str | None:
    """Log exception locally."""
    logger.error(
        "Exception occurred",
        exc_info=error,
        context=context,
        tags=tags,
    )
    return None


def capture_message(
    message: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str | None:
    """Log message locally."""
    log_func = getattr(logger, level, logger.info)
    log_func(message, context=context, tags=tags)
    return None


def set_user(
    user_id: str | None = None,
    device_id: str | None = None,
    **extra: Any,
) -> None:
    """No-op."""
    pass


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: dict[str, Any] | None = None,
) -> None:
    """No-op."""
    pass


@contextmanager
def error_context(
    operation: str,
    context: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    reraise: bool = True,
):
    """Context manager that catches and logs errors."""
    try:
        yield
    except Exception as e:
        capture_exception(
            e,
            context={"operation": operation, **(context or {})},
            tags={"operation": operation, **(tags or {})},
        )
        if reraise:
            raise


def capture_errors(
    operation: str | None = None,
    tags: dict[str, str] | None = None,
    reraise: bool = True,
) -> Callable[[F], F]:
    """Decorator for capturing errors from functions."""

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

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def flush(timeout: float = 2.0) -> None:
    """No-op."""
    pass
