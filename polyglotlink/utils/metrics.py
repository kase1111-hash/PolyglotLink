"""
PolyglotLink Metrics and Telemetry Module

Provides Prometheus-compatible metrics for monitoring system performance,
throughput, and health.

Usage:
    from polyglotlink.utils.metrics import metrics

    # Record a message processed
    metrics.record_message_processed("MQTT", "sensor-001", success=True)

    # Time an operation
    with metrics.time_operation("schema_extraction"):
        schema = extractor.extract_schema(raw)

    # Start metrics server
    metrics.start_server(port=9090)
"""

import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)


class PolyglotLinkMetrics:
    """Centralized metrics collection for PolyglotLink."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._server_started = False

    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""

        # Application info
        self.app_info = Info(
            "polyglotlink",
            "PolyglotLink application information",
            registry=self.registry,
        )

        # Message processing metrics
        self.messages_received_total = Counter(
            "polyglotlink_messages_received_total",
            "Total number of messages received",
            ["protocol", "device_type"],
            registry=self.registry,
        )

        self.messages_processed_total = Counter(
            "polyglotlink_messages_processed_total",
            "Total number of messages successfully processed",
            ["protocol", "device_type"],
            registry=self.registry,
        )

        self.messages_failed_total = Counter(
            "polyglotlink_messages_failed_total",
            "Total number of messages that failed processing",
            ["protocol", "device_type", "error_type"],
            registry=self.registry,
        )

        self.message_processing_seconds = Histogram(
            "polyglotlink_message_processing_seconds",
            "Time spent processing messages",
            ["protocol", "stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        # Pipeline stage metrics
        self.schema_extraction_seconds = Histogram(
            "polyglotlink_schema_extraction_seconds",
            "Time spent extracting schemas",
            ["payload_type"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry,
        )

        self.semantic_translation_seconds = Histogram(
            "polyglotlink_semantic_translation_seconds",
            "Time spent on semantic translation",
            ["method"],  # embedding, llm, cache
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        self.normalization_seconds = Histogram(
            "polyglotlink_normalization_seconds",
            "Time spent on normalization",
            ["has_conversions"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits_total = Counter(
            "polyglotlink_cache_hits_total",
            "Total cache hits",
            ["cache_type"],  # schema, embedding, mapping
            registry=self.registry,
        )

        self.cache_misses_total = Counter(
            "polyglotlink_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_size = Gauge(
            "polyglotlink_cache_size",
            "Current cache size",
            ["cache_type"],
            registry=self.registry,
        )

        # LLM metrics
        self.llm_requests_total = Counter(
            "polyglotlink_llm_requests_total",
            "Total LLM API requests",
            ["model", "status"],
            registry=self.registry,
        )

        self.llm_tokens_total = Counter(
            "polyglotlink_llm_tokens_total",
            "Total LLM tokens used",
            ["model", "token_type"],  # input, output
            registry=self.registry,
        )

        self.llm_request_seconds = Histogram(
            "polyglotlink_llm_request_seconds",
            "LLM request latency",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )

        # Output metrics
        self.output_messages_total = Counter(
            "polyglotlink_output_messages_total",
            "Total messages sent to outputs",
            ["destination", "status"],  # kafka, mqtt, http, etc.
            registry=self.registry,
        )

        self.output_bytes_total = Counter(
            "polyglotlink_output_bytes_total",
            "Total bytes sent to outputs",
            ["destination"],
            registry=self.registry,
        )

        # Connection metrics
        self.active_connections = Gauge(
            "polyglotlink_active_connections",
            "Number of active connections",
            ["protocol"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "polyglotlink_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self.registry,
        )

        # System metrics
        self.uptime_seconds = Gauge(
            "polyglotlink_uptime_seconds",
            "Time since application start",
            registry=self.registry,
        )

        self.queue_size = Gauge(
            "polyglotlink_queue_size",
            "Current queue size",
            ["queue_name"],
            registry=self.registry,
        )

        # Device metrics
        self.unique_devices = Gauge(
            "polyglotlink_unique_devices",
            "Number of unique devices seen",
            ["protocol"],
            registry=self.registry,
        )

        self.device_message_rate = Gauge(
            "polyglotlink_device_message_rate",
            "Message rate per device (msgs/min)",
            ["device_id"],
            registry=self.registry,
        )

    def set_app_info(self, version: str, environment: str, **extra):
        """Set application info labels."""
        self.app_info.info({"version": version, "environment": environment, **extra})

    def record_message_received(self, protocol: str, device_type: str = "unknown"):
        """Record a received message."""
        self.messages_received_total.labels(protocol=protocol, device_type=device_type).inc()

    def record_message_processed(
        self,
        protocol: str,
        device_type: str = "unknown",
        success: bool = True,
        error_type: str = "",
    ):
        """Record a processed message."""
        if success:
            self.messages_processed_total.labels(protocol=protocol, device_type=device_type).inc()
        else:
            self.messages_failed_total.labels(
                protocol=protocol, device_type=device_type, error_type=error_type
            ).inc()

    def record_processing_time(self, protocol: str, stage: str, duration_seconds: float):
        """Record message processing time."""
        self.message_processing_seconds.labels(protocol=protocol, stage=stage).observe(
            duration_seconds
        )

    def record_schema_extraction(self, payload_type: str, duration_seconds: float):
        """Record schema extraction time."""
        self.schema_extraction_seconds.labels(payload_type=payload_type).observe(duration_seconds)

    def record_semantic_translation(self, method: str, duration_seconds: float):
        """Record semantic translation time."""
        self.semantic_translation_seconds.labels(method=method).observe(duration_seconds)

    def record_normalization(self, has_conversions: bool, duration_seconds: float):
        """Record normalization time."""
        self.normalization_seconds.labels(has_conversions=str(has_conversions).lower()).observe(
            duration_seconds
        )

    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache hit or miss."""
        if hit:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses_total.labels(cache_type=cache_type).inc()

    def set_cache_size(self, cache_type: str, size: int):
        """Set current cache size."""
        self.cache_size.labels(cache_type=cache_type).set(size)

    def record_llm_request(
        self,
        model: str,
        success: bool,
        duration_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record LLM API request."""
        status = "success" if success else "error"
        self.llm_requests_total.labels(model=model, status=status).inc()
        self.llm_request_seconds.labels(model=model).observe(duration_seconds)

        if input_tokens > 0:
            self.llm_tokens_total.labels(model=model, token_type="input").inc(input_tokens)

        if output_tokens > 0:
            self.llm_tokens_total.labels(model=model, token_type="output").inc(output_tokens)

    def record_output(self, destination: str, success: bool, bytes_sent: int = 0):
        """Record output message."""
        status = "success" if success else "error"
        self.output_messages_total.labels(destination=destination, status=status).inc()

        if bytes_sent > 0:
            self.output_bytes_total.labels(destination=destination).inc(bytes_sent)

    def record_error(self, error_type: str, component: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def set_active_connections(self, protocol: str, count: int):
        """Set active connection count."""
        self.active_connections.labels(protocol=protocol).set(count)

    def set_queue_size(self, queue_name: str, size: int):
        """Set current queue size."""
        self.queue_size.labels(queue_name=queue_name).set(size)

    def set_unique_devices(self, protocol: str, count: int):
        """Set unique device count."""
        self.unique_devices.labels(protocol=protocol).set(count)

    @contextmanager
    def time_operation(self, operation: str, labels: dict | None = None):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            # Generic timing - can be extended based on operation type
            if operation == "schema_extraction":
                payload_type = labels.get("payload_type", "json") if labels else "json"
                self.record_schema_extraction(payload_type, duration)
            elif operation == "semantic_translation":
                method = labels.get("method", "unknown") if labels else "unknown"
                self.record_semantic_translation(method, duration)
            elif operation == "normalization":
                has_conv = labels.get("has_conversions", False) if labels else False
                self.record_normalization(has_conv, duration)

    def timed(self, operation: str):
        """Decorator for timing functions."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.time_operation(operation):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def start_server(self, port: int = 9090, addr: str = ""):
        """Start the Prometheus metrics HTTP server."""
        if not self._server_started:
            start_http_server(port, addr, registry=self.registry)
            self._server_started = True
            return True
        return False

    def get_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
metrics = PolyglotLinkMetrics()


def get_metrics() -> PolyglotLinkMetrics:
    """Get the global metrics instance."""
    return metrics
