"""
PolyglotLink Output Broker Module

This module routes normalized messages to various output destinations:
Kafka, MQTT, HTTP webhooks, WebSocket, TimescaleDB, and JSON-LD export.
"""

import asyncio
import contextlib
import json
from dataclasses import dataclass
from datetime import datetime

import structlog

from polyglotlink.models.schemas import NormalizedMessage

logger = structlog.get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class KafkaOutputConfig:
    """Kafka output configuration."""

    enabled: bool = False
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "iot.normalized"
    acks: str = "all"
    compression_type: str = "gzip"
    batch_size: int = 16384
    linger_ms: int = 10


@dataclass
class MQTTOutputConfig:
    """MQTT output configuration."""

    enabled: bool = False
    broker_host: str = "localhost"
    broker_port: int = 1883
    topic_prefix: str = "normalized"
    qos: int = 1
    retain: bool = False
    username: str | None = None
    password: str | None = None


@dataclass
class HTTPOutputConfig:
    """HTTP webhook output configuration."""

    enabled: bool = False
    endpoints: list[str] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    headers: dict[str, str] = None

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.headers is None:
            self.headers = {}


@dataclass
class WebSocketOutputConfig:
    """WebSocket broadcast configuration."""

    enabled: bool = False
    channel_prefix: str = "devices"


@dataclass
class TimescaleOutputConfig:
    """TimescaleDB output configuration."""

    enabled: bool = False
    connection_string: str = "postgresql://localhost/iot"
    table_name: str = "iot_metrics"
    batch_size: int = 100
    flush_interval_seconds: float = 1.0


@dataclass
class OutputBrokerConfig:
    """Complete output broker configuration."""

    kafka: KafkaOutputConfig = None
    mqtt: MQTTOutputConfig = None
    http: HTTPOutputConfig = None
    websocket: WebSocketOutputConfig = None
    timescale: TimescaleOutputConfig = None
    output_format: str = "json"  # "json" or "json-ld"

    def __post_init__(self):
        if self.kafka is None:
            self.kafka = KafkaOutputConfig()
        if self.mqtt is None:
            self.mqtt = MQTTOutputConfig()
        if self.http is None:
            self.http = HTTPOutputConfig()
        if self.websocket is None:
            self.websocket = WebSocketOutputConfig()
        if self.timescale is None:
            self.timescale = TimescaleOutputConfig()


# ============================================================================
# Routing
# ============================================================================


@dataclass
class OutputRouting:
    """Computed routing for a message."""

    kafka_topic: str | None = None
    mqtt_topic: str | None = None
    http_endpoints: list[str] = None
    websocket_channel: str | None = None
    format: str = "json"

    def __post_init__(self):
        if self.http_endpoints is None:
            self.http_endpoints = []


class TopicMapper:
    """Maps device/context to output topics."""

    def __init__(self):
        self._device_rules: dict[str, dict[str, str]] = {}
        self._context_rules: dict[str, dict[str, str]] = {}

    def add_device_rule(
        self, device_pattern: str, kafka_topic: str | None = None, mqtt_topic: str | None = None
    ) -> None:
        """Add routing rule for device pattern."""
        self._device_rules[device_pattern] = {"kafka": kafka_topic, "mqtt": mqtt_topic}

    def add_context_rule(
        self, context: str, kafka_topic: str | None = None, mqtt_topic: str | None = None
    ) -> None:
        """Add routing rule for device context."""
        self._context_rules[context] = {"kafka": kafka_topic, "mqtt": mqtt_topic}

    def get_kafka_topic(self, device_id: str, context: str | None, default_prefix: str) -> str:
        """Determine Kafka topic for message."""
        # Check device rules
        for pattern, rules in self._device_rules.items():
            if pattern in device_id and rules.get("kafka"):
                return rules["kafka"]

        # Check context rules
        if context and context in self._context_rules and self._context_rules[context].get("kafka"):
            return self._context_rules[context]["kafka"]

        # Default: use context-based topic
        suffix = context.replace(" ", "_").lower() if context else "general"
        return f"{default_prefix}.{suffix}"

    def get_mqtt_topic(
        self,
        device_id: str,
        context: str | None,  # noqa: ARG002
        default_prefix: str,
    ) -> str:
        """Determine MQTT topic for message."""
        # Check device rules
        for pattern, rules in self._device_rules.items():
            if pattern in device_id and rules.get("mqtt"):
                return rules["mqtt"]

        # Default: device-based topic
        return f"{default_prefix}/{device_id}"


# ============================================================================
# Publish Result
# ============================================================================


@dataclass
class PublishResult:
    """Result of publishing a message."""

    message_id: str
    outputs: list[tuple[str, bool]]
    published_at: datetime

    @property
    def success(self) -> bool:
        """Check if all outputs succeeded."""
        return all(success for _, success in self.outputs)

    @property
    def partial_success(self) -> bool:
        """Check if at least one output succeeded."""
        return any(success for _, success in self.outputs)


# ============================================================================
# WebSocket Manager
# ============================================================================


class WebSocketManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self._connections: dict[str, set] = {}  # channel -> set of websockets

    def subscribe(self, channel: str, websocket) -> None:
        """Subscribe a websocket to a channel."""
        if channel not in self._connections:
            self._connections[channel] = set()
        self._connections[channel].add(websocket)

    def unsubscribe(self, channel: str, websocket) -> None:
        """Unsubscribe a websocket from a channel."""
        if channel in self._connections:
            self._connections[channel].discard(websocket)
            if not self._connections[channel]:
                del self._connections[channel]

    async def broadcast(self, channel: str, payload: bytes) -> int:
        """Broadcast to all subscribers of a channel."""
        if channel not in self._connections:
            return 0

        count = 0
        dead_connections = []

        for ws in self._connections[channel]:
            try:
                await ws.send(payload)
                count += 1
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            self._connections[channel].discard(ws)

        return count


# ============================================================================
# Output Broker
# ============================================================================


class OutputBroker:
    """
    Routes normalized messages to configured output destinations.
    """

    def __init__(self, config: OutputBrokerConfig | None = None):
        self.config = config or OutputBrokerConfig()

        self._kafka_producer = None
        self._mqtt_client = None
        self._http_session = None
        self._websocket_manager = WebSocketManager()
        self._timescale_pool = None
        self._timescale_buffer: list[dict] = []
        self._timescale_task: asyncio.Task | None = None

        self.topic_mapper = TopicMapper()

    async def initialize(self) -> None:
        """Initialize output connections."""
        if self.config.kafka.enabled:
            await self._init_kafka()

        if self.config.mqtt.enabled:
            await self._init_mqtt()

        if self.config.http.enabled:
            await self._init_http()

        if self.config.timescale.enabled:
            await self._init_timescale()

        logger.info("Output broker initialized")

    async def shutdown(self) -> None:
        """Shutdown output connections."""
        if self._kafka_producer:
            self._kafka_producer.close()

        if self._mqtt_client:
            self._mqtt_client.disconnect()

        if self._http_session:
            await self._http_session.close()

        if self._timescale_task:
            self._timescale_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._timescale_task

        if self._timescale_pool:
            await self._timescale_pool.close()

        logger.info("Output broker shutdown")

    async def _init_kafka(self) -> None:
        """Initialize Kafka producer."""
        try:
            from kafka import KafkaProducer

            self._kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka.bootstrap_servers,
                acks=self.config.kafka.acks,
                compression_type=self.config.kafka.compression_type,
                batch_size=self.config.kafka.batch_size,
                linger_ms=self.config.kafka.linger_ms,
                value_serializer=lambda v: v,  # We'll serialize ourselves
            )
            logger.info("Kafka producer initialized")
        except ImportError:
            logger.warning("kafka-python not installed, Kafka output disabled")
            self.config.kafka.enabled = False
        except Exception as e:
            logger.error("Failed to initialize Kafka", error=str(e))
            self.config.kafka.enabled = False

    async def _init_mqtt(self) -> None:
        """Initialize MQTT client."""
        try:
            import paho.mqtt.client as mqtt

            self._mqtt_client = mqtt.Client()

            if self.config.mqtt.username:
                self._mqtt_client.username_pw_set(
                    self.config.mqtt.username, self.config.mqtt.password
                )

            self._mqtt_client.connect_async(
                self.config.mqtt.broker_host, self.config.mqtt.broker_port
            )
            self._mqtt_client.loop_start()
            logger.info("MQTT client initialized")
        except ImportError:
            logger.warning("paho-mqtt not installed, MQTT output disabled")
            self.config.mqtt.enabled = False
        except Exception as e:
            logger.error("Failed to initialize MQTT", error=str(e))
            self.config.mqtt.enabled = False

    async def _init_http(self) -> None:
        """Initialize HTTP session."""
        try:
            import httpx

            self._http_session = httpx.AsyncClient(
                timeout=self.config.http.timeout_seconds, headers=self.config.http.headers
            )
            logger.info("HTTP client initialized")
        except ImportError:
            logger.warning("httpx not installed, HTTP output disabled")
            self.config.http.enabled = False
        except Exception as e:
            logger.error("Failed to initialize HTTP client", error=str(e))
            self.config.http.enabled = False

    async def _init_timescale(self) -> None:
        """Initialize TimescaleDB connection pool."""
        try:
            import asyncpg

            self._timescale_pool = await asyncpg.create_pool(
                self.config.timescale.connection_string, min_size=1, max_size=10
            )

            # Start flush task
            self._timescale_task = asyncio.create_task(self._timescale_flush_loop())

            logger.info("TimescaleDB connection initialized")
        except ImportError:
            logger.warning("asyncpg not installed, TimescaleDB output disabled")
            self.config.timescale.enabled = False
        except Exception as e:
            logger.error("Failed to initialize TimescaleDB", error=str(e))
            self.config.timescale.enabled = False

    async def _timescale_flush_loop(self) -> None:
        """Periodically flush TimescaleDB buffer."""
        while True:
            await asyncio.sleep(self.config.timescale.flush_interval_seconds)
            if self._timescale_buffer:
                await self._flush_timescale_buffer()

    async def _flush_timescale_buffer(self) -> None:
        """Flush buffered metrics to TimescaleDB."""
        if not self._timescale_buffer or not self._timescale_pool:
            return

        buffer = self._timescale_buffer
        self._timescale_buffer = []

        try:
            async with self._timescale_pool.acquire() as conn:
                await conn.executemany(
                    f"""
                    INSERT INTO {self.config.timescale.table_name}
                    (time, device_id, metric, value)
                    VALUES ($1, $2, $3, $4)
                    """,
                    [(m["time"], m["device_id"], m["metric"], m["value"]) for m in buffer],
                )
            logger.debug("Flushed metrics to TimescaleDB", count=len(buffer))
        except Exception as e:
            logger.error("Failed to flush to TimescaleDB", error=str(e))
            # Put failed records back
            self._timescale_buffer.extend(buffer)

    async def publish(self, message: NormalizedMessage) -> PublishResult:
        """
        Route and publish a normalized message to all configured outputs.
        """
        results: list[tuple[str, bool]] = []

        # Compute routing
        routing = self._compute_routing(message)

        # Format message
        formatted = self._format_message(message, routing.format)

        # Publish to each enabled output
        if self.config.kafka.enabled and routing.kafka_topic:
            result = await self._kafka_publish(routing.kafka_topic, formatted)
            results.append(("kafka", result))

        if self.config.mqtt.enabled and routing.mqtt_topic:
            result = await self._mqtt_publish(routing.mqtt_topic, formatted)
            results.append(("mqtt", result))

        if self.config.http.enabled and routing.http_endpoints:
            for endpoint in routing.http_endpoints:
                result = await self._http_post(endpoint, formatted)
                results.append(("http", result))

        if self.config.websocket.enabled and routing.websocket_channel:
            result = await self._websocket_broadcast(routing.websocket_channel, formatted)
            results.append(("websocket", result))

        if self.config.timescale.enabled:
            result = await self._store_timeseries(message)
            results.append(("timescale", result))

        return PublishResult(
            message_id=message.message_id, outputs=results, published_at=datetime.utcnow()
        )

    def _compute_routing(self, message: NormalizedMessage) -> OutputRouting:
        """Determine output routing for a message."""
        routing = OutputRouting(format=self.config.output_format)

        if self.config.kafka.enabled:
            routing.kafka_topic = self.topic_mapper.get_kafka_topic(
                message.device_id, message.context, self.config.kafka.topic_prefix
            )

        if self.config.mqtt.enabled:
            routing.mqtt_topic = self.topic_mapper.get_mqtt_topic(
                message.device_id, message.context, self.config.mqtt.topic_prefix
            )

        if self.config.http.enabled:
            routing.http_endpoints = self.config.http.endpoints.copy()

        if self.config.websocket.enabled:
            routing.websocket_channel = (
                f"{self.config.websocket.channel_prefix}/{message.device_id}"
            )

        return routing

    def _format_message(self, message: NormalizedMessage, format: str) -> bytes:
        """Format message for output."""
        if format == "json-ld":
            return self._to_jsonld(message).encode()
        else:
            return json.dumps(message.model_dump(), default=str).encode()

    def _to_jsonld(self, message: NormalizedMessage) -> str:
        """Convert message to JSON-LD format."""
        jsonld = {
            "@context": {
                "@vocab": "https://schema.org/",
                "iot": "https://www.w3.org/2019/wot/td#",
                "sosa": "http://www.w3.org/ns/sosa/",
            },
            "@type": "sosa:Observation",
            "@id": f"urn:polyglotlink:message:{message.message_id}",
            "sosa:hasFeatureOfInterest": {
                "@type": "iot:Thing",
                "@id": f"urn:polyglotlink:device:{message.device_id}",
                "name": message.device_id,
            },
            "sosa:resultTime": message.timestamp.isoformat(),
            "sosa:hasResult": {"@type": "sosa:Result", **message.data},
            "iot:hasSecurityScheme": {"scheme": "nosec"},
        }

        if message.context:
            jsonld["sosa:hasFeatureOfInterest"]["description"] = message.context

        if message.metadata:
            jsonld["additionalProperty"] = [
                {"@type": "PropertyValue", "name": k, "value": v}
                for k, v in message.metadata.items()
            ]

        return json.dumps(jsonld, default=str)

    async def _kafka_publish(self, topic: str, payload: bytes) -> bool:
        """Publish to Kafka topic."""
        try:
            future = self._kafka_producer.send(
                topic, value=payload, timestamp_ms=int(datetime.utcnow().timestamp() * 1000)
            )
            # Wait for send to complete
            await asyncio.get_event_loop().run_in_executor(None, lambda: future.get(timeout=10))
            return True
        except Exception as e:
            logger.error("Kafka publish failed", topic=topic, error=str(e))
            return False

    async def _mqtt_publish(self, topic: str, payload: bytes) -> bool:
        """Publish to MQTT topic."""
        try:
            result = self._mqtt_client.publish(
                topic, payload, qos=self.config.mqtt.qos, retain=self.config.mqtt.retain
            )
            # Check if publish was successful
            return result.rc == 0
        except Exception as e:
            logger.error("MQTT publish failed", topic=topic, error=str(e))
            return False

    async def _http_post(self, endpoint: str, payload: bytes) -> bool:
        """POST to HTTP endpoint with retries."""
        for attempt in range(self.config.http.retry_count):
            try:
                response = await self._http_session.post(
                    endpoint, content=payload, headers={"Content-Type": "application/json"}
                )
                return response.status_code < 400
            except Exception as e:
                logger.warning(
                    "HTTP post failed", endpoint=endpoint, attempt=attempt + 1, error=str(e)
                )
                if attempt < self.config.http.retry_count - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        return False

    async def _websocket_broadcast(self, channel: str, payload: bytes) -> bool:
        """Broadcast to WebSocket channel."""
        try:
            count = await self._websocket_manager.broadcast(channel, payload)
            return count > 0
        except Exception as e:
            logger.error("WebSocket broadcast failed", channel=channel, error=str(e))
            return False

    async def _store_timeseries(self, message: NormalizedMessage) -> bool:
        """Store normalized data in TimescaleDB."""
        try:
            # Extract numeric fields for time-series
            for field, value in message.data.items():
                if isinstance(value, (int, float)) and not field.startswith("_"):
                    self._timescale_buffer.append(
                        {
                            "time": message.timestamp,
                            "device_id": message.device_id,
                            "metric": field,
                            "value": float(value),
                        }
                    )

            # Flush if buffer is full
            if len(self._timescale_buffer) >= self.config.timescale.batch_size:
                await self._flush_timescale_buffer()

            return True
        except Exception as e:
            logger.error("TimescaleDB insert failed", error=str(e))
            return False

    def get_websocket_manager(self) -> WebSocketManager:
        """Get WebSocket manager for external use."""
        return self._websocket_manager
