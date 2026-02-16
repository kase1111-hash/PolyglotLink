"""
PolyglotLink Server Module

Main server application that orchestrates all components.
"""

import asyncio
import contextlib
import signal
from datetime import datetime, timezone
from typing import Any

from polyglotlink.utils.config import get_settings
from polyglotlink.utils.error_logging import add_breadcrumb, capture_errors
from polyglotlink.utils.logging import LogContext, get_logger, log_performance

logger = get_logger(__name__)


class PolyglotLinkServer:
    """
    Main server class that orchestrates all PolyglotLink components.
    """

    def __init__(
        self,
        http_enabled: bool = True,
        mqtt_enabled: bool = True,
        coap_enabled: bool = True,
        websocket_enabled: bool = True,
        output_enabled: bool = True,
    ):
        self.http_enabled = http_enabled
        self.mqtt_enabled = mqtt_enabled
        self.coap_enabled = coap_enabled
        self.websocket_enabled = websocket_enabled
        self.output_enabled = output_enabled

        self._running = False
        self._protocol_listener = None
        self._schema_extractor = None
        self._semantic_translator = None
        self._normalization_engine = None
        self._output_broker = None
        self._processing_task = None
        self._metrics = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_failed": 0,
            "start_time": None,
        }

    async def start(self) -> None:
        """Start all server components."""
        logger.info("Starting PolyglotLink server")
        self._metrics["start_time"] = datetime.now(timezone.utc)

        # Initialize components
        await self._init_schema_extractor()
        await self._init_semantic_translator()
        await self._init_normalization_engine()

        if self.output_enabled:
            await self._init_output_broker()

        await self._init_protocol_listener()

        # Start message processing loop
        self._running = True
        self._processing_task = asyncio.create_task(self._process_messages())

        logger.info(
            "PolyglotLink server started",
            http=self.http_enabled,
            mqtt=self.mqtt_enabled,
            coap=self.coap_enabled,
            websocket=self.websocket_enabled,
        )

    async def stop(self) -> None:
        """Stop all server components."""
        logger.info("Stopping PolyglotLink server")
        self._running = False

        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        # Stop components in reverse order
        if self._protocol_listener:
            await self._protocol_listener.stop_listeners()

        if self._output_broker:
            await self._output_broker.shutdown()

        logger.info(
            "PolyglotLink server stopped",
            messages_processed=self._metrics["messages_processed"],
            messages_failed=self._metrics["messages_failed"],
        )

    async def _init_protocol_listener(self) -> None:
        """Initialize protocol listener."""
        from polyglotlink.models.schemas import (
            CoAPConfig,
            HTTPConfig,
            MQTTConfig,
            ProtocolListenerConfig,
            WebSocketConfig,
        )
        from polyglotlink.modules.protocol_listener import ProtocolListener

        settings = get_settings()

        config = ProtocolListenerConfig(
            mqtt=MQTTConfig(
                enabled=self.mqtt_enabled and settings.mqtt.enabled,
                broker_host=settings.mqtt.broker_host,
                broker_port=settings.mqtt.broker_port,
                client_id=settings.mqtt.client_id,
                username=settings.mqtt.username,
                password=settings.mqtt.password,
                tls_enabled=settings.mqtt.tls_enabled,
                ca_cert=settings.mqtt.ca_cert,
                topic_patterns=settings.mqtt.topic_patterns,
                qos=settings.mqtt.qos,
            ),
            http=HTTPConfig(
                enabled=self.http_enabled and settings.http.enabled,
                host=settings.http.host,
                port=settings.http.port,
                path_prefix=settings.http.path_prefix,
            ),
            coap=CoAPConfig(
                enabled=self.coap_enabled and settings.coap.enabled,
                host=settings.coap.host,
                port=settings.coap.port,
            ),
            websocket=WebSocketConfig(
                enabled=self.websocket_enabled and settings.websocket.enabled,
                host=settings.websocket.host,
                port=settings.websocket.port,
            ),
        )

        self._protocol_listener = ProtocolListener(config)
        await self._protocol_listener.start_listeners()

    async def _init_schema_extractor(self) -> None:
        """Initialize schema extractor with Redis-backed cache when available."""
        from polyglotlink.modules.schema_extractor import SchemaCache, SchemaExtractor

        settings = get_settings()
        redis_client = None
        try:
            import redis

            redis_client = redis.Redis.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                decode_responses=True,
            )
            redis_client.ping()
            logger.info("Redis cache connected", url=settings.redis.url)
        except Exception as e:
            logger.warning("Redis unavailable, using in-memory cache only", error=str(e))
            redis_client = None

        cache = SchemaCache(ttl_days=30, redis_client=redis_client)
        self._schema_extractor = SchemaExtractor(cache=cache)

    async def _init_semantic_translator(self) -> None:
        """Initialize semantic translator."""
        from polyglotlink.models.schemas import SemanticTranslatorConfig
        from polyglotlink.modules.semantic_translator_agent import SemanticTranslator

        settings = get_settings()

        config = SemanticTranslatorConfig(
            llm_model=settings.llm.model,
            llm_temperature=settings.llm.temperature,
            llm_max_tokens=settings.llm.max_tokens,
            max_llm_retries=settings.llm.max_retries,
            timeout_seconds=settings.llm.timeout_seconds,
            embedding_model=settings.llm.embedding_model,
            embedding_threshold=settings.llm.embedding_threshold,
        )

        # Initialize OpenAI client if API key available
        openai_client = None
        if settings.llm.openai_api_key:
            try:
                from openai import AsyncOpenAI

                openai_client = AsyncOpenAI(api_key=settings.llm.openai_api_key)
            except ImportError:
                logger.warning("openai package not installed")

        self._semantic_translator = SemanticTranslator(
            config=config,
            openai_client=openai_client,
        )

    async def _init_normalization_engine(self) -> None:
        """Initialize normalization engine."""
        from polyglotlink.modules.normalization_engine import NormalizationEngine

        self._normalization_engine = NormalizationEngine()

    async def _init_output_broker(self) -> None:
        """Initialize output broker."""
        from polyglotlink.modules.output_broker import (
            KafkaOutputConfig,
            MQTTOutputConfig,
            OutputBroker,
            OutputBrokerConfig,
        )

        settings = get_settings()

        config = OutputBrokerConfig(
            kafka=KafkaOutputConfig(
                enabled=settings.kafka.enabled,
                bootstrap_servers=settings.kafka.bootstrap_servers,
                topic_prefix=settings.kafka.topic_prefix,
            ),
            mqtt=MQTTOutputConfig(
                enabled=settings.mqtt.enabled,
                broker_host=settings.mqtt.broker_host,
                broker_port=settings.mqtt.broker_port,
            ),
        )

        self._output_broker = OutputBroker(config)
        await self._output_broker.initialize()

    @capture_errors(operation="message_processing")
    async def _process_messages(self) -> None:
        """Main message processing loop."""
        logger.info("Message processing loop started")

        async for raw_message in self._protocol_listener.messages():
            if not self._running:
                break

            self._metrics["messages_received"] += 1

            try:
                with LogContext(
                    message_id=raw_message.message_id,
                    device_id=raw_message.device_id,
                ):
                    await self._process_single_message(raw_message)
                    self._metrics["messages_processed"] += 1

            except Exception as e:
                self._metrics["messages_failed"] += 1
                logger.error(
                    "Failed to process message",
                    message_id=raw_message.message_id,
                    error=str(e),
                )

    async def _process_single_message(self, raw_message) -> None:
        """Process a single message through the pipeline."""
        start_time = datetime.now(timezone.utc)

        add_breadcrumb(
            f"Processing message from {raw_message.protocol.value}",
            category="processing",
            data={"topic": raw_message.topic},
        )

        # Step 1: Extract schema
        schema = self._schema_extractor.extract_schema(raw_message)
        logger.debug(
            "Schema extracted",
            signature=schema.schema_signature,
            fields=len(schema.fields),
        )

        # Step 2: Translate to semantic mapping
        mapping = await self._semantic_translator.translate_schema(schema)
        logger.debug(
            "Schema translated",
            confidence=mapping.confidence,
            llm_used=mapping.llm_generated,
        )

        # Step 3: Normalize values
        normalized = self._normalization_engine.normalize_message(schema, mapping)
        logger.debug(
            "Message normalized",
            fields=len(normalized.data),
            errors=len(normalized.validation_errors),
        )

        # Step 4: Publish to outputs
        if self._output_broker:
            result = await self._output_broker.publish(normalized)
            output_names = [o[0] for o in result.outputs] if result.outputs else []
            logger.debug(
                "Message published",
                outputs=output_names,
                success=result.success,
            )

        log_performance(
            logger,
            "message_processing",
            start_time,
            protocol=raw_message.protocol.value,
            device_id=raw_message.device_id,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get server metrics."""
        uptime = None
        if self._metrics["start_time"]:
            uptime = (datetime.now(timezone.utc) - self._metrics["start_time"]).total_seconds()

        return {
            **self._metrics,
            "uptime_seconds": uptime,
            "running": self._running,
        }


def create_app():
    """Create FastAPI application for HTTP endpoints."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from polyglotlink.api.routes.v1 import router as v1_router

    app = FastAPI(
        title="PolyglotLink",
        description="Semantic API Translator for IoT Device Ecosystems",
        version="0.1.0",
    )

    # Server instance (will be set on startup)
    app.state.server = None

    # Mount v1 API routes
    app.include_router(v1_router, prefix="/api/v1", tags=["v1"])

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/metrics")
    async def metrics():
        if app.state.server:
            return app.state.server.get_metrics()
        return {"status": "not started"}

    @app.get("/ready")
    async def ready():
        if app.state.server and app.state.server._running:
            return {"status": "ready"}
        return JSONResponse(
            status_code=503,
            content={"status": "not ready"},
        )

    return app


async def run_server(
    host: str = "0.0.0.0",  # noqa: ARG001  # nosec B104 - binding to all interfaces is intentional
    port: int = 8080,  # noqa: ARG001
    workers: int = 1,  # noqa: ARG001
    reload: bool = False,  # noqa: ARG001
    **kwargs,
) -> None:
    """Run the PolyglotLink server."""
    server = PolyglotLinkServer(**kwargs)

    # Set up signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()

        # Keep running until stopped
        while server._running:
            await asyncio.sleep(1)

    finally:
        await server.stop()
