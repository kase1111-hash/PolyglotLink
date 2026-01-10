"""
PolyglotLink Protocol Listener Module

This module provides unified ingestion of IoT device messages across multiple
protocols: MQTT, CoAP, Modbus, OPC-UA, HTTP, and WebSocket.
"""

import asyncio
import base64
import json
import re
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import structlog

from polyglotlink.models.schemas import (
    CoAPConfig,
    HTTPConfig,
    ModbusConfig,
    MQTTConfig,
    OPCUAConfig,
    PayloadEncoding,
    Protocol,
    ProtocolListenerConfig,
    RawMessage,
    WebSocketConfig,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Payload Encoding Detection
# ============================================================================

def is_likely_protobuf(payload: bytes) -> bool:
    """Heuristic check for protobuf format based on field tags."""
    if len(payload) < 2:
        return False
    # Protobuf messages typically start with a field tag (varint)
    first_byte = payload[0]
    wire_type = first_byte & 0x07
    return wire_type in (0, 1, 2, 5)  # Valid wire types


def is_likely_csv(payload: bytes) -> bool:
    """Check if payload looks like CSV data."""
    try:
        text = payload.decode('utf-8')
        lines = text.strip().split('\n')
        if len(lines) < 1:
            return False
        # Check for consistent delimiter usage
        delimiters = [',', ';', '\t']
        for delim in delimiters:
            counts = [line.count(delim) for line in lines]
            if counts[0] > 0 and all(c == counts[0] for c in counts):
                return True
        return False
    except (UnicodeDecodeError, Exception):
        return False


def detect_encoding(payload: bytes) -> PayloadEncoding:
    """
    Detect the encoding/format of raw payload bytes.
    """
    if not payload:
        return PayloadEncoding.BINARY

    # Try JSON
    try:
        json.loads(payload)
        return PayloadEncoding.JSON
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # Try XML
    try:
        ET.fromstring(payload)
        return PayloadEncoding.XML
    except ET.ParseError:
        pass

    # Try CBOR (requires cbor2)
    try:
        import cbor2
        cbor2.loads(payload)
        return PayloadEncoding.CBOR
    except Exception:
        pass

    # Try Protobuf (heuristic)
    if is_likely_protobuf(payload):
        return PayloadEncoding.PROTOBUF

    # Try CSV
    if is_likely_csv(payload):
        return PayloadEncoding.CSV

    # Check for Modbus register format
    if len(payload) % 2 == 0 and len(payload) > 0:
        # Could be Modbus registers (16-bit values)
        return PayloadEncoding.MODBUS_REGISTERS

    # Binary fallback
    return PayloadEncoding.BINARY


def xml_to_dict(element: ET.Element) -> Dict[str, Any]:
    """Convert XML element to dictionary."""
    result: Dict[str, Any] = {}

    # Add attributes
    if element.attrib:
        result['@attributes'] = dict(element.attrib)

    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0:
            return element.text.strip()
        result['#text'] = element.text.strip()

    # Add children
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            # Convert to list if multiple children with same tag
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data

    return result


def csv_to_dict(payload: bytes) -> Dict[str, Any]:
    """Convert CSV payload to dictionary."""
    import csv
    import io

    text = payload.decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if len(rows) == 1:
        return rows[0]
    return {'rows': rows, '_count': len(rows)}


def parse_modbus_registers(payload: bytes) -> Dict[str, Any]:
    """Parse Modbus register data."""
    registers = []
    for i in range(0, len(payload), 2):
        value = int.from_bytes(payload[i:i+2], 'big')
        registers.append(value)
    return {'registers': registers, '_count': len(registers)}


# Encoding parsers registry
ENCODING_PARSERS: Dict[PayloadEncoding, Callable[[bytes], Any]] = {
    PayloadEncoding.JSON: lambda p: json.loads(p),
    PayloadEncoding.XML: lambda p: xml_to_dict(ET.fromstring(p)),
    PayloadEncoding.CSV: csv_to_dict,
    PayloadEncoding.MODBUS_REGISTERS: parse_modbus_registers,
    PayloadEncoding.BINARY: lambda p: {
        'raw_hex': p.hex(),
        'raw_base64': base64.b64encode(p).decode()
    },
}

# Add CBOR parser if available
try:
    import cbor2
    ENCODING_PARSERS[PayloadEncoding.CBOR] = lambda p: cbor2.loads(p)
except ImportError:
    ENCODING_PARSERS[PayloadEncoding.CBOR] = lambda p: {'error': 'cbor2 not installed'}


def generate_uuid() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


def extract_device_id(identifier: str) -> str:
    """
    Extract device ID from topic/path.
    Common patterns: devices/{id}/telemetry, sensors/{id}, etc.
    """
    patterns = [
        r'devices?/([^/]+)',
        r'sensors?/([^/]+)',
        r'things?/([^/]+)',
        r'([^/]+)/telemetry',
        r'([^/]+)/data',
    ]

    for pattern in patterns:
        match = re.search(pattern, identifier, re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback: use the last meaningful segment
    segments = [s for s in identifier.split('/') if s]
    return segments[-1] if segments else 'unknown'


# ============================================================================
# Base Protocol Handler
# ============================================================================

class BaseProtocolHandler(ABC):
    """Abstract base class for protocol handlers."""

    def __init__(self, protocol: Protocol):
        self.protocol = protocol
        self._running = False
        self._message_queue: asyncio.Queue[RawMessage] = asyncio.Queue()

    @abstractmethod
    async def start(self) -> None:
        """Start the protocol handler."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the protocol handler."""
        pass

    async def messages(self) -> AsyncGenerator[RawMessage, None]:
        """Yield messages from the handler."""
        while self._running or not self._message_queue.empty():
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                yield message
            except asyncio.TimeoutError:
                continue

    async def emit_message(self, message: RawMessage) -> None:
        """Add a message to the queue."""
        await self._message_queue.put(message)


# ============================================================================
# MQTT Handler
# ============================================================================

class MQTTHandler(BaseProtocolHandler):
    """MQTT protocol handler."""

    def __init__(self, config: MQTTConfig):
        super().__init__(Protocol.MQTT)
        self.config = config
        self._client = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start MQTT listener."""
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(
                client_id=self.config.client_id,
                protocol=mqtt.MQTTv311
            )

            if self.config.username and self.config.password:
                self._client.username_pw_set(
                    self.config.username,
                    self.config.password
                )

            if self.config.tls_enabled and self.config.ca_cert:
                self._client.tls_set(ca_certs=self.config.ca_cert)

            # Set up callbacks
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            self._running = True
            self._client.connect_async(
                self.config.broker_host,
                self.config.broker_port
            )
            self._client.loop_start()

            logger.info(
                "MQTT handler started",
                broker=self.config.broker_host,
                port=self.config.broker_port
            )
        except Exception as e:
            logger.error("Failed to start MQTT handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop MQTT listener."""
        self._running = False
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        logger.info("MQTT handler stopped")

    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            logger.info("MQTT connected successfully")
            for pattern in self.config.topic_patterns:
                client.subscribe(pattern, qos=self.config.qos)
                logger.info("Subscribed to topic", pattern=pattern)
        else:
            logger.error("MQTT connection failed", rc=rc)

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        try:
            raw = RawMessage(
                message_id=generate_uuid(),
                device_id=extract_device_id(msg.topic),
                protocol=Protocol.MQTT,
                topic=msg.topic,
                payload_raw=msg.payload,
                payload_encoding=detect_encoding(msg.payload),
                qos=msg.qos,
                retained=msg.retain,
                timestamp=datetime.utcnow(),
                metadata={
                    'broker': self.config.broker_host,
                    'topic_pattern': self._get_matching_pattern(msg.topic)
                }
            )

            # Use thread-safe method to add to queue
            asyncio.run_coroutine_threadsafe(
                self.emit_message(raw),
                asyncio.get_event_loop()
            )
        except Exception as e:
            logger.error("Error processing MQTT message", error=str(e))

    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        if rc != 0:
            logger.warning("MQTT unexpected disconnection", rc=rc)

    def _get_matching_pattern(self, topic: str) -> Optional[str]:
        """Find which subscription pattern matched this topic."""
        import fnmatch
        for pattern in self.config.topic_patterns:
            mqtt_pattern = pattern.replace('+', '*').replace('#', '**')
            if fnmatch.fnmatch(topic, mqtt_pattern):
                return pattern
        return None


# ============================================================================
# HTTP Webhook Handler
# ============================================================================

class HTTPHandler(BaseProtocolHandler):
    """HTTP webhook handler using FastAPI."""

    def __init__(self, config: HTTPConfig):
        super().__init__(Protocol.HTTP)
        self.config = config
        self._app = None
        self._server = None

    async def start(self) -> None:
        """Start HTTP server."""
        try:
            from fastapi import FastAPI, Request
            import uvicorn

            self._app = FastAPI(title="PolyglotLink HTTP Ingress")

            @self._app.post(f"{self.config.path_prefix}/{{path:path}}")
            async def ingest(request: Request, path: str = ""):
                body = await request.body()
                full_path = f"{self.config.path_prefix}/{path}" if path else self.config.path_prefix

                raw = RawMessage(
                    message_id=generate_uuid(),
                    device_id=(
                        request.headers.get("X-Device-ID") or
                        extract_device_id(full_path)
                    ),
                    protocol=Protocol.HTTP,
                    topic=full_path,
                    payload_raw=body,
                    payload_encoding=detect_encoding(body),
                    timestamp=datetime.utcnow(),
                    metadata={
                        'method': request.method,
                        'content_type': request.headers.get("Content-Type"),
                        'headers': dict(request.headers),
                        'query_params': dict(request.query_params)
                    }
                )

                await self.emit_message(raw)
                return {"status": "accepted", "message_id": raw.message_id}

            @self._app.get("/health")
            async def health():
                return {"status": "healthy"}

            self._running = True

            config = uvicorn.Config(
                self._app,
                host=self.config.host,
                port=self.config.port,
                log_level="warning"
            )
            self._server = uvicorn.Server(config)

            # Run server in background task
            asyncio.create_task(self._server.serve())

            logger.info(
                "HTTP handler started",
                host=self.config.host,
                port=self.config.port,
                path_prefix=self.config.path_prefix
            )
        except Exception as e:
            logger.error("Failed to start HTTP handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop HTTP server."""
        self._running = False
        if self._server:
            self._server.should_exit = True
        logger.info("HTTP handler stopped")


# ============================================================================
# CoAP Handler
# ============================================================================

class CoAPHandler(BaseProtocolHandler):
    """CoAP protocol handler."""

    def __init__(self, config: CoAPConfig):
        super().__init__(Protocol.COAP)
        self.config = config
        self._context = None

    async def start(self) -> None:
        """Start CoAP server."""
        try:
            import aiocoap
            import aiocoap.resource as resource

            class IngestResource(resource.Resource):
                def __init__(self, handler: 'CoAPHandler'):
                    super().__init__()
                    self.handler = handler

                async def render_post(self, request):
                    path = '/'.join(request.opt.uri_path)

                    raw = RawMessage(
                        message_id=generate_uuid(),
                        device_id=extract_device_id(path),
                        protocol=Protocol.COAP,
                        topic=path,
                        payload_raw=request.payload,
                        payload_encoding=detect_encoding(request.payload),
                        timestamp=datetime.utcnow(),
                        metadata={
                            'coap_type': request.mtype.name if hasattr(request.mtype, 'name') else str(request.mtype),
                            'content_format': request.opt.content_format
                        }
                    )

                    await self.handler.emit_message(raw)

                    return aiocoap.Message(code=aiocoap.CHANGED, payload=b'OK')

            root = resource.Site()
            root.add_resource(('.well-known', 'core'), resource.WKCResource(root.get_resources_as_linkheader))
            root.add_resource(('ingest',), IngestResource(self))

            self._context = await aiocoap.Context.create_server_context(
                root,
                bind=(self.config.host, self.config.port)
            )

            self._running = True

            logger.info(
                "CoAP handler started",
                host=self.config.host,
                port=self.config.port
            )
        except ImportError:
            logger.warning("aiocoap not installed, CoAP handler disabled")
        except Exception as e:
            logger.error("Failed to start CoAP handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop CoAP server."""
        self._running = False
        if self._context:
            await self._context.shutdown()
        logger.info("CoAP handler stopped")


# ============================================================================
# Modbus Handler
# ============================================================================

class ModbusHandler(BaseProtocolHandler):
    """Modbus TCP polling handler."""

    def __init__(self, config: ModbusConfig):
        super().__init__(Protocol.MODBUS)
        self.config = config
        self._client = None
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start Modbus polling."""
        try:
            from pymodbus.client import ModbusTcpClient

            self._client = ModbusTcpClient(
                self.config.host,
                port=self.config.port
            )

            if not self._client.connect():
                raise ConnectionError(f"Failed to connect to Modbus at {self.config.host}:{self.config.port}")

            self._running = True
            self._poll_task = asyncio.create_task(self._poll_loop())

            logger.info(
                "Modbus handler started",
                host=self.config.host,
                port=self.config.port
            )
        except ImportError:
            logger.warning("pymodbus not installed, Modbus handler disabled")
        except Exception as e:
            logger.error("Failed to start Modbus handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop Modbus polling."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._client:
            self._client.close()
        logger.info("Modbus handler stopped")

    async def _poll_loop(self) -> None:
        """Poll Modbus devices at configured interval."""
        while self._running:
            for device in self.config.devices:
                for block in device.register_blocks:
                    try:
                        result = self._client.read_holding_registers(
                            address=block.get('start', 0),
                            count=block.get('count', 1),
                            slave=device.slave_id
                        )

                        if not result.isError():
                            # Encode registers as bytes
                            payload = b''.join(
                                r.to_bytes(2, 'big') for r in result.registers
                            )

                            raw = RawMessage(
                                message_id=generate_uuid(),
                                device_id=device.device_id,
                                protocol=Protocol.MODBUS,
                                topic=f"modbus/{device.slave_id}/{block.get('start', 0)}",
                                payload_raw=payload,
                                payload_encoding=PayloadEncoding.MODBUS_REGISTERS,
                                timestamp=datetime.utcnow(),
                                metadata={
                                    'slave_id': device.slave_id,
                                    'register_start': block.get('start', 0),
                                    'register_count': block.get('count', 1),
                                    'register_type': block.get('type', 'holding')
                                }
                            )

                            await self.emit_message(raw)
                    except Exception as e:
                        logger.error(
                            "Modbus poll error",
                            device=device.device_id,
                            error=str(e)
                        )

            await asyncio.sleep(self.config.poll_interval_seconds)


# ============================================================================
# OPC-UA Handler
# ============================================================================

class OPCUAHandler(BaseProtocolHandler):
    """OPC-UA subscription handler."""

    def __init__(self, config: OPCUAConfig):
        super().__init__(Protocol.OPCUA)
        self.config = config
        self._client = None
        self._subscription = None

    async def start(self) -> None:
        """Start OPC-UA client."""
        try:
            from asyncua import Client

            self._client = Client(self.config.endpoint_url)

            if self.config.security_policy:
                await self._client.set_security_string(self.config.security_policy)

            await self._client.connect()

            # Create subscription
            handler = self._create_handler()
            self._subscription = await self._client.create_subscription(
                period=self.config.subscription_interval_ms,
                handler=handler
            )

            # Subscribe to monitored nodes
            for node_id in self.config.monitored_nodes:
                node = self._client.get_node(node_id)
                await self._subscription.subscribe_data_change(node)

            self._running = True

            logger.info(
                "OPC-UA handler started",
                endpoint=self.config.endpoint_url,
                nodes=len(self.config.monitored_nodes)
            )
        except ImportError:
            logger.warning("asyncua not installed, OPC-UA handler disabled")
        except Exception as e:
            logger.error("Failed to start OPC-UA handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop OPC-UA client."""
        self._running = False
        if self._subscription:
            await self._subscription.delete()
        if self._client:
            await self._client.disconnect()
        logger.info("OPC-UA handler stopped")

    def _create_handler(self):
        """Create subscription handler class."""
        parent = self

        class SubscriptionHandler:
            async def datachange_notification(self, node, val, data):
                try:
                    # Serialize the value
                    if hasattr(val, '__dict__'):
                        payload = json.dumps(val.__dict__, default=str).encode()
                    else:
                        payload = json.dumps({'value': val}, default=str).encode()

                    raw = RawMessage(
                        message_id=generate_uuid(),
                        device_id=extract_device_id(str(node)),
                        protocol=Protocol.OPCUA,
                        topic=str(node),
                        payload_raw=payload,
                        payload_encoding=PayloadEncoding.JSON,
                        timestamp=data.monitored_item.Value.SourceTimestamp or datetime.utcnow(),
                        metadata={
                            'node_id': str(node),
                            'status_code': str(data.monitored_item.Value.StatusCode),
                            'server_timestamp': str(data.monitored_item.Value.ServerTimestamp)
                        }
                    )

                    await parent.emit_message(raw)
                except Exception as e:
                    logger.error("OPC-UA notification error", error=str(e))

        return SubscriptionHandler()


# ============================================================================
# WebSocket Handler
# ============================================================================

class WebSocketHandler(BaseProtocolHandler):
    """WebSocket server handler."""

    def __init__(self, config: WebSocketConfig):
        super().__init__(Protocol.WEBSOCKET)
        self.config = config
        self._server = None
        self._connections: set = set()

    async def start(self) -> None:
        """Start WebSocket server."""
        try:
            import websockets

            async def handler(websocket, path):
                self._connections.add(websocket)
                try:
                    async for message in websocket:
                        if isinstance(message, str):
                            payload = message.encode()
                        else:
                            payload = message

                        raw = RawMessage(
                            message_id=generate_uuid(),
                            device_id=extract_device_id(path),
                            protocol=Protocol.WEBSOCKET,
                            topic=path,
                            payload_raw=payload,
                            payload_encoding=detect_encoding(payload),
                            timestamp=datetime.utcnow(),
                            metadata={
                                'remote_address': str(websocket.remote_address),
                                'path': path
                            }
                        )

                        await self.emit_message(raw)
                finally:
                    self._connections.discard(websocket)

            self._server = await websockets.serve(
                handler,
                self.config.host,
                self.config.port
            )

            self._running = True

            logger.info(
                "WebSocket handler started",
                host=self.config.host,
                port=self.config.port
            )
        except ImportError:
            logger.warning("websockets not installed, WebSocket handler disabled")
        except Exception as e:
            logger.error("Failed to start WebSocket handler", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        # Close all connections
        for ws in self._connections:
            await ws.close()
        logger.info("WebSocket handler stopped")


# ============================================================================
# Protocol Listener Manager
# ============================================================================

class ProtocolListener:
    """
    Manages all protocol handlers and provides unified message stream.
    """

    def __init__(self, config: ProtocolListenerConfig):
        self.config = config
        self._handlers: List[BaseProtocolHandler] = []
        self._running = False

    async def start_listeners(self) -> None:
        """Initialize and start all configured protocol handlers."""
        if self.config.mqtt.enabled:
            handler = MQTTHandler(self.config.mqtt)
            await handler.start()
            self._handlers.append(handler)

        if self.config.http.enabled:
            handler = HTTPHandler(self.config.http)
            await handler.start()
            self._handlers.append(handler)

        if self.config.coap.enabled:
            handler = CoAPHandler(self.config.coap)
            await handler.start()
            self._handlers.append(handler)

        if self.config.modbus.enabled:
            handler = ModbusHandler(self.config.modbus)
            await handler.start()
            self._handlers.append(handler)

        if self.config.opcua.enabled:
            handler = OPCUAHandler(self.config.opcua)
            await handler.start()
            self._handlers.append(handler)

        if self.config.websocket.enabled:
            handler = WebSocketHandler(self.config.websocket)
            await handler.start()
            self._handlers.append(handler)

        self._running = True
        logger.info(
            "Protocol listener started",
            handlers=len(self._handlers)
        )

    async def stop_listeners(self) -> None:
        """Stop all protocol handlers."""
        self._running = False
        for handler in self._handlers:
            await handler.stop()
        self._handlers.clear()
        logger.info("Protocol listener stopped")

    async def messages(self) -> AsyncGenerator[RawMessage, None]:
        """
        Yield messages from all handlers.
        Merges streams from all active protocol handlers.
        """
        async def handler_messages(handler: BaseProtocolHandler):
            async for msg in handler.messages():
                yield msg

        # Create tasks for all handlers
        queues = [handler._message_queue for handler in self._handlers]

        while self._running or any(not q.empty() for q in queues):
            for handler in self._handlers:
                try:
                    message = handler._message_queue.get_nowait()
                    yield message
                except asyncio.QueueEmpty:
                    continue

            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

    def get_handler(self, protocol: Protocol) -> Optional[BaseProtocolHandler]:
        """Get handler for specific protocol."""
        for handler in self._handlers:
            if handler.protocol == protocol:
                return handler
        return None
