# PolyglotLink — Semantic API Translator for IoT Device Ecosystems
## Technical Specification v1.0

**Tagline:** Unifying fragmented IoT data into a single semantic stream.

---

## 1. System Overview

### 1.1 Purpose

PolyglotLink transforms heterogeneous IoT device payloads—across protocols (MQTT, CoAP, Modbus, OPC-UA, HTTP), formats (JSON, XML, CSV, binary), and schemas—into normalized, semantically enriched JSON objects. It uses LLMs as real-time semantic translators, dynamically learning new device schemas without manual adapter development.

### 1.2 Core Value Proposition

| Challenge | Traditional Approach | PolyglotLink Approach |
|-----------|---------------------|----------------------|
| Protocol diversity | Manual adapter per protocol | Unified listener abstraction |
| Schema variation | Static ETL mappings | LLM-inferred semantic mapping |
| New device types | Engineering effort (days/weeks) | Auto-learned (< 1 minute) |
| Ontology drift | Manual maintenance | Self-evolving ontology registry |

### 1.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              POLYGLOTLINK CORE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         INGRESS LAYER                                   │   │
│  │                                                                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │  MQTT   │  │  CoAP   │  │ Modbus  │  │ OPC-UA  │  │  HTTP   │      │   │
│  │  │Listener │  │Listener │  │Listener │  │Listener │  │Listener │      │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │   │
│  │       └───────────┬┴───────────┴┬───────────┴───────────┬┘            │   │
│  │                   ▼             ▼                       ▼              │   │
│  │              ┌─────────────────────────────────────────────┐           │   │
│  │              │           PROTOCOL LISTENER                 │           │   │
│  │              │         protocol_listener.py                │           │   │
│  │              │    [Unified ingestion + metadata capture]   │           │   │
│  │              └─────────────────────┬───────────────────────┘           │   │
│  └────────────────────────────────────┼────────────────────────────────────┘   │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         SCHEMA EXTRACTION                               │   │
│  │                       schema_extractor.py                               │   │
│  │         [Parse structure → Detect types → Generate schema summary]      │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     SEMANTIC TRANSLATION (DLP)                          │   │
│  │                   semantic_translator_agent.py                          │   │
│  │                                                                         │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │   │
│  │  │  LLM Mapper  │◀──▶│  Embedding   │◀──▶│   Ontology   │              │   │
│  │  │  (GPT-5 /    │    │   Cache      │    │   Registry   │              │   │
│  │  │   Local)     │    │  (Weaviate)  │    │   (Neo4j)    │              │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘              │   │
│  │         [Infer semantics → Resolve mappings → Learn new schemas]        │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      NORMALIZATION ENGINE                               │   │
│  │                     normalization_engine.py                             │   │
│  │     [Validate → Convert units → Enrich metadata → Emit standard JSON]   │   │
│  └─────────────────────────────────┬───────────────────────────────────────┘   │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         OUTPUT BROKER                                   │   │
│  │                        output_broker.py                                 │   │
│  │                                                                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │  Kafka  │  │  MQTT   │  │  REST   │  │Websocket│  │ JSON-LD │      │   │
│  │  │Publisher│  │Publisher│  │   API   │  │  Stream │  │ Export  │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           INFRASTRUCTURE                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │    Redis     │  │   Weaviate   │  │    Neo4j     │  │  TimescaleDB │        │
│  │   (Cache)    │  │  (Vectors)   │  │  (Ontology)  │  │   (Metrics)  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Data Flow Summary

| Stage | Component | Input | Output | Latency Target |
|-------|-----------|-------|--------|----------------|
| 1 | Protocol Listener | Raw device message | RawMessage object | < 5ms |
| 2 | Schema Extractor | RawMessage | ExtractedSchema | < 10ms |
| 3 | Semantic Translator | ExtractedSchema | SemanticMapping | < 150ms |
| 4 | Normalization Engine | SemanticMapping | NormalizedMessage | < 20ms |
| 5 | Output Broker | NormalizedMessage | Published event | < 15ms |
| **Total** | End-to-end | Raw payload | Normalized JSON | **< 200ms** |

---

## 2. Module Specifications

### 2.1 Protocol Listener Module

**File:** `modules/protocol_listener.py`

#### 2.1.1 Responsibilities

| Function | Description |
|----------|-------------|
| `start_listeners()` | Initialize all configured protocol handlers |
| `mqtt_handler()` | Subscribe to MQTT topics, receive messages |
| `coap_handler()` | Listen for CoAP requests |
| `modbus_handler()` | Poll Modbus registers |
| `opcua_handler()` | Subscribe to OPC-UA nodes |
| `http_handler()` | Receive HTTP POST webhooks |
| `websocket_handler()` | Maintain WebSocket connections |
| `extract_metadata()` | Parse protocol-specific metadata |
| `emit_raw_message()` | Push to processing pipeline |

#### 2.1.2 Protocol Handlers

```python
# MQTT Handler
ASYNC FUNCTION mqtt_handler(config: MQTTConfig) -> AsyncGenerator[RawMessage]:
    client = mqtt.Client(client_id=config.client_id)
    client.username_pw_set(config.username, config.password)
    
    IF config.tls_enabled:
        client.tls_set(ca_certs=config.ca_cert)
    
    AWAIT client.connect(config.broker_host, config.broker_port)
    
    FOR topic_pattern IN config.topic_patterns:
        client.subscribe(topic_pattern, qos=config.qos)
    
    ASYNC FOR message IN client.messages():
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id=extract_device_id(message.topic),
            protocol="MQTT",
            topic=message.topic,
            payload_raw=message.payload,
            payload_encoding=detect_encoding(message.payload),
            qos=message.qos,
            retained=message.retain,
            timestamp=datetime.utcnow(),
            metadata={
                "broker": config.broker_host,
                "topic_pattern": get_matching_pattern(message.topic)
            }
        )
        YIELD raw


# CoAP Handler
ASYNC FUNCTION coap_handler(config: CoAPConfig) -> AsyncGenerator[RawMessage]:
    protocol = await aiocoap.Context.create_server_context(
        bind=(config.host, config.port)
    )
    
    ASYNC FOR request IN protocol.requests():
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id=extract_device_id(request.opt.uri_path),
            protocol="CoAP",
            topic="/".join(request.opt.uri_path),
            payload_raw=request.payload,
            payload_encoding=detect_encoding(request.payload),
            timestamp=datetime.utcnow(),
            metadata={
                "coap_type": request.mtype.name,
                "content_format": request.opt.content_format
            }
        )
        YIELD raw


# Modbus Handler
ASYNC FUNCTION modbus_handler(config: ModbusConfig) -> AsyncGenerator[RawMessage]:
    client = ModbusTcpClient(config.host, port=config.port)
    
    WHILE True:
        FOR device IN config.devices:
            FOR register_block IN device.register_blocks:
                result = client.read_holding_registers(
                    address=register_block.start,
                    count=register_block.count,
                    slave=device.slave_id
                )
                
                IF NOT result.isError():
                    raw = RawMessage(
                        message_id=generate_uuid(),
                        device_id=device.device_id,
                        protocol="Modbus",
                        topic=f"modbus/{device.slave_id}/{register_block.start}",
                        payload_raw=encode_registers(result.registers),
                        payload_encoding="modbus_registers",
                        timestamp=datetime.utcnow(),
                        metadata={
                            "slave_id": device.slave_id,
                            "register_start": register_block.start,
                            "register_count": register_block.count,
                            "register_type": register_block.type
                        }
                    )
                    YIELD raw
        
        AWAIT asyncio.sleep(config.poll_interval_seconds)


# OPC-UA Handler
ASYNC FUNCTION opcua_handler(config: OPCUAConfig) -> AsyncGenerator[RawMessage]:
    client = Client(config.endpoint_url)
    
    IF config.security_policy:
        client.set_security_string(config.security_policy)
    
    AWAIT client.connect()
    
    # Create subscription
    subscription = AWAIT client.create_subscription(
        period=config.subscription_interval_ms,
        handler=OPCUAHandler()
    )
    
    FOR node_id IN config.monitored_nodes:
        node = client.get_node(node_id)
        AWAIT subscription.subscribe_data_change(node)
    
    ASYNC FOR data_change IN subscription.data_changes():
        raw = RawMessage(
            message_id=generate_uuid(),
            device_id=extract_device_from_node(data_change.node_id),
            protocol="OPC-UA",
            topic=str(data_change.node_id),
            payload_raw=serialize_variant(data_change.value),
            payload_encoding="opcua_variant",
            timestamp=data_change.source_timestamp or datetime.utcnow(),
            metadata={
                "node_id": str(data_change.node_id),
                "status_code": data_change.status_code.name,
                "server_timestamp": data_change.server_timestamp.isoformat()
            }
        )
        YIELD raw


# HTTP Webhook Handler
ASYNC FUNCTION http_handler(request: Request) -> RawMessage:
    body = AWAIT request.body()
    
    raw = RawMessage(
        message_id=generate_uuid(),
        device_id=request.headers.get("X-Device-ID") or extract_from_path(request.url.path),
        protocol="HTTP",
        topic=request.url.path,
        payload_raw=body,
        payload_encoding=detect_encoding(body),
        timestamp=datetime.utcnow(),
        metadata={
            "method": request.method,
            "content_type": request.headers.get("Content-Type"),
            "headers": dict(request.headers)
        }
    )
    
    RETURN raw
```

#### 2.1.3 Payload Encoding Detection

```python
FUNCTION detect_encoding(payload: bytes) -> str:
    """
    Detect the encoding/format of raw payload bytes.
    """
    # Try JSON
    TRY:
        json.loads(payload)
        RETURN "json"
    EXCEPT: PASS
    
    # Try XML
    TRY:
        ET.fromstring(payload)
        RETURN "xml"
    EXCEPT: PASS
    
    # Try CBOR
    TRY:
        cbor2.loads(payload)
        RETURN "cbor"
    EXCEPT: PASS
    
    # Try Protobuf (heuristic: starts with field tags)
    IF is_likely_protobuf(payload):
        RETURN "protobuf"
    
    # Try CSV (has delimiters and newlines)
    IF is_likely_csv(payload):
        RETURN "csv"
    
    # Check for Modbus register format
    IF len(payload) % 2 == 0 AND all(is_printable_or_null(b) FOR b IN payload):
        RETURN "modbus_registers"
    
    # Binary fallback
    RETURN "binary"


ENCODING_PARSERS = {
    "json": lambda p: json.loads(p),
    "xml": lambda p: xml_to_dict(ET.fromstring(p)),
    "cbor": lambda p: cbor2.loads(p),
    "csv": lambda p: csv_to_dict(p),
    "modbus_registers": lambda p: parse_modbus_registers(p),
    "binary": lambda p: {"raw_hex": p.hex(), "raw_base64": base64.b64encode(p).decode()}
}
```

#### 2.1.4 Output Schema

```python
RawMessage = {
    "message_id": str,              # UUID
    "device_id": str,               # Extracted device identifier
    "protocol": str,                # "MQTT" | "CoAP" | "Modbus" | "OPC-UA" | "HTTP" | "WebSocket"
    "topic": str,                   # Topic/path/node identifier
    "payload_raw": bytes,           # Original payload
    "payload_encoding": str,        # Detected format
    "timestamp": datetime,          # Receive timestamp
    "qos": Optional[int],           # MQTT QoS level
    "retained": Optional[bool],     # MQTT retained flag
    "metadata": Dict[str, Any]      # Protocol-specific metadata
}
```

#### 2.1.5 Configuration

```python
ProtocolListenerConfig = {
    "mqtt": {
        "enabled": True,
        "broker_host": "iot-broker.company.com",
        "broker_port": 1883,
        "client_id": "polyglotlink-listener",
        "username": str,
        "password": str,
        "tls_enabled": False,
        "ca_cert": Optional[str],
        "topic_patterns": ["sensors/#", "devices/+/telemetry"],
        "qos": 1
    },
    "coap": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 5683
    },
    "modbus": {
        "enabled": False,
        "host": "192.168.1.100",
        "port": 502,
        "poll_interval_seconds": 5,
        "devices": [
            {
                "device_id": "plc-001",
                "slave_id": 1,
                "register_blocks": [
                    {"start": 0, "count": 10, "type": "holding"},
                    {"start": 100, "count": 5, "type": "input"}
                ]
            }
        ]
    },
    "opcua": {
        "enabled": False,
        "endpoint_url": "opc.tcp://localhost:4840",
        "security_policy": None,
        "subscription_interval_ms": 1000,
        "monitored_nodes": ["ns=2;s=Temperature", "ns=2;s=Pressure"]
    },
    "http": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8080,
        "path_prefix": "/ingest"
    },
    "websocket": {
        "enabled": False,
        "host": "0.0.0.0",
        "port": 8081
    }
}
```

#### 2.1.6 Dependencies

```
paho-mqtt>=1.6
aiocoap>=0.4
pymodbus>=3.5
asyncua>=1.0
fastapi>=0.109
uvicorn>=0.27
websockets>=12.0
cbor2>=5.5
```

---

### 2.2 Schema Extractor Module

**File:** `modules/schema_extractor.py`

#### 2.2.1 Responsibilities

| Function | Description |
|----------|-------------|
| `extract_schema()` | Parse payload into field structure |
| `detect_field_types()` | Infer data types for each field |
| `detect_units()` | Heuristic unit detection from field names/values |
| `flatten_nested()` | Convert nested structures to flat field list |
| `generate_schema_hash()` | Create fingerprint for schema caching |
| `check_schema_cache()` | Look up known schema mappings |

#### 2.2.2 Extraction Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RawMessage    │───▶│  Decode Payload │───▶│  Flatten        │
│   (bytes)       │    │  (JSON/XML/etc) │    │  Nested Fields  │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Schema Hash    │◀───│  Detect Types   │◀───│  Extract Fields │
│  + Cache Check  │    │  + Units        │    │  (key, value)   │
└────────┬────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ ExtractedSchema │
│ (ready for LLM) │
└─────────────────┘
```

#### 2.2.3 Schema Extraction Logic

```python
FUNCTION extract_schema(raw: RawMessage) -> ExtractedSchema:
    # Decode payload based on detected encoding
    parser = ENCODING_PARSERS.get(raw.payload_encoding)
    IF NOT parser:
        RAISE UnsupportedEncodingError(raw.payload_encoding)
    
    decoded = parser(raw.payload_raw)
    
    # Flatten nested structures
    flat_fields = flatten_dict(decoded, separator=".")
    
    # Extract field information
    fields = []
    FOR key, value IN flat_fields.items():
        field = ExtractedField(
            key=key,
            original_key=key,
            value=value,
            value_type=detect_type(value),
            inferred_unit=infer_unit_from_key(key),
            inferred_semantic=infer_semantic_hint(key, value),
            is_timestamp=is_timestamp_field(key, value),
            is_identifier=is_identifier_field(key, value)
        )
        fields.append(field)
    
    # Generate schema fingerprint
    schema_signature = generate_schema_hash(fields)
    
    # Check cache for known mapping
    cached_mapping = check_schema_cache(schema_signature)
    
    RETURN ExtractedSchema(
        message_id=raw.message_id,
        device_id=raw.device_id,
        protocol=raw.protocol,
        topic=raw.topic,
        fields=fields,
        schema_signature=schema_signature,
        cached_mapping=cached_mapping,
        payload_decoded=decoded,
        extracted_at=datetime.utcnow()
    )


FUNCTION flatten_dict(d: Dict, parent_key: str = "", separator: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dictionary into dot-notation keys.
    {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []
    FOR key, value IN d.items():
        new_key = f"{parent_key}{separator}{key}" IF parent_key ELSE key
        
        IF isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        ELIF isinstance(value, list):
            # Handle arrays
            IF len(value) > 0 AND isinstance(value[0], dict):
                # Array of objects - flatten first element as template
                items.extend(flatten_dict(value[0], f"{new_key}[0]", separator).items())
                items.append((f"{new_key}._count", len(value)))
            ELSE:
                items.append((new_key, value))
        ELSE:
            items.append((new_key, value))
    
    RETURN dict(items)
```

#### 2.2.4 Type and Unit Detection

```python
FUNCTION detect_type(value: Any) -> str:
    IF value IS None:
        RETURN "null"
    IF isinstance(value, bool):
        RETURN "boolean"
    IF isinstance(value, int):
        RETURN "integer"
    IF isinstance(value, float):
        RETURN "float"
    IF isinstance(value, str):
        # Check for ISO datetime
        IF is_iso_datetime(value):
            RETURN "datetime"
        # Check for numeric string
        IF is_numeric_string(value):
            RETURN "numeric_string"
        RETURN "string"
    IF isinstance(value, list):
        RETURN "array"
    IF isinstance(value, dict):
        RETURN "object"
    RETURN "unknown"


# Unit inference patterns
UNIT_PATTERNS = {
    # Temperature
    r"(temp|temperature).*(_c|_celsius|_centigrade)$": "celsius",
    r"(temp|temperature).*(_f|_fahrenheit)$": "fahrenheit",
    r"(temp|temperature).*(_k|_kelvin)$": "kelvin",
    r"^(tmp|temp|t)$": "celsius",  # Default assumption
    
    # Humidity
    r"(hum|humidity|rh).*(_pct|_percent|%)?$": "percent",
    
    # Pressure
    r"(press|pressure).*(_pa|_pascal)$": "pascal",
    r"(press|pressure).*(_bar)$": "bar",
    r"(press|pressure).*(_psi)$": "psi",
    r"(press|pressure).*(_hpa)$": "hectopascal",
    
    # Voltage/Current
    r"(volt|voltage|v)$": "volt",
    r"(current|amp|ampere|i)$": "ampere",
    r"(power|watt|w)$": "watt",
    
    # Speed/Flow
    r"(speed|velocity).*(_mps|_ms)$": "meters_per_second",
    r"(speed|velocity).*(_kmh|_kph)$": "kilometers_per_hour",
    r"(flow).*(_lpm|_lm)$": "liters_per_minute",
    
    # Distance/Length
    r"(dist|distance|length).*(_m|_meter)$": "meter",
    r"(dist|distance|length).*(_cm)$": "centimeter",
    r"(dist|distance|length).*(_mm)$": "millimeter",
    
    # Time
    r"(time|duration).*(_s|_sec|_seconds)$": "seconds",
    r"(time|duration).*(_ms|_milliseconds)$": "milliseconds",
    
    # Mass
    r"(weight|mass).*(_kg|_kilogram)$": "kilogram",
    r"(weight|mass).*(_g|_gram)$": "gram",
    
    # Percentage
    r".*(_pct|_percent|%)$": "percent",
    r"^(pct|percent|ratio)$": "percent"
}


FUNCTION infer_unit_from_key(key: str) -> Optional[str]:
    key_lower = key.lower()
    
    FOR pattern, unit IN UNIT_PATTERNS.items():
        IF re.match(pattern, key_lower):
            RETURN unit
    
    RETURN None


# Semantic hint inference
SEMANTIC_HINTS = {
    # Environment
    r"(temp|temperature|tmp)": "temperature",
    r"(hum|humidity|rh)": "humidity",
    r"(press|pressure|baro)": "pressure",
    r"(light|lux|luminosity)": "illuminance",
    r"(co2|carbon_dioxide)": "co2_concentration",
    r"(pm25|pm2\.5)": "particulate_matter_2_5",
    r"(noise|sound|db|decibel)": "noise_level",
    
    # Motion/Position
    r"(accel|acceleration)": "acceleration",
    r"(gyro|angular)": "angular_velocity",
    r"(lat|latitude)": "latitude",
    r"(lon|lng|longitude)": "longitude",
    r"(alt|altitude|elevation)": "altitude",
    r"(speed|velocity)": "speed",
    
    # Electrical
    r"(volt|voltage)": "voltage",
    r"(current|ampere|amp)": "current",
    r"(power|watt)": "power",
    r"(energy|kwh)": "energy",
    r"(freq|frequency|hz)": "frequency",
    
    # State/Status
    r"(state|status)": "state",
    r"(online|connected|alive)": "connectivity",
    r"(battery|batt)": "battery_level",
    r"(rssi|signal)": "signal_strength",
    
    # Identifiers
    r"(id|uuid|guid)$": "identifier",
    r"(name|label)$": "name",
    r"(timestamp|time|ts|datetime)": "timestamp"
}


FUNCTION infer_semantic_hint(key: str, value: Any) -> Optional[str]:
    key_lower = key.lower()
    
    FOR pattern, semantic IN SEMANTIC_HINTS.items():
        IF re.search(pattern, key_lower):
            RETURN semantic
    
    RETURN None
```

#### 2.2.5 Schema Caching

```python
FUNCTION generate_schema_hash(fields: List[ExtractedField]) -> str:
    """
    Generate a fingerprint for the schema based on field names and types.
    Used for caching semantic mappings.
    """
    # Create canonical representation
    canonical = sorted([
        f"{f.key}:{f.value_type}"
        FOR f IN fields
        IF NOT f.is_timestamp AND NOT f.is_identifier
    ])
    
    schema_string = "|".join(canonical)
    RETURN hashlib.sha256(schema_string.encode()).hexdigest()[:16]


FUNCTION check_schema_cache(schema_signature: str) -> Optional[CachedMapping]:
    """
    Check if we've seen this schema before and have a cached mapping.
    """
    cache_key = f"schema:{schema_signature}"
    cached = redis_client.get(cache_key)
    
    IF cached:
        mapping = CachedMapping.parse_raw(cached)
        # Update hit count for analytics
        redis_client.hincrby(f"schema_stats:{schema_signature}", "hits", 1)
        RETURN mapping
    
    RETURN None


FUNCTION cache_schema_mapping(schema_signature: str, mapping: SemanticMapping) -> None:
    """
    Store a learned schema mapping for future reuse.
    """
    cache_key = f"schema:{schema_signature}"
    cached = CachedMapping(
        schema_signature=schema_signature,
        field_mappings=mapping.field_mappings,
        confidence=mapping.confidence,
        created_at=datetime.utcnow(),
        source="llm" IF mapping.llm_generated ELSE "manual"
    )
    
    redis_client.setex(
        cache_key,
        timedelta(days=SchemaConfig.cache_ttl_days),
        cached.json()
    )
```

#### 2.2.6 Output Schema

```python
ExtractedField = {
    "key": str,                     # Flattened key (e.g., "sensor.temp")
    "original_key": str,            # Original key name
    "value": Any,                   # Actual value
    "value_type": str,              # Detected type
    "inferred_unit": Optional[str], # Heuristic unit guess
    "inferred_semantic": Optional[str],  # Semantic category hint
    "is_timestamp": bool,           # Likely a timestamp field
    "is_identifier": bool           # Likely an ID field
}

ExtractedSchema = {
    "message_id": str,
    "device_id": str,
    "protocol": str,
    "topic": str,
    "fields": List[ExtractedField],
    "schema_signature": str,        # Hash for caching
    "cached_mapping": Optional[CachedMapping],
    "payload_decoded": Dict,        # Original decoded payload
    "extracted_at": datetime
}

CachedMapping = {
    "schema_signature": str,
    "field_mappings": List[FieldMapping],
    "confidence": float,
    "created_at": datetime,
    "source": str,                  # "llm" | "manual" | "learned"
    "hit_count": int
}
```

#### 2.2.7 Configuration

```python
SchemaExtractorConfig = {
    "max_nesting_depth": 10,
    "max_array_sample_size": 5,
    "cache_ttl_days": 30,
    "enable_unit_inference": True,
    "enable_semantic_hints": True,
    "flatten_arrays": True,
    "preserve_null_fields": False
}
```

---

### 2.3 Semantic Translator Agent Module

**File:** `modules/semantic_translator_agent.py`

#### 2.3.1 Responsibilities

| Function | Description |
|----------|-------------|
| `translate_schema()` | Map extracted fields to ontology concepts |
| `query_llm()` | Request semantic interpretation from LLM |
| `resolve_with_embeddings()` | Use vector similarity for known concepts |
| `validate_mapping()` | Check mapping against ontology constraints |
| `learn_new_concept()` | Add new mappings to ontology |
| `explain_translation()` | Generate human-readable reasoning |

#### 2.3.2 Translation Pipeline

```
┌─────────────────┐
│ ExtractedSchema │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    CACHE CHECK                              │
│         Is schema_signature in cache?                       │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     [CACHE HIT]                   [CACHE MISS]
          │                             │
          ▼                             ▼
┌─────────────────┐         ┌─────────────────────────────────┐
│  Apply Cached   │         │      EMBEDDING LOOKUP           │
│  Mapping        │         │  Query Weaviate for similar     │
└────────┬────────┘         │  field names/descriptions       │
         │                  └─────────────┬───────────────────┘
         │                                │
         │                   ┌────────────┴────────────┐
         │                   ▼                        ▼
         │            [HIGH SIMILARITY]        [LOW SIMILARITY]
         │                   │                        │
         │                   ▼                        ▼
         │         ┌─────────────────┐      ┌─────────────────┐
         │         │  Use Embedding  │      │   CALL LLM      │
         │         │  Match          │      │   for semantic  │
         │         └────────┬────────┘      │   inference     │
         │                  │               └────────┬────────┘
         │                  │                        │
         │                  └────────────┬───────────┘
         │                               │
         │                               ▼
         │                  ┌─────────────────────────┐
         │                  │   VALIDATE MAPPING      │
         │                  │   against ontology      │
         │                  └─────────────┬───────────┘
         │                                │
         │                                ▼
         │                  ┌─────────────────────────┐
         │                  │   CACHE NEW MAPPING     │
         │                  └─────────────┬───────────┘
         │                                │
         └────────────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────────┐
                  │    SemanticMapping      │
                  └─────────────────────────┘
```

#### 2.3.3 LLM Translation Prompt

```python
TRANSLATION_PROMPT_TEMPLATE = """
You are an IoT data semantics expert. Your task is to map raw device fields to standardized ontology concepts.

## Device Context
- Protocol: {protocol}
- Topic/Path: {topic}
- Device ID: {device_id}

## Raw Fields
{fields_table}

## Available Ontology Concepts
{ontology_concepts}

## Task
For each raw field, determine:
1. The standardized concept it maps to (from ontology or suggest new)
2. The standardized field name
3. Unit conversion if needed (source unit → target unit)
4. Confidence score (0.0-1.0)

## Output Format (JSON only)
{{
  "mappings": [
    {{
      "source_field": "original field name",
      "target_concept": "ontology concept ID",
      "target_field": "standardized field name",
      "source_unit": "detected unit or null",
      "target_unit": "standard unit",
      "conversion_formula": "formula if conversion needed, else null",
      "confidence": 0.95,
      "reasoning": "brief explanation"
    }}
  ],
  "device_context": "inferred device type/purpose",
  "suggested_new_concepts": [
    {{
      "concept_id": "suggested_concept_name",
      "description": "what this concept represents",
      "unit": "standard unit",
      "datatype": "float|integer|string|boolean"
    }}
  ]
}}

Generate the mapping:
"""


FUNCTION build_fields_table(fields: List[ExtractedField]) -> str:
    rows = ["| Field | Value | Type | Inferred Unit | Semantic Hint |"]
    rows.append("|-------|-------|------|---------------|---------------|")
    
    FOR field IN fields:
        rows.append(
            f"| {field.key} | {truncate(str(field.value), 30)} | "
            f"{field.value_type} | {field.inferred_unit or '-'} | "
            f"{field.inferred_semantic or '-'} |"
        )
    
    RETURN "\n".join(rows)


FUNCTION build_ontology_context(fields: List[ExtractedField]) -> str:
    """
    Fetch relevant ontology concepts based on field hints.
    """
    concepts = []
    
    FOR field IN fields:
        IF field.inferred_semantic:
            # Query ontology for related concepts
            related = ontology_registry.get_concepts_by_category(field.inferred_semantic)
            concepts.extend(related)
    
    # Also include commonly used concepts
    concepts.extend(ontology_registry.get_popular_concepts(limit=20))
    
    # Deduplicate
    concepts = list({c.concept_id: c FOR c IN concepts}.values())
    
    # Format for prompt
    lines = []
    FOR concept IN concepts[:50]:  # Limit to prevent prompt overflow
        lines.append(
            f"- {concept.concept_id}: {concept.description} "
            f"(unit: {concept.unit}, type: {concept.datatype})"
        )
    
    RETURN "\n".join(lines)
```

#### 2.3.4 Embedding-Based Resolution

```python
FUNCTION resolve_with_embeddings(
    field: ExtractedField,
    threshold: float = 0.85
) -> Optional[FieldMapping]:
    """
    Attempt to resolve field mapping using vector similarity.
    """
    # Generate embedding for field name + context
    query_text = f"{field.key} {field.inferred_semantic or ''} {field.inferred_unit or ''}"
    query_embedding = embedding_model.encode(query_text)
    
    # Search vector store
    results = weaviate_client.query.get(
        "OntologyConcept",
        ["concept_id", "canonical_name", "unit", "datatype", "aliases"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": threshold
    }).with_limit(5).do()
    
    IF results["data"]["Get"]["OntologyConcept"]:
        best_match = results["data"]["Get"]["OntologyConcept"][0]
        certainty = results["data"]["Get"]["OntologyConcept"][0]["_additional"]["certainty"]
        
        IF certainty >= threshold:
            RETURN FieldMapping(
                source_field=field.key,
                target_concept=best_match["concept_id"],
                target_field=best_match["canonical_name"],
                source_unit=field.inferred_unit,
                target_unit=best_match["unit"],
                conversion_formula=get_unit_conversion(field.inferred_unit, best_match["unit"]),
                confidence=certainty,
                resolution_method="embedding"
            )
    
    RETURN None


FUNCTION translate_schema(schema: ExtractedSchema) -> SemanticMapping:
    """
    Main translation function. Uses cache, embeddings, then LLM fallback.
    """
    # Check cache first
    IF schema.cached_mapping:
        RETURN apply_cached_mapping(schema, schema.cached_mapping)
    
    mappings = []
    fields_needing_llm = []
    
    # Try embedding resolution for each field
    FOR field IN schema.fields:
        # Skip timestamps and identifiers
        IF field.is_timestamp OR field.is_identifier:
            mappings.append(create_passthrough_mapping(field))
            CONTINUE
        
        embedding_result = resolve_with_embeddings(field)
        
        IF embedding_result:
            mappings.append(embedding_result)
        ELSE:
            fields_needing_llm.append(field)
    
    # Call LLM for unresolved fields
    IF fields_needing_llm:
        llm_mappings = call_llm_for_mapping(schema, fields_needing_llm)
        mappings.extend(llm_mappings.mappings)
        
        # Learn new concepts if suggested
        FOR new_concept IN llm_mappings.suggested_new_concepts:
            ontology_registry.add_concept(new_concept)
    
    # Validate all mappings
    validated_mappings = []
    FOR mapping IN mappings:
        IF validate_mapping(mapping):
            validated_mappings.append(mapping)
        ELSE:
            log_warning(f"Invalid mapping rejected: {mapping}")
            validated_mappings.append(create_fallback_mapping(mapping.source_field))
    
    result = SemanticMapping(
        message_id=schema.message_id,
        device_id=schema.device_id,
        schema_signature=schema.schema_signature,
        field_mappings=validated_mappings,
        device_context=llm_mappings.device_context IF fields_needing_llm ELSE None,
        confidence=compute_overall_confidence(validated_mappings),
        llm_generated=len(fields_needing_llm) > 0,
        translated_at=datetime.utcnow()
    )
    
    # Cache the mapping
    cache_schema_mapping(schema.schema_signature, result)
    
    RETURN result
```

#### 2.3.5 Output Schema

```python
FieldMapping = {
    "source_field": str,            # Original field name
    "target_concept": str,          # Ontology concept ID
    "target_field": str,            # Standardized field name
    "source_unit": Optional[str],   # Detected source unit
    "target_unit": Optional[str],   # Target standard unit
    "conversion_formula": Optional[str],  # Unit conversion (e.g., "value * 1.8 + 32")
    "confidence": float,            # 0.0 - 1.0
    "resolution_method": str,       # "cache" | "embedding" | "llm" | "passthrough"
    "reasoning": Optional[str]      # LLM explanation
}

SemanticMapping = {
    "message_id": str,
    "device_id": str,
    "schema_signature": str,
    "field_mappings": List[FieldMapping],
    "device_context": Optional[str],  # Inferred device type
    "confidence": float,            # Overall mapping confidence
    "llm_generated": bool,          # True if LLM was used
    "translated_at": datetime
}

SuggestedConcept = {
    "concept_id": str,
    "description": str,
    "unit": str,
    "datatype": str,
    "aliases": List[str]
}
```

#### 2.3.6 Configuration

```python
SemanticTranslatorConfig = {
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    "llm_temperature": 0.1,
    "llm_max_tokens": 2000,
    "embedding_model": "text-embedding-3-large",
    "embedding_threshold": 0.85,
    "max_llm_retries": 3,
    "timeout_seconds": 30,
    "enable_concept_learning": True,
    "min_confidence_threshold": 0.6,
    "cache_embeddings": True,
    "vector_store": "weaviate"
}
```

---

### 2.4 Ontology Registry Module

**File:** `modules/ontology_registry.py`

#### 2.4.1 Responsibilities

| Function | Description |
|----------|-------------|
| `get_concept()` | Retrieve concept by ID |
| `search_concepts()` | Search concepts by name, alias, or category |
| `add_concept()` | Register new concept |
| `update_concept()` | Modify existing concept |
| `get_aliases()` | Get all aliases for a concept |
| `add_alias()` | Register new alias for existing concept |
| `get_unit_conversions()` | Retrieve conversion formulas |
| `export_ontology()` | Export as JSON-LD or RDF |

#### 2.4.2 Ontology Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONTOLOGY GRAPH                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │   CATEGORY      │         │    CATEGORY     │               │
│  │   Environment   │         │    Electrical   │               │
│  └────────┬────────┘         └────────┬────────┘               │
│           │                           │                         │
│     ┌─────┴─────┐              ┌──────┴──────┐                 │
│     ▼           ▼              ▼             ▼                 │
│ ┌───────┐  ┌───────┐      ┌───────┐    ┌───────┐              │
│ │CONCEPT│  │CONCEPT│      │CONCEPT│    │CONCEPT│              │
│ │ temp_ │  │ humid_│      │voltage│    │current│              │
│ │celsius│  │percent│      │       │    │       │              │
│ └───┬───┘  └───┬───┘      └───┬───┘    └───┬───┘              │
│     │          │              │            │                   │
│  ┌──┴───┐  ┌───┴──┐       ┌───┴──┐     ┌───┴──┐               │
│  │ALIAS │  │ALIAS │       │ALIAS │     │ALIAS │               │
│  │ tmp  │  │ hum  │       │ volt │     │ amp  │               │
│  │ temp │  │ rh   │       │ v    │     │ i    │               │
│  │ t    │  │humidity│     │ vdc  │     │ mA   │               │
│  └──────┘  └──────┘       └──────┘     └──────┘               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    UNIT CONVERSIONS                     │   │
│  │  celsius ←→ fahrenheit: (c * 9/5) + 32                  │   │
│  │  celsius ←→ kelvin: c + 273.15                          │   │
│  │  pascal ←→ bar: p / 100000                              │   │
│  │  percent ←→ ratio: p / 100                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 Neo4j Schema

```cypher
// Category nodes
CREATE CONSTRAINT category_id IF NOT EXISTS
FOR (c:Category) REQUIRE c.id IS UNIQUE;

// Concept nodes
CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

// Create categories
CREATE (env:Category {id: "environment", name: "Environment Sensors"})
CREATE (elec:Category {id: "electrical", name: "Electrical Measurements"})
CREATE (motion:Category {id: "motion", name: "Motion & Position"})
CREATE (state:Category {id: "state", name: "State & Status"})

// Create concepts
CREATE (temp_c:Concept {
    concept_id: "temperature_celsius",
    canonical_name: "temperature_celsius",
    description: "Temperature measurement in degrees Celsius",
    unit: "celsius",
    datatype: "float",
    min_value: -273.15,
    max_value: 1000,
    created_at: datetime()
})

CREATE (humid:Concept {
    concept_id: "humidity_percent",
    canonical_name: "humidity_percent",
    description: "Relative humidity as percentage",
    unit: "percent",
    datatype: "float",
    min_value: 0,
    max_value: 100,
    created_at: datetime()
})

// Create relationships
CREATE (temp_c)-[:BELONGS_TO]->(env)
CREATE (humid)-[:BELONGS_TO]->(env)

// Create aliases
CREATE (temp_c)-[:HAS_ALIAS]->(:Alias {name: "tmp", weight: 0.9})
CREATE (temp_c)-[:HAS_ALIAS]->(:Alias {name: "temp", weight: 0.95})
CREATE (temp_c)-[:HAS_ALIAS]->(:Alias {name: "temperature", weight: 1.0})
CREATE (temp_c)-[:HAS_ALIAS]->(:Alias {name: "t", weight: 0.5})

// Unit conversions
CREATE (celsius:Unit {name: "celsius", symbol: "°C"})
CREATE (fahrenheit:Unit {name: "fahrenheit", symbol: "°F"})
CREATE (kelvin:Unit {name: "kelvin", symbol: "K"})

CREATE (celsius)-[:CONVERTS_TO {formula: "(value * 9/5) + 32"}]->(fahrenheit)
CREATE (fahrenheit)-[:CONVERTS_TO {formula: "(value - 32) * 5/9"}]->(celsius)
CREATE (celsius)-[:CONVERTS_TO {formula: "value + 273.15"}]->(kelvin)
CREATE (kelvin)-[:CONVERTS_TO {formula: "value - 273.15"}]->(celsius)
```

#### 2.4.4 Registry Operations

```python
CLASS OntologyRegistry:
    
    FUNCTION __init__(self, neo4j_driver, weaviate_client):
        self.neo4j = neo4j_driver
        self.weaviate = weaviate_client
    
    FUNCTION get_concept(self, concept_id: str) -> Optional[Concept]:
        query = """
        MATCH (c:Concept {concept_id: $concept_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(cat:Category)
        OPTIONAL MATCH (c)-[:HAS_ALIAS]->(a:Alias)
        RETURN c, cat, collect(a.name) as aliases
        """
        result = self.neo4j.run(query, concept_id=concept_id)
        record = result.single()
        
        IF record:
            RETURN Concept(
                concept_id=record["c"]["concept_id"],
                canonical_name=record["c"]["canonical_name"],
                description=record["c"]["description"],
                unit=record["c"]["unit"],
                datatype=record["c"]["datatype"],
                category=record["cat"]["id"] IF record["cat"] ELSE None,
                aliases=record["aliases"],
                min_value=record["c"].get("min_value"),
                max_value=record["c"].get("max_value")
            )
        
        RETURN None
    
    FUNCTION search_concepts(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Concept]:
        # First try exact alias match
        cypher = """
        MATCH (c:Concept)-[:HAS_ALIAS]->(a:Alias)
        WHERE toLower(a.name) = toLower($query)
        RETURN c, a.weight as weight
        ORDER BY weight DESC
        LIMIT $limit
        """
        results = self.neo4j.run(cypher, query=query, limit=limit)
        
        IF results:
            RETURN [self._record_to_concept(r) FOR r IN results]
        
        # Fall back to fuzzy search via embeddings
        RETURN self._embedding_search(query, category, limit)
    
    FUNCTION add_concept(self, concept: SuggestedConcept) -> Concept:
        # Validate uniqueness
        IF self.get_concept(concept.concept_id):
            RAISE ConceptExistsError(concept.concept_id)
        
        # Create in Neo4j
        cypher = """
        CREATE (c:Concept {
            concept_id: $concept_id,
            canonical_name: $concept_id,
            description: $description,
            unit: $unit,
            datatype: $datatype,
            created_at: datetime(),
            source: 'auto_learned'
        })
        RETURN c
        """
        self.neo4j.run(cypher, **concept.dict())
        
        # Add aliases
        FOR alias IN concept.aliases:
            self.add_alias(concept.concept_id, alias)
        
        # Add to vector store
        embedding = self._generate_concept_embedding(concept)
        self.weaviate.data_object.create(
            class_name="OntologyConcept",
            data_object={
                "concept_id": concept.concept_id,
                "canonical_name": concept.concept_id,
                "description": concept.description,
                "unit": concept.unit,
                "datatype": concept.datatype,
                "aliases": concept.aliases
            },
            vector=embedding
        )
        
        log_info(f"Added new concept: {concept.concept_id}")
        RETURN self.get_concept(concept.concept_id)
    
    FUNCTION get_unit_conversion(
        self,
        from_unit: str,
        to_unit: str
    ) -> Optional[str]:
        IF from_unit == to_unit:
            RETURN None
        
        query = """
        MATCH (from:Unit {name: $from_unit})-[conv:CONVERTS_TO]->(to:Unit {name: $to_unit})
        RETURN conv.formula as formula
        """
        result = self.neo4j.run(query, from_unit=from_unit, to_unit=to_unit)
        record = result.single()
        
        IF record:
            RETURN record["formula"]
        
        # Try transitive conversion (e.g., fahrenheit → celsius → kelvin)
        RETURN self._find_transitive_conversion(from_unit, to_unit)
    
    FUNCTION export_ontology(self, format: str = "json-ld") -> str:
        IF format == "json-ld":
            RETURN self._export_jsonld()
        ELIF format == "rdf":
            RETURN self._export_rdf()
        ELIF format == "json":
            RETURN self._export_json()
        ELSE:
            RAISE UnsupportedFormatError(format)
```

#### 2.4.5 Output Schema

```python
Concept = {
    "concept_id": str,              # Unique identifier
    "canonical_name": str,          # Standardized field name
    "description": str,             # Human-readable description
    "unit": str,                    # Standard unit
    "datatype": str,                # "float" | "integer" | "string" | "boolean"
    "category": Optional[str],      # Category ID
    "aliases": List[str],           # Known aliases
    "min_value": Optional[float],   # Valid range minimum
    "max_value": Optional[float],   # Valid range maximum
    "created_at": datetime,
    "source": str                   # "manual" | "auto_learned"
}

UnitConversion = {
    "from_unit": str,
    "to_unit": str,
    "formula": str,                 # Python expression with 'value' variable
    "bidirectional": bool
}
```

#### 2.4.6 Seed Ontology

```python
SEED_ONTOLOGY = {
    "categories": [
        {"id": "environment", "name": "Environment Sensors"},
        {"id": "electrical", "name": "Electrical Measurements"},
        {"id": "motion", "name": "Motion & Position"},
        {"id": "state", "name": "State & Status"},
        {"id": "industrial", "name": "Industrial Process"},
        {"id": "building", "name": "Building Automation"}
    ],
    "concepts": [
        # Environment
        {
            "concept_id": "temperature_celsius",
            "description": "Temperature in degrees Celsius",
            "unit": "celsius",
            "datatype": "float",
            "category": "environment",
            "aliases": ["tmp", "temp", "temperature", "t", "thermo"]
        },
        {
            "concept_id": "humidity_percent",
            "description": "Relative humidity percentage",
            "unit": "percent",
            "datatype": "float",
            "category": "environment",
            "aliases": ["hum", "humidity", "rh", "relative_humidity"]
        },
        {
            "concept_id": "pressure_pascal",
            "description": "Atmospheric or process pressure",
            "unit": "pascal",
            "datatype": "float",
            "category": "environment",
            "aliases": ["press", "pressure", "baro", "atm"]
        },
        {
            "concept_id": "co2_ppm",
            "description": "Carbon dioxide concentration",
            "unit": "ppm",
            "datatype": "float",
            "category": "environment",
            "aliases": ["co2", "carbon_dioxide", "co2_level"]
        },
        # Electrical
        {
            "concept_id": "voltage_volt",
            "description": "Electrical voltage",
            "unit": "volt",
            "datatype": "float",
            "category": "electrical",
            "aliases": ["volt", "voltage", "v", "vdc", "vac"]
        },
        {
            "concept_id": "current_ampere",
            "description": "Electrical current",
            "unit": "ampere",
            "datatype": "float",
            "category": "electrical",
            "aliases": ["current", "amp", "ampere", "i", "amps"]
        },
        {
            "concept_id": "power_watt",
            "description": "Electrical power",
            "unit": "watt",
            "datatype": "float",
            "category": "electrical",
            "aliases": ["power", "watt", "w", "watts"]
        },
        # Motion
        {
            "concept_id": "latitude_degrees",
            "description": "Geographic latitude",
            "unit": "degrees",
            "datatype": "float",
            "category": "motion",
            "aliases": ["lat", "latitude"]
        },
        {
            "concept_id": "longitude_degrees",
            "description": "Geographic longitude",
            "unit": "degrees",
            "datatype": "float",
            "category": "motion",
            "aliases": ["lon", "lng", "longitude"]
        },
        {
            "concept_id": "speed_mps",
            "description": "Speed in meters per second",
            "unit": "meters_per_second",
            "datatype": "float",
            "category": "motion",
            "aliases": ["speed", "velocity", "spd"]
        },
        # State
        {
            "concept_id": "battery_percent",
            "description": "Battery charge level",
            "unit": "percent",
            "datatype": "float",
            "category": "state",
            "aliases": ["battery", "batt", "bat_level", "charge"]
        },
        {
            "concept_id": "signal_rssi",
            "description": "Signal strength (RSSI)",
            "unit": "dbm",
            "datatype": "integer",
            "category": "state",
            "aliases": ["rssi", "signal", "signal_strength"]
        },
        {
            "concept_id": "connectivity_status",
            "description": "Online/offline status",
            "unit": "boolean",
            "datatype": "boolean",
            "category": "state",
            "aliases": ["online", "connected", "alive", "status"]
        }
    ],
    "unit_conversions": [
        {"from": "celsius", "to": "fahrenheit", "formula": "(value * 9/5) + 32"},
        {"from": "fahrenheit", "to": "celsius", "formula": "(value - 32) * 5/9"},
        {"from": "celsius", "to": "kelvin", "formula": "value + 273.15"},
        {"from": "kelvin", "to": "celsius", "formula": "value - 273.15"},
        {"from": "pascal", "to": "bar", "formula": "value / 100000"},
        {"from": "bar", "to": "pascal", "formula": "value * 100000"},
        {"from": "pascal", "to": "psi", "formula": "value * 0.000145038"},
        {"from": "psi", "to": "pascal", "formula": "value / 0.000145038"},
        {"from": "meters_per_second", "to": "kilometers_per_hour", "formula": "value * 3.6"},
        {"from": "kilometers_per_hour", "to": "meters_per_second", "formula": "value / 3.6"},
        {"from": "percent", "to": "ratio", "formula": "value / 100"},
        {"from": "ratio", "to": "percent", "formula": "value * 100"}
    ]
}
```

#### 2.4.7 Configuration

```python
OntologyRegistryConfig = {
    "neo4j_uri": "neo4j://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": str,
    "weaviate_url": "http://localhost:8080",
    "enable_auto_learning": True,
    "min_alias_weight": 0.5,
    "export_format": "json-ld"
}
```

---

### 2.5 Normalization Engine Module

**File:** `modules/normalization_engine.py`

#### 2.5.1 Responsibilities

| Function | Description |
|----------|-------------|
| `normalize_message()` | Apply semantic mapping to raw values |
| `convert_units()` | Execute unit conversion formulas |
| `validate_values()` | Check values against ontology constraints |
| `enrich_metadata()` | Add device/location context |
| `enforce_types()` | Cast values to expected types |
| `handle_nulls()` | Apply null value strategies |

#### 2.5.2 Normalization Pipeline

```
┌─────────────────┐    ┌─────────────────┐
│ ExtractedSchema │    │ SemanticMapping │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   MERGE SCHEMA +      │
        │   MAPPING             │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   UNIT CONVERSION     │
        │   Apply formulas      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   TYPE ENFORCEMENT    │
        │   Cast to expected    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   VALUE VALIDATION    │
        │   Check min/max/enum  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   METADATA ENRICHMENT │
        │   Add context         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  NormalizedMessage    │
        └───────────────────────┘
```

#### 2.5.3 Normalization Logic

```python
FUNCTION normalize_message(
    schema: ExtractedSchema,
    mapping: SemanticMapping
) -> NormalizedMessage:
    normalized_fields = {}
    validation_errors = []
    conversions_applied = []
    
    # Build lookup from source field to mapping
    mapping_lookup = {m.source_field: m FOR m IN mapping.field_mappings}
    
    FOR field IN schema.fields:
        field_mapping = mapping_lookup.get(field.key)
        
        IF NOT field_mapping:
            # Passthrough unmapped fields with prefix
            normalized_fields[f"_unmapped.{field.key}"] = field.value
            CONTINUE
        
        value = field.value
        target_field = field_mapping.target_field
        
        # Step 1: Unit conversion
        IF field_mapping.conversion_formula:
            TRY:
                value = apply_conversion(value, field_mapping.conversion_formula)
                conversions_applied.append(ConversionRecord(
                    field=field.key,
                    from_unit=field_mapping.source_unit,
                    to_unit=field_mapping.target_unit,
                    original_value=field.value,
                    converted_value=value
                ))
            EXCEPT ConversionError as e:
                validation_errors.append(ValidationError(
                    field=field.key,
                    error="conversion_failed",
                    details=str(e)
                ))
                CONTINUE
        
        # Step 2: Type enforcement
        concept = ontology_registry.get_concept(field_mapping.target_concept)
        IF concept:
            TRY:
                value = enforce_type(value, concept.datatype)
            EXCEPT TypeError as e:
                validation_errors.append(ValidationError(
                    field=field.key,
                    error="type_mismatch",
                    expected=concept.datatype,
                    actual=type(value).__name__
                ))
                CONTINUE
            
            # Step 3: Value validation
            IF NOT validate_value(value, concept):
                validation_errors.append(ValidationError(
                    field=field.key,
                    error="out_of_range",
                    value=value,
                    min=concept.min_value,
                    max=concept.max_value
                ))
                # Still include value but flag it
                normalized_fields[f"_invalid.{target_field}"] = value
                CONTINUE
        
        normalized_fields[target_field] = value
    
    # Step 4: Metadata enrichment
    metadata = enrich_metadata(schema, mapping)
    
    RETURN NormalizedMessage(
        message_id=schema.message_id,
        device_id=schema.device_id,
        timestamp=extract_timestamp(schema) or datetime.utcnow(),
        data=normalized_fields,
        metadata=metadata,
        context=mapping.device_context,
        schema_signature=schema.schema_signature,
        confidence=mapping.confidence,
        conversions=conversions_applied,
        validation_errors=validation_errors,
        normalized_at=datetime.utcnow()
    )


FUNCTION apply_conversion(value: Any, formula: str) -> float:
    """
    Safely execute unit conversion formula.
    """
    IF value IS None:
        RETURN None
    
    # Validate formula (only allow safe operations)
    allowed_tokens = set("value0123456789.+-*/() ")
    IF NOT all(c IN allowed_tokens FOR c IN formula):
        RAISE UnsafeFormulaError(formula)
    
    # Execute
    result = eval(formula, {"__builtins__": {}}, {"value": float(value)})
    RETURN round(result, 6)  # Limit precision


FUNCTION enforce_type(value: Any, target_type: str) -> Any:
    IF value IS None:
        RETURN None
    
    IF target_type == "float":
        RETURN float(value)
    ELIF target_type == "integer":
        RETURN int(float(value))
    ELIF target_type == "string":
        RETURN str(value)
    ELIF target_type == "boolean":
        IF isinstance(value, bool):
            RETURN value
        IF isinstance(value, str):
            RETURN value.lower() IN ("true", "1", "yes", "on")
        RETURN bool(value)
    
    RETURN value


FUNCTION validate_value(value: Any, concept: Concept) -> bool:
    IF value IS None:
        RETURN True  # Nulls handled separately
    
    IF concept.min_value IS NOT None AND value < concept.min_value:
        RETURN False
    
    IF concept.max_value IS NOT None AND value > concept.max_value:
        RETURN False
    
    RETURN True


FUNCTION enrich_metadata(schema: ExtractedSchema, mapping: SemanticMapping) -> Dict:
    metadata = {
        "source_protocol": schema.protocol,
        "source_topic": schema.topic,
        "translation_confidence": mapping.confidence,
        "schema_signature": schema.schema_signature
    }
    
    # Add device registry info if available
    device_info = device_registry.get(schema.device_id)
    IF device_info:
        metadata["device_type"] = device_info.type
        metadata["device_name"] = device_info.name
        metadata["location"] = device_info.location
        metadata["tags"] = device_info.tags
    
    # Add inferred context
    IF mapping.device_context:
        metadata["inferred_context"] = mapping.device_context
    
    RETURN metadata
```

#### 2.5.4 Output Schema

```python
NormalizedMessage = {
    "message_id": str,
    "device_id": str,
    "timestamp": datetime,          # Extracted or receive time
    "data": Dict[str, Any],         # Normalized field values
    "metadata": Dict[str, Any],     # Enriched metadata
    "context": Optional[str],       # Inferred device context
    "schema_signature": str,
    "confidence": float,
    "conversions": List[ConversionRecord],
    "validation_errors": List[ValidationError],
    "normalized_at": datetime
}

ConversionRecord = {
    "field": str,
    "from_unit": str,
    "to_unit": str,
    "original_value": Any,
    "converted_value": Any
}

ValidationError = {
    "field": str,
    "error": str,                   # "conversion_failed" | "type_mismatch" | "out_of_range"
    "details": Optional[str],
    "expected": Optional[str],
    "actual": Optional[str],
    "value": Optional[Any],
    "min": Optional[float],
    "max": Optional[float]
}
```

#### 2.5.5 Configuration

```python
NormalizationConfig = {
    "null_strategy": "preserve",    # "preserve" | "omit" | "default"
    "default_values": {
        "float": 0.0,
        "integer": 0,
        "string": "",
        "boolean": False
    },
    "precision": {
        "float": 6,
        "temperature": 2,
        "percentage": 1
    },
    "include_unmapped": True,
    "include_invalid": True,
    "timestamp_field_names": ["timestamp", "ts", "time", "datetime", "created_at"],
    "device_registry_enabled": True
}
```

---

### 2.6 Output Broker Module

**File:** `modules/output_broker.py`

#### 2.6.1 Responsibilities

| Function | Description |
|----------|-------------|
| `publish()` | Route normalized message to configured outputs |
| `kafka_publish()` | Send to Kafka topic |
| `mqtt_publish()` | Send to MQTT topic |
| `http_post()` | POST to REST endpoint |
| `websocket_broadcast()` | Push to WebSocket subscribers |
| `store_timeseries()` | Write to TimescaleDB |
| `export_jsonld()` | Generate JSON-LD for semantic web |

#### 2.6.2 Output Routing

```
┌─────────────────────┐
│  NormalizedMessage  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT ROUTER                              │
│                                                                 │
│   ┌─────────────────┐    ┌─────────────────┐                   │
│   │  Topic Mapper   │───▶│  Format         │                   │
│   │  (device_id →   │    │  Transformer    │                   │
│   │   output topic) │    │  (JSON/JSON-LD) │                   │
│   └─────────────────┘    └─────────────────┘                   │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┬──────────────┐
         ▼                    ▼                    ▼              ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐
│     KAFKA       │  │      MQTT       │  │     REST        │  │   WS    │
│   Publisher     │  │    Publisher    │  │    Webhook      │  │ Broadcast│
│                 │  │                 │  │                 │  │         │
│ Topic:          │  │ Topic:          │  │ POST to         │  │ Channel:│
│ iot.normalized. │  │ normalized/     │  │ configured      │  │ devices/│
│ {device_type}   │  │ {device_id}     │  │ endpoints       │  │ {id}    │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────┘
         │                    │                    │              │
         └────────────────────┼────────────────────┴──────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   TimescaleDB   │
                    │   (optional)    │
                    │   time-series   │
                    │   storage       │
                    └─────────────────┘
```

#### 2.6.3 Publishing Logic

```python
CLASS OutputBroker:
    
    FUNCTION __init__(self, config: OutputBrokerConfig):
        self.config = config
        self.kafka_producer = None
        self.mqtt_client = None
        self.websocket_manager = None
        self.http_session = None
        
        self._init_outputs()
    
    ASYNC FUNCTION publish(self, message: NormalizedMessage) -> PublishResult:
        results = []
        
        # Determine output topics based on device/context
        routing = self._compute_routing(message)
        
        # Format message for output
        formatted = self._format_message(message, routing.format)
        
        # Publish to each enabled output
        IF self.config.kafka.enabled AND routing.kafka_topic:
            result = AWAIT self._kafka_publish(routing.kafka_topic, formatted)
            results.append(("kafka", result))
        
        IF self.config.mqtt.enabled AND routing.mqtt_topic:
            result = AWAIT self._mqtt_publish(routing.mqtt_topic, formatted)
            results.append(("mqtt", result))
        
        IF self.config.http.enabled AND routing.http_endpoints:
            FOR endpoint IN routing.http_endpoints:
                result = AWAIT self._http_post(endpoint, formatted)
                results.append(("http", result))
        
        IF self.config.websocket.enabled:
            result = AWAIT self._websocket_broadcast(message.device_id, formatted)
            results.append(("websocket", result))
        
        IF self.config.timescale.enabled:
            result = AWAIT self._store_timeseries(message)
            results.append(("timescale", result))
        
        RETURN PublishResult(
            message_id=message.message_id,
            outputs=results,
            published_at=datetime.utcnow()
        )
    
    ASYNC FUNCTION _kafka_publish(self, topic: str, payload: bytes) -> bool:
        TRY:
            future = self.kafka_producer.send(
                topic,
                value=payload,
                timestamp_ms=int(datetime.utcnow().timestamp() * 1000)
            )
            AWAIT asyncio.wrap_future(future)
            RETURN True
        EXCEPT KafkaError as e:
            log_error(f"Kafka publish failed: {e}")
            RETURN False
    
    ASYNC FUNCTION _mqtt_publish(self, topic: str, payload: bytes) -> bool:
        TRY:
            result = self.mqtt_client.publish(
                topic,
                payload,
                qos=self.config.mqtt.qos,
                retain=self.config.mqtt.retain
            )
            AWAIT asyncio.wrap_future(result)
            RETURN True
        EXCEPT MQTTError as e:
            log_error(f"MQTT publish failed: {e}")
            RETURN False
    
    ASYNC FUNCTION _http_post(self, endpoint: str, payload: bytes) -> bool:
        TRY:
            async with self.http_session.post(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.config.http.timeout_seconds
            ) as response:
                RETURN response.status < 400
        EXCEPT Exception as e:
            log_error(f"HTTP post failed: {e}")
            RETURN False
    
    ASYNC FUNCTION _websocket_broadcast(self, device_id: str, payload: bytes) -> bool:
        channel = f"devices/{device_id}"
        AWAIT self.websocket_manager.broadcast(channel, payload)
        RETURN True
    
    ASYNC FUNCTION _store_timeseries(self, message: NormalizedMessage) -> bool:
        """
        Store normalized data in TimescaleDB for analytics.
        """
        TRY:
            # Extract numeric fields for time-series
            metrics = []
            FOR field, value IN message.data.items():
                IF isinstance(value, (int, float)) AND NOT field.startswith("_"):
                    metrics.append({
                        "time": message.timestamp,
                        "device_id": message.device_id,
                        "metric": field,
                        "value": value
                    })
            
            IF metrics:
                AWAIT self.timescale_pool.executemany(
                    """
                    INSERT INTO iot_metrics (time, device_id, metric, value)
                    VALUES ($1, $2, $3, $4)
                    """,
                    [(m["time"], m["device_id"], m["metric"], m["value"]) FOR m IN metrics]
                )
            
            RETURN True
        EXCEPT Exception as e:
            log_error(f"TimescaleDB insert failed: {e}")
            RETURN False
    
    FUNCTION _format_message(self, message: NormalizedMessage, format: str) -> bytes:
        IF format == "json":
            RETURN json.dumps(message.dict(), default=str).encode()
        ELIF format == "json-ld":
            RETURN self._to_jsonld(message).encode()
        ELSE:
            RETURN json.dumps(message.dict(), default=str).encode()
    
    FUNCTION _to_jsonld(self, message: NormalizedMessage) -> str:
        """
        Convert to JSON-LD format for semantic web compatibility.
        """
        jsonld = {
            "@context": {
                "@vocab": "https://schema.org/",
                "iot": "https://www.w3.org/2019/wot/td#",
                "sosa": "http://www.w3.org/ns/sosa/"
            },
            "@type": "sosa:Observation",
            "@id": f"urn:iot:observation:{message.message_id}",
            "sosa:madeBySensor": f"urn:iot:device:{message.device_id}",
            "sosa:resultTime": message.timestamp.isoformat(),
            "sosa:hasResult": {}
        }
        
        FOR field, value IN message.data.items():
            IF NOT field.startswith("_"):
                jsonld["sosa:hasResult"][field] = value
        
        RETURN json.dumps(jsonld, indent=2)
    
    FUNCTION _compute_routing(self, message: NormalizedMessage) -> OutputRouting:
        context = message.context or "unknown"
        device_id = message.device_id
        
        RETURN OutputRouting(
            kafka_topic=f"iot.normalized.{context}",
            mqtt_topic=f"normalized/{device_id}",
            http_endpoints=self._get_http_endpoints(context),
            format=self.config.default_format
        )
```

#### 2.6.4 Output Schema

```python
PublishResult = {
    "message_id": str,
    "outputs": List[Tuple[str, bool]],  # (output_type, success)
    "published_at": datetime
}

OutputRouting = {
    "kafka_topic": Optional[str],
    "mqtt_topic": Optional[str],
    "http_endpoints": List[str],
    "websocket_channels": List[str],
    "format": str                   # "json" | "json-ld"
}
```

#### 2.6.5 TimescaleDB Schema

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main metrics table
CREATE TABLE iot_metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('iot_metrics', 'time');

-- Create indexes
CREATE INDEX idx_metrics_device ON iot_metrics (device_id, time DESC);
CREATE INDEX idx_metrics_metric ON iot_metrics (metric, time DESC);

-- Compression policy (compress chunks older than 7 days)
ALTER TABLE iot_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id,metric'
);

SELECT add_compression_policy('iot_metrics', INTERVAL '7 days');

-- Retention policy (drop data older than 90 days)
SELECT add_retention_policy('iot_metrics', INTERVAL '90 days');

-- Continuous aggregate for hourly averages
CREATE MATERIALIZED VIEW iot_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    device_id,
    metric,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    COUNT(*) AS sample_count
FROM iot_metrics
GROUP BY bucket, device_id, metric;

-- Refresh policy
SELECT add_continuous_aggregate_policy('iot_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

#### 2.6.6 Configuration

```python
OutputBrokerConfig = {
    "default_format": "json",
    "kafka": {
        "enabled": True,
        "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
        "topic_prefix": "iot.normalized",
        "compression": "gzip",
        "acks": "all"
    },
    "mqtt": {
        "enabled": True,
        "broker_host": "mqtt-broker.company.com",
        "broker_port": 1883,
        "topic_prefix": "normalized",
        "qos": 1,
        "retain": False
    },
    "http": {
        "enabled": False,
        "endpoints": [
            "https://analytics.company.com/ingest",
            "https://backup.company.com/iot"
        ],
        "timeout_seconds": 10,
        "retry_count": 3
    },
    "websocket": {
        "enabled": True,
        "host": "0.0.0.0",
        "port": 8082
    },
    "timescale": {
        "enabled": True,
        "dsn": "postgresql://iot:password@timescale:5432/iot_data",
        "pool_size": 10
    }
}
```

---

## 3. API Specification

### 3.1 REST Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/ingest` | Receive HTTP device payload | Raw JSON/XML | Normalized JSON |
| POST | `/ingest/batch` | Batch ingest | Array of payloads | Array of results |
| GET | `/devices` | List known devices | - | Device list |
| GET | `/devices/{id}` | Get device info | - | Device details |
| GET | `/devices/{id}/latest` | Latest normalized message | - | NormalizedMessage |
| GET | `/ontology/concepts` | List ontology concepts | `?category=X` | Concept list |
| GET | `/ontology/concepts/{id}` | Get concept details | - | Concept |
| POST | `/ontology/concepts` | Add new concept | Concept JSON | Created concept |
| GET | `/schemas` | List known schemas | - | Schema list |
| GET | `/schemas/{signature}` | Get schema mapping | - | CachedMapping |
| GET | `/metrics` | System metrics | - | Prometheus format |
| GET | `/health` | Health check | - | Status |
| WS | `/ws/devices/{id}` | Stream device data | - | Real-time messages |

### 3.2 WebSocket Protocol

```python
# Client connects to /ws/devices/{device_id}
# Or /ws/devices/* for all devices

# Server sends normalized messages as JSON:
{
    "type": "message",
    "device_id": "A12",
    "timestamp": "2026-01-09T15:30:00Z",
    "data": {
        "temperature_celsius": 23.5,
        "humidity_percent": 46
    }
}

# Client can send subscription updates:
{
    "type": "subscribe",
    "devices": ["A12", "B34"],
    "filters": {
        "metrics": ["temperature_celsius"]
    }
}

# Heartbeat
{
    "type": "ping"
}
# Response
{
    "type": "pong"
}
```

---

## 4. Configuration

### 4.1 Master Configuration File

**File:** `config.yaml`

```yaml
# PolyglotLink Configuration

# Application
app:
  name: "PolyglotLink"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

# Protocol Listeners
protocols:
  mqtt:
    enabled: true
    broker_host: "mqtt://iot-broker.company.com"
    broker_port: 1883
    client_id: "polyglotlink-${HOSTNAME}"
    topic_patterns:
      - "sensors/#"
      - "devices/+/telemetry"
    qos: 1
    
  coap:
    enabled: true
    host: "0.0.0.0"
    port: 5683
    
  modbus:
    enabled: false
    poll_interval_seconds: 5
    devices: []
    
  opcua:
    enabled: false
    endpoint_url: "opc.tcp://localhost:4840"
    
  http:
    enabled: true
    host: "0.0.0.0"
    port: 8080

# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4o"
  embedding_model: "text-embedding-3-large"
  temperature: 0.1
  max_tokens: 2000
  timeout_seconds: 30

# Ontology Registry
ontology:
  neo4j_uri: "neo4j://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "${NEO4J_PASSWORD}"
  enable_auto_learning: true

# Vector Store
vector_store:
  provider: "weaviate"
  url: "http://localhost:8080"
  embedding_threshold: 0.85

# Schema Cache
cache:
  backend: "redis"
  url: "redis://localhost:6379"
  ttl_days: 30

# Output Broker
output:
  default_format: "json"
  kafka:
    enabled: true
    bootstrap_servers:
      - "kafka:9092"
    topic_prefix: "iot.normalized"
  mqtt:
    enabled: true
    broker_host: "mqtt-broker.company.com"
    topic_prefix: "normalized"
  timescale:
    enabled: true
    dsn: "postgresql://iot:${DB_PASSWORD}@timescale:5432/iot_data"

# Metrics & Observability
observability:
  prometheus_enabled: true
  prometheus_port: 9090
  tracing_enabled: false
  sentry_dsn: "${SENTRY_DSN}"
```

---

## 5. Directory Structure

```
polyglotlink/
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration loader
│   └── dependencies.py             # DI container
│
├── modules/
│   ├── __init__.py
│   ├── protocol_listener.py        # Multi-protocol ingestion
│   ├── schema_extractor.py         # Payload parsing
│   ├── semantic_translator.py      # LLM-based translation
│   ├── ontology_registry.py        # Concept management
│   ├── normalization_engine.py     # Value transformation
│   └── output_broker.py            # Multi-output publishing
│
├── protocols/
│   ├── __init__.py
│   ├── mqtt_handler.py
│   ├── coap_handler.py
│   ├── modbus_handler.py
│   ├── opcua_handler.py
│   ├── http_handler.py
│   └── websocket_handler.py
│
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── ingest.py
│   │   ├── devices.py
│   │   ├── ontology.py
│   │   └── schemas.py
│   └── websocket.py
│
├── models/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic models
│   └── database.py                 # SQLAlchemy/TimescaleDB models
│
├── services/
│   ├── __init__.py
│   ├── llm_service.py
│   ├── embedding_service.py
│   └── device_registry.py
│
├── utils/
│   ├── __init__.py
│   ├── encoding_detector.py
│   ├── unit_converter.py
│   └── logging.py
│
├── ontology/
│   ├── seed_data.json              # Initial ontology
│   └── migrations/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_listener.py
│   ├── test_extractor.py
│   ├── test_translator.py
│   ├── test_normalization.py
│   └── fixtures/
│       ├── sample_payloads.json
│       └── expected_outputs.json
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.edge             # Lightweight for edge devices
│   └── docker-compose.yaml
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 6. Testing Strategy

### 6.1 Test Matrix

| Test Type | Tool | Target | Coverage |
|-----------|------|--------|----------|
| Unit | pytest | All modules | ≥ 90% |
| Integration | pytest + docker-compose | End-to-end pipeline | ≥ 80% |
| Protocol | mqtt-spy, coap-client | Protocol handlers | 100% |
| Load | locust | Throughput benchmarks | 500+ msg/s |
| Semantic | Manual + LLM eval | Translation accuracy | ≥ 90% |

### 6.2 Key Test Scenarios

```python
# test_translator.py

def test_temperature_mapping():
    """Test basic temperature field translation."""
    raw = {"tmp": 23.5, "hum": 46}
    result = translate(raw, device_context="thermostat")
    
    assert "temperature_celsius" in result.data
    assert result.data["temperature_celsius"] == 23.5
    assert result.confidence >= 0.9


def test_unit_conversion():
    """Test automatic unit conversion."""
    raw = {"temp_f": 98.6}  # Fahrenheit
    result = translate(raw, target_unit="celsius")
    
    assert abs(result.data["temperature_celsius"] - 37.0) < 0.1


def test_unknown_schema_learning():
    """Test that new schemas trigger LLM and get cached."""
    raw = {"custom_sensor_xyz": 42.0, "weird_metric": "high"}
    
    result1 = translate(raw)
    assert result1.llm_generated == True
    
    result2 = translate(raw)  # Same schema
    assert result2.llm_generated == False  # Should use cache


def test_modbus_register_parsing():
    """Test Modbus register decoding."""
    registers = [0x4248, 0x0000]  # IEEE 754 float: 50.0
    raw = RawMessage(
        protocol="Modbus",
        payload_raw=encode_registers(registers),
        payload_encoding="modbus_registers"
    )
    
    schema = extract_schema(raw)
    assert schema.fields[0].value == 50.0
```

### 6.3 Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Mapping Accuracy | ≥ 90% | Correct semantic matches |
| End-to-end Latency (p99) | < 200ms | Ingest to output |
| Throughput | ≥ 500 msg/s | Messages processed per second |
| Schema Learning Time | < 1 min | New schema to cached mapping |
| Unit Conversion Accuracy | ≥ 95% | Correct conversions |
| Cache Hit Rate | ≥ 80% | After warmup period |

---

## 7. Deployment

### 7.1 Docker Compose (Development)

```yaml
version: "3.9"

services:
  polyglotlink:
    build: .
    ports:
      - "8080:8080"
      - "5683:5683/udp"
      - "8082:8082"
    environment:
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - redis
      - neo4j
      - weaviate
      - kafka
      - timescale

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8081:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  timescale:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: iot
      POSTGRES_PASSWORD: password
      POSTGRES_DB: iot_data
    ports:
      - "5432:5432"

  mosquitto:
    image: eclipse-mosquitto:latest
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
```

### 7.2 Edge Deployment (Lightweight)

```dockerfile
# Dockerfile.edge
FROM python:3.11-slim

# Minimal dependencies for edge
RUN pip install --no-cache-dir \
    paho-mqtt \
    aiocoap \
    fastapi \
    uvicorn \
    httpx \
    redis

WORKDIR /app
COPY modules/ modules/
COPY app/ app/
COPY config.yaml .

# Use local LLM or remote API
ENV LLM_PROVIDER=remote
ENV LLM_ENDPOINT=https://cloud.polyglotlink.com/translate

EXPOSE 8080 5683/udp

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 7.3 Deployment Modes

| Mode | Description | Resources | Use Case |
|------|-------------|-----------|----------|
| **Edge** | Lightweight, remote LLM | 512MB RAM, ARM/x86 | Gateway devices |
| **Standalone** | Full stack, single node | 4GB RAM | Small deployments |
| **Cloud** | Kubernetes, auto-scaling | Variable | Enterprise |
| **Hybrid** | Edge ingestion, cloud translation | Mixed | Large IoT networks |

---

## 8. Dependencies

### 8.1 Python Requirements

```
# Core
python>=3.11
fastapi>=0.109
uvicorn>=0.27
pydantic>=2.5

# Protocols
paho-mqtt>=1.6
aiocoap>=0.4
pymodbus>=3.5
asyncua>=1.0
websockets>=12.0

# Data Formats
cbor2>=5.5
xmltodict>=0.13

# LLM/Embeddings
openai>=1.0
tiktoken>=0.5

# Storage
redis>=5.0
neo4j>=5.0
weaviate-client>=4.0
asyncpg>=0.29
sqlalchemy>=2.0

# Messaging
kafka-python>=2.0
aiokafka>=0.10

# Utilities
structlog>=24.1
prometheus-client>=0.19
httpx>=0.26
pyyaml>=6.0

# Testing
pytest>=7.4
pytest-asyncio>=0.23
locust>=2.20
```

---

## 9. Future Extensions

| Extension | Description | Priority |
|-----------|-------------|----------|
| **Dynamic Few-Shot Learning** | Fine-tune from confirmed mappings | High |
| **Auto-Ontology Expansion** | Crowdsource IoT schemas | Medium |
| **Binary Protocol Support** | Protobuf, CBOR, custom binary | High |
| **Explainable Translation** | Show LLM reasoning chain | Medium |
| **AegisLang Integration** | Policy-based compliance rules | Low |
| **Federated Learning** | Cross-deployment schema sharing | Low |
| **Local LLM Support** | Ollama/vLLM for air-gapped | Medium |

---

*Specification complete. Ready for implementation.*
