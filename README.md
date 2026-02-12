# PolyglotLink

Semantic API Translator for IoT Device Ecosystems.

Drop in any IoT device speaking any protocol, and PolyglotLink automatically understands its data schema and normalizes it into a unified ontology — no adapter code required.

## Quick Start

```bash
git clone https://github.com/polyglotlink/polyglotlink.git
cd polyglotlink
pip install -e .
make demo
```

The demo sends 3 different IoT device payloads through the pipeline and shows normalized output:

- **Device A:** Temperature sensor (Celsius, flat JSON)
- **Device B:** Weather station (Fahrenheit, different field names)
- **Device C:** Industrial sensor (nested JSON, Kelvin)

All three produce normalized output with auto-detected schemas, inferred semantics, and unit conversions — without writing any adapter code.

## What It Does

PolyglotLink runs a 5-stage pipeline on every incoming IoT message:

```
Raw Device Payload
    |
1. Protocol Listener -- ingest from MQTT, CoAP, Modbus, OPC-UA, HTTP, WebSocket
    |
2. Schema Extractor -- flatten, detect types, infer units and semantic hints
    |
3. Semantic Translator -- map fields to ontology concepts (via embeddings or LLM)
    |
4. Normalization Engine -- convert units, enforce types, validate ranges
    |
5. Output Broker -- publish to Kafka, MQTT, HTTP webhooks, TimescaleDB
    |
Normalized JSON
```

## REST API

All endpoints are under `/api/v1/`. OpenAPI docs at `/docs` when running.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/test` | Dry-run: process a payload, see normalized output, no side effects |
| `POST` | `/api/v1/ingest` | Process and optionally publish (`?publish=false` to skip) |
| `GET`  | `/api/v1/schemas` | List all auto-learned schemas |
| `GET`  | `/api/v1/schemas/{sig}` | Get a specific cached schema mapping |
| `GET`  | `/api/v1/health` | Detailed health: pipeline status, cache stats |

### Example

```bash
curl -X POST http://localhost:8080/api/v1/test \
  -H "Content-Type: application/json" \
  -d '{"payload": {"temp_c": 23.5, "humidity_pct": 45, "bat_v": 3.2}}'
```

## With Docker

```bash
cp .env.example .env
# Optionally add your OpenAI API key to .env for LLM-powered semantic translation
make docker-up    # Starts Redis, Mosquitto, Kafka, TimescaleDB
make run          # Starts PolyglotLink
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | Enables LLM-powered semantic translation. Without it, uses embedding fallback. |
| `MQTT_BROKER_HOST` | `localhost` | MQTT broker for protocol listener |
| `REDIS_URL` | `redis://localhost:6379/0` | Schema cache persistence |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Output destination |

See `.env.example` for the full list.

## Protocol Support

| Protocol | Status | Library |
|----------|--------|---------|
| MQTT | Working | `paho-mqtt` |
| HTTP | Working | FastAPI (built-in) |
| CoAP | Working | `aiocoap` |
| Modbus TCP | Working | `pymodbus` |
| OPC-UA | Working | `asyncua` |
| WebSocket | Working | `websockets` |

## Encoding Detection

Payloads are automatically detected as JSON, XML, CBOR, CSV, Protobuf (heuristic), Modbus registers, or binary.

## Schema Extraction

For every incoming payload, PolyglotLink:

- Flattens nested structures (e.g., `sensor.temperature.value`)
- Infers types (integer, float, boolean, datetime, string)
- Detects units from field names (50+ patterns: celsius, fahrenheit, pascal, volt, percent, ...)
- Assigns semantic hints (40+ categories: temperature, humidity, pressure, battery, location, ...)
- Generates a stable schema signature for caching

## Semantic Translation

Fields are mapped to ontology concepts using a cascade:

1. **Cache hit** -- instant (schema signature match from Redis)
2. **Embedding similarity** -- OpenAI `text-embedding-3-large` (or hash-based fallback)
3. **LLM inference** -- GPT-4o for complex/ambiguous mappings
4. **Passthrough** -- unmapped fields preserved with low confidence

Works without an OpenAI API key (uses embedding fallback), better with one.

## Unit Conversion

The normalization engine handles 50+ conversions:

- Temperature: Celsius, Fahrenheit, Kelvin
- Pressure: Pascal, Bar, PSI, Hectopascal
- Speed: m/s, km/h, mph
- Length, mass, volume, time, percentage

All formulas use AST-based safe evaluation (no `eval()`).

## Testing

```bash
make test          # Run all tests (285 tests)
make test-unit     # Unit tests only
make test-cov      # With coverage report
```

## Project Structure

```
polyglotlink/
  app/
    server.py              # Pipeline orchestration, FastAPI app
    cli.py                 # CLI interface (serve, check, test)
  api/routes/
    v1.py                  # REST API endpoints
  modules/
    protocol_listener.py   # Multi-protocol ingestion
    schema_extractor.py    # Schema extraction + caching
    semantic_translator_agent.py  # LLM/embedding-based mapping
    normalization_engine.py      # Unit conversion + validation
    output_broker.py       # Multi-destination publishing
  models/
    schemas.py             # Pydantic data models
  utils/                   # Config, logging, validation, metrics
  tests/                   # 285 tests across 9 files
```

## Architecture Deep Dive

See [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) for the full architecture specification.

## License

MIT
