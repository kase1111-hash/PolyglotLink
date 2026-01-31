# PolyglotLink

Semantic API Translator for IoT Device Ecosystems. Transforms heterogeneous IoT device payloads across multiple protocols and data formats into normalized, semantically-enriched JSON objects.

## Architecture

5-stage pipeline processing:

1. **Protocol Listener** (`polyglotlink/modules/protocol_listener.py`) - Multi-protocol ingestion (MQTT, CoAP, Modbus, OPC-UA, HTTP, WebSocket, SDR)
2. **Schema Extractor** (`polyglotlink/modules/schema_extractor.py`) - Automatic schema inference and type detection
3. **Semantic Translator** (`polyglotlink/modules/semantic_translator_agent.py`) - LLM/embedding-based field mapping using GPT-4o
4. **Normalization Engine** (`polyglotlink/modules/normalization_engine.py`) - Unit conversion and type coercion
5. **Output Broker** (`polyglotlink/modules/output_broker.py`) - Multi-destination publishing (Kafka, MQTT, HTTP, WebSocket, TimescaleDB)

## Tech Stack

- **Python 3.10+** with FastAPI, Pydantic v2, async/await
- **LLM**: OpenAI GPT-4o via `openai` SDK
- **Protocols**: paho-mqtt, aiocoap, pymodbus, asyncua, websockets
- **Data Formats**: JSON, XML (defusedxml), CBOR (cbor2), CSV, Protobuf
- **Storage**: Redis (cache), Neo4j (ontology), Weaviate (embeddings), TimescaleDB (time-series)
- **Messaging**: Apache Kafka
- **Monitoring**: Prometheus, Grafana, Sentry

## Project Structure

```
polyglotlink/
├── app/                    # Entry points (main.py, cli.py, server.py)
├── modules/                # Core processing modules (5 pipeline stages)
├── models/schemas.py       # Pydantic data models
├── api/routes/             # FastAPI routes
├── utils/                  # Config, logging, metrics, validation
└── tests/                  # Test suite (pytest)
config/                     # YAML configs (development, staging, production)
deploy/kubernetes/          # K8s manifests
```

## Development Commands

```bash
# Install
make install-dev           # Install with dev dependencies

# Run
make run                   # Start the server
make dev                   # Development mode with auto-reload

# Test
make test                  # Run all tests
make test-unit             # Unit tests only
make test-integration      # Integration tests only
make test-cov              # Tests with coverage report

# Code Quality
make lint                  # Run linters (ruff)
make format                # Format code (black)
make type-check            # Type checking (mypy)
make security-check        # Security scan (bandit)

# Docker
make docker-build          # Build Docker image
make docker-up             # Start Docker Compose stack
make docker-down           # Stop Docker Compose stack
```

## Configuration

Environment variables (see `.env.example`):
- `OPENAI_API_KEY` - Required for semantic translation
- `POLYGLOT_ENVIRONMENT` - development/staging/production
- Protocol-specific settings for MQTT, CoAP, Modbus, OPC-UA, etc.
- Storage connection URLs for Redis, Neo4j, Weaviate, TimescaleDB
- Output destinations for Kafka, MQTT republish, HTTP webhooks

YAML configs in `config/` directory for environment-specific settings.

## Key Conventions

- **Async throughout**: All I/O operations use async/await
- **Pydantic models**: All data structures defined as Pydantic BaseModel
- **Structured logging**: Uses structlog for JSON logging
- **Type hints**: Full type annotations, checked with mypy
- **Security-first**: defusedxml for XML, input validation, safe formula execution
- **snake_case** for modules/functions, **PascalCase** for classes

## Testing

Tests located in `polyglotlink/tests/`:
- `test_system.py` - End-to-end system tests
- `test_integration.py` - Component integration tests
- `test_schema_extractor.py` - Schema extraction unit tests
- `test_normalization_engine.py` - Unit conversion tests
- `test_security.py` - Security validation tests
- `test_fuzzing.py` - Property-based fuzzing tests

Run with: `pytest polyglotlink/tests/ -v`

## Performance Targets

| Stage | Latency |
|-------|---------|
| Protocol Listener | < 5ms |
| Schema Extractor | < 10ms |
| Semantic Translator | < 150ms |
| Normalization Engine | < 20ms |
| Output Broker | < 15ms |
| **End-to-End** | **< 200ms** |

## Documentation

- `README.md` - Full documentation
- `ARCHITECTURE.md` - System architecture with diagrams
- `docs/INSTALLATION.md` - Setup instructions
- `docs/TROUBLESHOOTING.md` - Debugging guide
- `docs/api/openapi.yaml` - API specification
