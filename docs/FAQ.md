# PolyglotLink FAQ

Frequently asked questions about PolyglotLink - the Semantic API Translator for IoT.

## General Questions

### What is PolyglotLink?

PolyglotLink is a semantic API translator that creates a unified layer for IoT device communication. It automatically translates heterogeneous device protocols and data formats into a normalized, semantically-enriched message format.

### What protocols does PolyglotLink support?

PolyglotLink supports the following protocols:
- **MQTT** - Message Queuing Telemetry Transport
- **CoAP** - Constrained Application Protocol
- **HTTP/REST** - Webhook-based ingestion
- **WebSocket** - Real-time bidirectional communication
- **Modbus** - Industrial protocol (TCP and RTU)
- **OPC-UA** - Industrial automation standard

### What payload formats are supported?

- JSON (most common)
- XML
- CBOR (Concise Binary Object Representation)
- Raw binary (protocol-specific parsing)

### Do I need to configure schemas for my devices?

No! PolyglotLink automatically detects and extracts schemas from incoming payloads. It uses machine learning to infer field types, units, and semantic meaning without prior configuration.

## Installation & Setup

### What are the system requirements?

- Python 3.10 or higher
- Docker (for containerized deployment)
- 2GB RAM minimum (4GB recommended)
- Network access to your IoT devices/brokers

### How do I install PolyglotLink?

**Using pip:**
```bash
pip install polyglotlink
```

**Using Docker:**
```bash
docker pull ghcr.io/polyglotlink/polyglotlink:latest
docker run -p 8080:8080 ghcr.io/polyglotlink/polyglotlink:latest
```

**From source:**
```bash
git clone https://github.com/polyglotlink/polyglotlink
cd polyglotlink
pip install -e ".[dev,test]"
```

### How do I configure the application?

PolyglotLink uses environment variables for configuration. Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

Key configuration options:
- `POLYGLOTLINK_ENV` - Environment (development/staging/production)
- `OPENAI_API_KEY` - Required for semantic translation
- `MQTT_BROKER_HOST` - MQTT broker address
- `REDIS_HOST` - Redis cache address

## Semantic Translation

### How does semantic translation work?

PolyglotLink uses a three-tier approach:

1. **Cache Check** - Previously translated schemas are retrieved instantly
2. **Embedding Matching** - Field names are compared to ontology concepts using vector similarity
3. **LLM Fallback** - For low-confidence matches, an LLM provides intelligent translation

### What is the ontology?

The ontology is a collection of canonical concepts that represent standard IoT measurements and properties. Examples include:
- `temperature_celsius` - Temperature in Celsius
- `humidity_percent` - Relative humidity
- `power_watts` - Electrical power
- `location_latitude` - GPS latitude

### Can I add custom concepts?

Yes! Use the API to add custom concepts:

```bash
curl -X POST http://localhost:8080/api/v1/ontology/concepts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_metric",
    "category": "custom",
    "canonical_unit": "units",
    "description": "My custom metric"
  }'
```

### What LLM providers are supported?

Currently, PolyglotLink supports OpenAI's GPT models. Support for other providers (Anthropic Claude, local models) is planned.

## Performance

### How many messages can PolyglotLink process?

Performance depends on your deployment:
- **Single instance**: 1,000-5,000 messages/second (cached schemas)
- **With LLM translation**: 10-50 messages/second (API rate limited)
- **Kubernetes cluster**: Scales horizontally to millions of messages/second

### How do I optimize performance?

1. **Enable caching** - Redis caching dramatically improves throughput
2. **Use batch ingestion** - Send multiple messages per request
3. **Pre-warm schemas** - Send sample messages to populate cache
4. **Scale horizontally** - Deploy multiple instances behind a load balancer

### Why is the first message slow?

The first message for a new device requires:
1. Schema extraction
2. Embedding generation
3. Semantic matching (possibly LLM call)

Subsequent messages with the same schema are processed from cache (~1ms).

## Troubleshooting

### PolyglotLink won't start

**Check logs:**
```bash
polyglotlink serve --verbose
# or
docker logs polyglotlink
```

**Common issues:**
- Missing `OPENAI_API_KEY` environment variable
- Redis/broker not reachable
- Port 8080 already in use

### Messages are not being processed

1. **Check health endpoint:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Verify broker connection:**
   ```bash
   polyglotlink check
   ```

3. **Check metrics:**
   ```bash
   curl http://localhost:8080/metrics | grep polyglotlink_messages
   ```

### High error rate

Check error types in metrics:
```bash
curl http://localhost:8080/metrics | grep polyglotlink_messages_failed
```

Common causes:
- Invalid JSON payloads
- Network timeouts to LLM API
- Database connection issues

### Schema extraction returns empty fields

Ensure your payload is valid JSON:
```bash
echo '{"temp": 23.5}' | python -m json.tool
```

Check that field names are meaningful (not just "a", "b", "c").

### LLM translation is slow

- Check OpenAI API status: https://status.openai.com
- Verify your API key has sufficient quota
- Consider using a faster model (gpt-4o-mini instead of gpt-4o)

## Security

### Is my data secure?

- All connections support TLS encryption
- API keys are never logged
- Payloads are processed in memory (not persisted by default)
- Sentry error tracking masks sensitive data

### How do I enable authentication?

Set in your configuration:
```bash
API_KEY_REQUIRED=true
POLYGLOTLINK_API_KEY=your-secret-key
```

Then include the key in requests:
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8080/ingest/mqtt
```

### What data is sent to OpenAI?

Only field names and sample values are sent for semantic translation. You can disable LLM translation and use embedding-only matching:

```bash
LLM_ENABLED=false
```

## Integration

### How do I send data to Kafka?

Configure Kafka output in your settings:
```bash
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC_PREFIX=polyglotlink_
```

### Can I use PolyglotLink with Home Assistant?

Yes! Configure MQTT output to republish normalized messages to your Home Assistant MQTT broker.

### How do I integrate with TimescaleDB?

```bash
TIMESCALE_ENABLED=true
TIMESCALE_HOST=timescaledb
TIMESCALE_DATABASE=iot_data
```

Normalized messages are automatically inserted into the `normalized_messages` hypertable.

## Development

### How do I run tests?

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make test-cov

# Fuzzing tests
pytest polyglotlink/tests/test_fuzzing.py -v
```

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Where can I get help?

- GitHub Issues: https://github.com/polyglotlink/polyglotlink/issues
- Documentation: https://polyglotlink.io/docs
- Discord: https://discord.gg/polyglotlink
