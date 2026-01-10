# PolyglotLink Troubleshooting Guide

This guide helps you diagnose and resolve common issues with PolyglotLink.

## Quick Diagnostics

### Health Check

```bash
# Check service health
curl http://localhost:8080/health

# Expected output:
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "redis": "ok",
    "timescaledb": "ok",
    "kafka": "ok"
  }
}
```

### System Check

```bash
# Run built-in diagnostics
polyglotlink check

# Or via Docker
docker exec polyglotlink polyglotlink check
```

### View Metrics

```bash
# Get Prometheus metrics
curl http://localhost:8080/metrics | grep -E "^polyglotlink_"
```

---

## Startup Issues

### Service Won't Start

**Symptoms:**
- Container exits immediately
- "Address already in use" error
- "Configuration error" in logs

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :8080
   # If occupied, use a different port:
   HTTP_PORT=8081 polyglotlink serve
   ```

2. **Verify environment variables:**
   ```bash
   # Required variables
   echo $OPENAI_API_KEY  # Must be set
   echo $POLYGLOTLINK_ENV  # development/staging/production
   ```

3. **Check configuration file:**
   ```bash
   polyglotlink check --config
   ```

### Missing Dependencies

**Symptoms:**
- "ModuleNotFoundError" in logs
- Import errors

**Solutions:**

```bash
# Reinstall with all dependencies
pip install -e ".[dev,test]"

# Or use Docker (includes all dependencies)
docker-compose up -d
```

### Permission Denied

**Symptoms:**
- Cannot bind to port
- Cannot write to log directory

**Solutions:**

```bash
# Use non-privileged port
HTTP_PORT=8080  # Not 80 or 443

# Fix directory permissions
sudo chown -R $USER:$USER /var/log/polyglotlink
```

---

## Connection Issues

### Cannot Connect to MQTT Broker

**Symptoms:**
- "Connection refused" errors
- Messages not being received

**Diagnosis:**
```bash
# Test MQTT connection
mosquitto_pub -h localhost -t test -m "hello"

# Check broker status
docker-compose logs mosquitto
```

**Solutions:**

1. **Verify broker address:**
   ```bash
   MQTT_BROKER_HOST=192.168.1.100  # Use IP, not hostname
   MQTT_BROKER_PORT=1883
   ```

2. **Check TLS configuration:**
   ```bash
   # If using TLS
   MQTT_TLS_ENABLED=true
   MQTT_CA_CERT=/path/to/ca.pem
   ```

3. **Verify credentials:**
   ```bash
   MQTT_USERNAME=user
   MQTT_PASSWORD=password
   ```

### Cannot Connect to Redis

**Symptoms:**
- "Connection refused" to Redis
- Cache not working

**Diagnosis:**
```bash
# Test Redis connection
redis-cli -h localhost ping
# Should return: PONG
```

**Solutions:**

1. **Check Redis is running:**
   ```bash
   docker-compose ps redis
   ```

2. **Verify connection settings:**
   ```bash
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=  # Empty if no password
   ```

### Cannot Connect to Kafka

**Symptoms:**
- Output messages not appearing in Kafka
- "NoBrokersAvailable" error

**Diagnosis:**
```bash
# List Kafka topics
kafka-topics.sh --list --bootstrap-server localhost:9092
```

**Solutions:**

1. **Check Kafka is healthy:**
   ```bash
   docker-compose logs kafka
   ```

2. **Verify bootstrap servers:**
   ```bash
   KAFKA_BOOTSTRAP_SERVERS=kafka:9092  # Use container name in Docker
   ```

---

## Processing Issues

### Messages Not Being Processed

**Symptoms:**
- Messages received but not appearing in output
- Low throughput

**Diagnosis:**
```bash
# Check message counters
curl http://localhost:8080/metrics | grep polyglotlink_messages
```

**Solutions:**

1. **Check for errors:**
   ```bash
   curl http://localhost:8080/metrics | grep polyglotlink_errors
   ```

2. **Increase log verbosity:**
   ```bash
   LOG_LEVEL=DEBUG polyglotlink serve
   ```

3. **Check queue backlog:**
   ```bash
   curl http://localhost:8080/metrics | grep polyglotlink_queue_size
   ```

### Schema Extraction Fails

**Symptoms:**
- "SchemaExtractionError" in logs
- Empty fields in extracted schema

**Diagnosis:**
```bash
# Test with a sample payload
curl -X POST http://localhost:8080/ingest/mqtt \
  -H "Content-Type: application/json" \
  -d '{"topic": "test", "payload": {"temp": 23.5}}'
```

**Solutions:**

1. **Verify JSON validity:**
   ```bash
   echo '{"temp": 23.5}' | python -m json.tool
   ```

2. **Check payload size:**
   ```bash
   # Default limit is 1MB
   MAX_PAYLOAD_SIZE=10485760  # 10MB
   ```

3. **Check encoding:**
   ```bash
   # Ensure UTF-8 encoding
   file your_payload.json
   ```

### Semantic Translation Slow

**Symptoms:**
- High latency on first message
- LLM timeouts

**Diagnosis:**
```bash
# Check LLM metrics
curl http://localhost:8080/metrics | grep polyglotlink_llm
```

**Solutions:**

1. **Verify API key:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

2. **Use faster model:**
   ```bash
   LLM_MODEL=gpt-4o-mini  # Faster than gpt-4o
   ```

3. **Increase timeout:**
   ```bash
   LLM_TIMEOUT=60  # seconds
   ```

4. **Enable caching:**
   ```bash
   CACHE_EMBEDDINGS=true
   CACHE_TTL=86400  # 24 hours
   ```

---

## Performance Issues

### High Memory Usage

**Symptoms:**
- OOM killer terminating process
- Slow garbage collection

**Diagnosis:**
```bash
# Check memory usage
curl http://localhost:8080/metrics | grep process_resident_memory
```

**Solutions:**

1. **Limit cache size:**
   ```bash
   MAX_CACHE_SIZE=10000  # entries
   ```

2. **Enable memory limits in Docker:**
   ```yaml
   # docker-compose.yml
   services:
     polyglotlink:
       mem_limit: 1g
   ```

3. **Reduce concurrent connections:**
   ```bash
   MAX_CONNECTIONS=100
   ```

### High CPU Usage

**Symptoms:**
- CPU at 100%
- Slow response times

**Diagnosis:**
```bash
# Check CPU usage
curl http://localhost:8080/metrics | grep process_cpu
```

**Solutions:**

1. **Profile the application:**
   ```bash
   ENABLE_PROFILING=true polyglotlink serve
   ```

2. **Reduce batch size:**
   ```bash
   BATCH_SIZE=10  # messages per batch
   ```

3. **Scale horizontally:**
   ```bash
   # In Kubernetes
   kubectl scale deployment polyglotlink --replicas=3
   ```

### High Latency

**Symptoms:**
- P95 latency > 1 second
- Slow API responses

**Diagnosis:**
```bash
# Check latency percentiles
curl http://localhost:8080/metrics | grep polyglotlink_message_processing_seconds
```

**Solutions:**

1. **Enable caching:**
   ```bash
   REDIS_ENABLED=true
   CACHE_TTL=3600
   ```

2. **Optimize database queries:**
   ```bash
   TIMESCALE_POOL_SIZE=20
   ```

3. **Use async processing:**
   ```bash
   ASYNC_PROCESSING=true
   ```

---

## Output Issues

### Messages Not Appearing in Kafka

**Diagnosis:**
```bash
# Check Kafka output metrics
curl http://localhost:8080/metrics | grep polyglotlink_output.*kafka
```

**Solutions:**

1. **Verify topic exists:**
   ```bash
   kafka-topics.sh --create --topic polyglotlink_normalized \
     --bootstrap-server localhost:9092
   ```

2. **Check producer config:**
   ```bash
   KAFKA_ACKS=all
   KAFKA_RETRIES=5
   ```

### TimescaleDB Insert Failures

**Diagnosis:**
```bash
# Check database connection
psql -h localhost -U postgres -d polyglotlink -c "SELECT 1"
```

**Solutions:**

1. **Run migrations:**
   ```bash
   make db-migrate
   ```

2. **Check table exists:**
   ```sql
   SELECT * FROM normalized_messages LIMIT 1;
   ```

3. **Verify hypertable:**
   ```sql
   SELECT * FROM timescaledb_information.hypertables;
   ```

---

## Logging & Debugging

### Enable Debug Logging

```bash
# Environment variable
LOG_LEVEL=DEBUG polyglotlink serve

# Or in configuration
logging:
  level: DEBUG
  format: console
```

### View Structured Logs

```bash
# Parse JSON logs
docker logs polyglotlink 2>&1 | jq .

# Filter by level
docker logs polyglotlink 2>&1 | jq 'select(.level == "error")'
```

### Enable Request Tracing

```bash
# Enable trace headers
ENABLE_TRACING=true

# View traces in Jaeger (if configured)
open http://localhost:16686
```

---

## Getting Help

### Collect Diagnostic Information

```bash
# Generate diagnostic report
polyglotlink diagnose > diagnostic_report.txt
```

### Report an Issue

Include in your bug report:
1. PolyglotLink version: `polyglotlink version`
2. Python version: `python --version`
3. Operating system: `uname -a`
4. Configuration (sanitized)
5. Relevant log excerpts
6. Steps to reproduce

File issues at: https://github.com/polyglotlink/polyglotlink/issues
