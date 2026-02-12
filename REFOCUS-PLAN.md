# PolyglotLink Refocus Plan

**Goal:** Strip premature scaffolding, close critical product gaps, and ship a system that a user can actually interact with.

**Guiding principle:** Every change either removes weight or adds load-bearing structure. Nothing cosmetic.

---

## Phase 0 — Cut Dead Weight

**Objective:** Reduce the project's surface area to what's real. Remove everything that creates the appearance of maturity without the substance.

### 0.1 Remove SDR module

SDR signal processing (ADS-B aircraft tracking, POCSAG pager decoding, APRS, ACARS, RDS, FLEX) is a different product. An IoT semantic translator has no business decoding pager messages.

**Delete:**
- `polyglotlink/modules/sdr_handler.py`

**Remove from `polyglotlink/models/schemas.py`:**
- `SDR` value from the `Protocol` enum
- All `SDR_*` values from the `PayloadEncoding` enum (SDR_ADS_B, SDR_POCSAG, SDR_APRS, SDR_ACARS, SDR_RDS, SDR_FLEX, SDR_RAW_IQ)
- `SDRConfig` class and its reference in `ProtocolListenerConfig`

**Remove from `pyproject.toml`:**
- Optional SDR dependencies: `pyrtlsdr`, `scipy`, `matplotlib`

**Remove from `polyglotlink/utils/config.py`:**
- Any `SDRSettings` or SDR-related configuration fields

**Update `polyglotlink/modules/protocol_listener.py`:**
- Remove SDR handler imports, initialization, and any SDR-related branches in encoding detection

### 0.2 Remove unwired infrastructure backends

Neo4j is configured but never queried. Weaviate is referenced but the system degrades gracefully without it. Remove both containers and their dependencies until features that use them are actually built.

**`docker-compose.yml` — remove these service blocks entirely:**
- `neo4j` (lines 108-128) and its volumes (`neo4j-data`, `neo4j-logs`)
- `weaviate` (lines 130-146) and its volume (`weaviate-data`)

**`docker-compose.yml` — remove from polyglotlink service environment:**
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (lines 22-24)
- `WEAVIATE_URL` (line 25)

**`pyproject.toml` — remove dependencies:**
- `neo4j>=5.0`
- `weaviate-client>=4.0`

**`polyglotlink/utils/config.py` — remove or comment out:**
- `Neo4jSettings` class
- `WeaviateSettings` class
- Their references in the root `Settings` class

**Keep in code but make optional (already degrades gracefully):**
- Weaviate references in `semantic_translator_agent.py` — the system already falls back to hash-based similarity when Weaviate is unavailable. No code change needed; just removing the container.

### 0.3 Remove premature deployment/governance artifacts

**Delete entirely:**
- `deploy/kubernetes/deployment.yaml`
- `deploy/kubernetes/configmap-production.yaml`
- `deploy/kubernetes/service.yaml`
- `deploy/kubernetes/hpa.yaml`
- `deploy/monitoring/grafana-dashboard.json`
- `deploy/monitoring/prometheus-alerts.yaml`
- `deploy/kubernetes/` (directory)
- `deploy/monitoring/` (directory)
- `.github/workflows/release.yml`
- `.bumpversion.toml`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `docs/FAQ.md`

**Remove from `Makefile`:**
- `deploy-staging`, `deploy-prod`, `rollback-*` targets (no deployment target exists)
- `version-patch`, `version-minor`, `version-major` targets (bumpversion removed)

### 0.4 Remove premature observability

**Delete:**
- `polyglotlink/utils/error_logging.py` (Sentry integration)

**Defer but keep in tree (low cost, already written):**
- `polyglotlink/utils/metrics.py` (Prometheus) — keep the file but remove any Sentry imports from it. Metrics will be useful once the API exists.

**Remove from `pyproject.toml` if present as required dep:**
- `sentry-sdk` (if listed)

### Phase 0 exit criteria
- `docker-compose up` starts 5 services (polyglotlink, redis, mosquitto, zookeeper, kafka) + timescaledb — not 8
- `import polyglotlink` has no SDR imports in the critical path
- No references to Neo4j or Weaviate in docker-compose.yml
- `deploy/` directory contains only `deploy/` (empty or with only deploy.sh and scripts)
- All existing tests still pass

---

## Phase 1 — Wire Up Redis Schema Caching

**Objective:** Schema learnings survive restarts. This is required before the REST API (Phase 2) because the API needs to serve cached schemas.

### 1.1 Initialize Redis client in server startup

**File:** `polyglotlink/app/server.py`, line 155 (the TODO)

Replace:
```python
# TODO: Initialize Redis-backed cache if available
cache = SchemaCache(ttl_days=30)
```

With Redis client initialization:
```python
redis_client = None
try:
    import redis
    settings = get_settings()
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
```

This is safe — `SchemaCache` (at `schema_extractor.py:279-333`) already handles `redis_client=None` gracefully with local-only fallback.

### 1.2 Verify cache persistence

**New test in `polyglotlink/tests/test_schema_extractor.py`:**
- Create a `SchemaCache` with a mock Redis client
- `set()` a mapping, verify `.setex()` was called with the correct key format (`schema:{signature}`)
- `get()` a mapping, verify local-first then Redis fallback
- Verify TTL is passed correctly (timedelta → seconds for Redis)

### Phase 1 exit criteria
- Schema cache hits Redis on get/set when Redis is available
- System degrades to in-memory when Redis is down (no crash)
- Schema mappings survive a server restart when Redis is running

---

## Phase 2 — Build the REST API

**Objective:** Transform PolyglotLink from a headless daemon into something a user can interact with via HTTP.

### 2.1 Create API router

**New file:** `polyglotlink/api/routes/v1.py`

Endpoints:

| Method | Path | Purpose | Response Model |
|--------|------|---------|----------------|
| `GET` | `/api/v1/health` | Detailed health (pipeline status, connected protocols, cache stats) | JSON |
| `POST` | `/api/v1/ingest` | Submit a raw payload for processing through the full pipeline | `NormalizedMessage` |
| `GET` | `/api/v1/schemas` | List all cached schema signatures with metadata | `list[ExtractedSchema]` summary |
| `GET` | `/api/v1/schemas/{signature}` | Get a specific cached schema and its mapping | `CachedMapping` |
| `GET` | `/api/v1/mappings` | List recent semantic mappings | `list[SemanticMapping]` summary |
| `POST` | `/api/v1/test` | Dry-run: process a payload and return the result without publishing to output broker | `NormalizedMessage` |

### 2.2 Wire router into FastAPI app

**File:** `polyglotlink/app/server.py`, in `create_app()` (line 316-349)

After existing route definitions, add:
```python
from polyglotlink.api.routes.v1 import router as v1_router
app.include_router(v1_router, prefix="/api/v1")
```

### 2.3 Implement the ingest endpoint

The `POST /api/v1/ingest` endpoint is the core user-facing feature. It should:

1. Accept a JSON body with `payload` (the raw device data) and optional `device_id`, `protocol` hint
2. Construct a `RawMessage` from the request
3. Run it through the pipeline: extract → translate → normalize
4. Optionally publish to output broker (controlled by query param `?publish=true`)
5. Return the `NormalizedMessage` as JSON

This reuses the exact same pipeline that `_process_messages()` in `server.py` uses, just triggered by HTTP instead of MQTT/CoAP.

### 2.4 Implement the test endpoint

`POST /api/v1/test` is identical to ingest but **never publishes**. This lets users iterate on their payloads and see normalized output without side effects.

### 2.5 Implement schema listing

`GET /api/v1/schemas` reads from `SchemaCache` (now Redis-backed from Phase 1) and returns all known schema signatures with their field counts, last-seen timestamps, and hit counts.

This requires adding a `list_all()` method to `SchemaCache`:
- In-memory: iterate `_local_cache`
- Redis: `SCAN` for keys matching `schema:*`

### 2.6 API tests

**New file:** `polyglotlink/tests/test_api.py`

Using FastAPI's `TestClient`:
- `POST /api/v1/test` with a simple JSON payload → verify `NormalizedMessage` response
- `POST /api/v1/test` with XML payload → verify encoding detection works via HTTP
- `GET /api/v1/schemas` → verify empty list, then ingest, then verify non-empty
- `GET /api/v1/schemas/{signature}` → verify 404 for unknown, 200 for known
- `POST /api/v1/ingest?publish=false` → verify no side effects

### Phase 2 exit criteria
- `curl -X POST localhost:8080/api/v1/test -d '{"payload": {"temp": 25.5, "hum": 60}}'` returns a normalized JSON object
- `curl localhost:8080/api/v1/schemas` returns cached schemas
- All new endpoints have tests
- OpenAPI docs auto-generated at `/docs`

---

## Phase 3 — Build the Demo

**Objective:** Prove the concept works in 60 seconds. One command, visible result.

### 3.1 Create demo script

**New file:** `scripts/demo.py`

The script should:
1. Check that docker-compose services are running (Redis, Mosquitto at minimum)
2. Send 3 different device payloads via the REST API (`POST /api/v1/test`):
   - **Device A:** Temperature sensor (Celsius) — `{"temp_c": 23.5, "humidity_pct": 45, "bat_v": 3.2}`
   - **Device B:** Weather station (Fahrenheit, different field names) — `{"temperature_f": 74.3, "rh": 45, "pressure_hpa": 1013}`
   - **Device C:** Industrial sensor (nested, metric units) — `{"readings": {"temperature": {"value": 296.65, "unit": "K"}, "vibration": {"x": 0.02, "y": 0.01, "z": 0.03}}}`
3. Print side-by-side: raw input vs normalized output
4. Show that all three devices' temperature readings are normalized to the same unit and ontology concept
5. Query `GET /api/v1/schemas` to show 3 different schemas were auto-learned

### 3.2 Add Makefile target

**File:** `Makefile`

```makefile
demo:  ## Run end-to-end demo with sample IoT payloads
	python scripts/demo.py
```

### 3.3 Update README

Replace the aspirational architecture documentation with:
1. A "Quick Start" section: `make docker-up && make demo`
2. Actual output from the demo showing normalized data
3. Remove references to Neo4j ontology querying, Weaviate vector search, and SDR as current features
4. Move the architecture deep-dive to `docs/architecture/ARCHITECTURE.md` (already exists)
5. Keep the README under 500 lines — focused on what the project does today, not what it might do

### Phase 3 exit criteria
- `make demo` runs successfully and produces visible, understandable output
- A new user can go from `git clone` to seeing normalized IoT data in under 5 minutes
- README accurately reflects current capabilities

---

## Phase 4 — Integration Tests with Real Services

**Objective:** Prove the system works with actual MQTT, Redis, and the HTTP API running — not just mocked unit tests.

### 4.1 Create docker-compose.test.yml

A minimal compose file for testing:
- Redis (for schema caching)
- Mosquitto (for MQTT protocol testing)
- PolyglotLink (the app)

No Kafka, no TimescaleDB, no Neo4j, no Weaviate. Minimal footprint.

### 4.2 Write integration tests

**New file:** `polyglotlink/tests/test_integration_live.py`

Tests that require running services (marked with `@pytest.mark.integration`):

1. **MQTT end-to-end:** Publish a JSON payload to Mosquitto topic → verify it arrives in the pipeline → verify normalized output via `GET /api/v1/schemas`
2. **Redis persistence:** Ingest a payload → restart the app (not Redis) → verify schema cache survives
3. **Multi-format:** Send JSON, XML, and CSV payloads via `POST /api/v1/test` → verify all three are normalized
4. **Schema reuse:** Send the same payload twice → verify second hit uses cached schema (check cache stats via API)

### 4.3 Add Makefile target

```makefile
test-live:  ## Run integration tests against real services
	docker compose -f docker-compose.test.yml up -d
	pytest polyglotlink/tests/test_integration_live.py -v --timeout=60
	docker compose -f docker-compose.test.yml down
```

### Phase 4 exit criteria
- `make test-live` passes with real MQTT and Redis
- CI can optionally run integration tests (add to `ci.yml` as a separate job with service containers)

---

## Dependency Graph

```
Phase 0 (Cut)
    │
    ▼
Phase 1 (Redis Caching)
    │
    ▼
Phase 2 (REST API)  ← requires Phase 1 for schema listing
    │
    ▼
Phase 3 (Demo)  ← requires Phase 2 for HTTP endpoints
    │
    ▼
Phase 4 (Integration Tests)  ← requires Phase 2 + Phase 3
```

Phases are strictly sequential. Each phase builds on the prior.

---

## What This Plan Does NOT Include

These are deliberately excluded. They are good ideas for later, not now:

- **Neo4j ontology registry** — build it when you have 50+ ontology concepts that outgrow the hardcoded defaults
- **Weaviate vector search** — add it when hash-based similarity proves insufficient for real-world schemas
- **Kubernetes deployment** — deploy to K8s when you have a production user, not before
- **Prometheus/Grafana dashboards** — instrument when there's traffic to monitor
- **JSON-LD export** — add when a user requests semantic web compatibility
- **Schema management CLI** — build after the REST API exists (CLI can call the API)
- **CONTRIBUTING.md / SECURITY.md** — re-add when the project has contributors
- **Release automation** — re-add when there's something to release

---

## Success Metric

After all four phases, this command sequence should work:

```bash
git clone <repo>
cp .env.example .env
make docker-up
# wait 30 seconds for services
make demo
```

And produce visible output showing three different IoT device payloads automatically normalized to a unified schema — proving the core value proposition in under 2 minutes.
