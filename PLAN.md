# Implementation Plan: Fix All Security Audit Findings

Fixes all 20 findings from SECURITY-AUDIT.md, grouped into 7 implementation steps ordered by dependency (not just severity). Each step lists the exact files to create/modify and the changes needed.

---

## Step 1: Add Security Settings to Configuration

**Files:** `polyglotlink/utils/config.py`

Add a `SecuritySettings` class to `config.py` that actually loads the security config defined in `production.yaml` but never wired up:

```python
class SecuritySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECURITY_")
    api_key_required: bool = Field(default=False, validation_alias="API_KEY_REQUIRED")
    api_key: str | None = Field(default=None, validation_alias="API_KEY")
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_per_minute: int = Field(default=1000, ge=1, le=100000)
    max_request_size_bytes: int = Field(default=1_048_576)  # 1MB
    max_json_depth: int = Field(default=50, ge=1, le=200)
    cors_origins: list[str] = Field(default_factory=list)
```

Add `LLMSettings` fields for cost controls:
```python
    max_llm_calls_per_minute: int = Field(default=60, ge=1, le=10000)
```

Add `security: SecuritySettings` to the main `Settings` class.

Remove default credentials from `TimescaleSettings.url` — set to empty string and add a validator that rejects `postgres:postgres` in production.

Add a startup warning in `RedisSettings` when URL has no password in non-development envs.

**Fixes:** CRITICAL-01, LOW-01 (partial), CRITICAL-02 (config only), CRITICAL-03 (config only), HIGH-02 (config only), MEDIUM-05 (config only)

---

## Step 2: Add API Key Authentication Middleware

**Files:** `polyglotlink/utils/middleware.py` (new), `polyglotlink/app/server.py`

Create `middleware.py` with:

1. **`APIKeyMiddleware`** — A Starlette middleware that:
   - Reads `SecuritySettings` from app state
   - Skips auth for `/health` and `/ready` (load balancer probes)
   - For all other routes, checks the `X-API-Key` header against configured key
   - Returns 401 with generic message on failure
   - Logs failed auth attempts (IP, path) at WARNING level

2. **`RateLimitMiddleware`** — A simple in-memory token-bucket rate limiter:
   - Per-IP rate limiting using `rate_limit_per_minute` from settings
   - Returns 429 with `Retry-After` header when exceeded
   - Uses a dict of `{ip: (tokens, last_refill_time)}`
   - Cleans up stale entries periodically

3. **`RequestSizeLimitMiddleware`** — Checks `Content-Length` against `max_request_size_bytes`, returns 413 if exceeded.

Update `create_app()` in `server.py` to:
- Load settings and attach `SecuritySettings` to `app.state`
- Add the three middlewares (conditionally: auth only when `api_key_required=True`)
- Add `CORSMiddleware` from FastAPI with `cors_origins` from settings (default: no origins allowed)
- Move `/metrics` behind auth (keep `/health` and `/ready` public)

**Fixes:** CRITICAL-02, CRITICAL-03, MEDIUM-05, LOW-05

---

## Step 3: Wire Input Validation into API Layer & Sanitize Error Responses

**Files:** `polyglotlink/api/routes/v1.py`, `polyglotlink/app/server.py`

In `v1.py`:
- Add a `validate_ingest_body()` dependency that:
  - Calls `validate_payload_size()` on the raw body
  - Calls `validate_json_depth()` on the parsed payload
  - Calls `validate_json_size()` on the parsed payload
  - Returns the validated body or raises HTTPException(400)
- Apply this as a `Depends()` on `POST /ingest` and `POST /test`

In `server.py` `create_app()`:
- Add a global exception handler for `PolyglotLinkError` that:
  - In production: returns `{"error": error.code, "message": "Processing failed"}` (generic)
  - In development: returns the full `error.to_dict()`
  - Always logs the full exception server-side
- Add a fallback exception handler for unhandled `Exception` that returns 500 with generic message and logs the traceback

Replace the bare `except Exception as e: raise HTTPException(status_code=422, detail=str(e))` in both `/ingest` and `/test` with the structured error handling above.

**Fixes:** MEDIUM-02, MEDIUM-03

---

## Step 4: Prompt Injection Defense & LLM Cost Controls

**Files:** `polyglotlink/modules/semantic_translator_agent.py`

**Prompt injection defense:**
- In `build_fields_table()`:
  - Truncate field **names** to 60 characters (values already truncated to 30)
  - Strip control characters and non-printable Unicode from both names and values
  - Escape pipe characters `|` in field names/values (they break the Markdown table)
- In `LLMTranslator.translate()`:
  - After parsing LLM JSON response, validate that each `target_concept` is either in `DEFAULT_ONTOLOGY_CONCEPTS` or starts with `_` (passthrough)
  - Reject any mapping with a `target_concept` not in the known set (log and skip it)
  - Clamp `confidence` values to [0.0, 1.0]

**LLM cost controls:**
- Add a `_call_count` and `_window_start` to `LLMTranslator.__init__`
- In `_call_llm()`, before making the API call:
  - Check if calls in the current minute exceed `max_llm_calls_per_minute`
  - If exceeded, log a warning and fall through to rule-based fallback instead of calling the API
- Log estimated token count after each LLM call using `response.usage.total_tokens`

**Fixes:** HIGH-03, HIGH-02

---

## Step 5: Fix Docker, Credentials & Infrastructure

**Files:** `docker-compose.yml`, `docker-compose.prod.yml` (new), `Dockerfile`, `.env.example`

**docker-compose.yml** (development):
- Change TimescaleDB `POSTGRES_PASSWORD` from `postgres` to `${POSTGRES_PASSWORD:-devpassword123}`
- Change Redis command to `redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-devpassword123}`
- Keep port mappings (needed for development)

**docker-compose.prod.yml** (new file for production overrides):
- Remove all host port mappings except `8080` (the API)
- Override TimescaleDB to require `${POSTGRES_PASSWORD}` (no default)
- Override Redis to require `${REDIS_PASSWORD}` (no default)
- Disable Kafka auto-topic-creation: `KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"`
- Pin TimescaleDB image to `timescale/timescaledb:2.14.2-pg15` (specific version)
- Add an nginx/Traefik reverse proxy service for TLS termination (or document as requirement)

**Dockerfile:**
- Replace `COPY polyglotlink/requirements.txt .` / `pip install -r requirements.txt` with:
  ```dockerfile
  COPY pyproject.toml .
  COPY polyglotlink/ ./polyglotlink/
  RUN pip install --no-cache-dir .
  ```

**.env.example:**
- Change `OPENAI_API_KEY=sk-your-api-key-here` to `OPENAI_API_KEY=`
- Change `TIMESCALE_URL` to `TIMESCALE_URL=postgresql://postgres:CHANGEME@localhost:5432/iot`
- Add `REDIS_PASSWORD=` and `POSTGRES_PASSWORD=` entries
- Add `API_KEY_REQUIRED=false` and `API_KEY=` entries
- Add `SECURITY_RATE_LIMIT_PER_MINUTE=1000`

**Fixes:** CRITICAL-01, CRITICAL-04, HIGH-01, HIGH-05, MEDIUM-04, MEDIUM-06, LOW-03, LOW-04

---

## Step 6: Pin Dependencies & Add CI

**Files:** `pyproject.toml`, `polyglotlink/requirements.txt`, `.github/workflows/security.yml` (new)

**pyproject.toml:**
- Add `slowapi>=0.1.9` to dependencies (for potential future migration from custom rate limiter)
- No version pin changes here — `pyproject.toml` keeps `>=` for library compatibility

**requirements.txt:**
- Regenerate by running `pip-compile pyproject.toml -o polyglotlink/requirements.txt` (or manually pin current resolved versions)
- Remove stale entries (`neo4j`, `weaviate-client`) that were removed in Phase 0 but remain in requirements.txt
- Add `defusedxml`, `pydantic-settings`, `python-dotenv` which are in pyproject.toml but missing from requirements.txt

**.github/workflows/security.yml:**
```yaml
name: Security
on: [push, pull_request]
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install .[dev,test]
      - run: pip-audit
      - run: bandit -r polyglotlink/ -x polyglotlink/tests/
      - run: ruff check polyglotlink/
      - run: mypy polyglotlink/ --ignore-missing-imports
```

Add a scheduled weekly run for dependency scanning.

**Fixes:** HIGH-04, LOW-02

---

## Step 7: Low-Severity Cleanup & Secrets Logging

**Files:** `polyglotlink/utils/secrets.py`, `polyglotlink/app/server.py`, `polyglotlink/utils/config.py`

**secrets.py:**
- Change `logger.info("Loaded secrets from .env", count=...)` to `logger.debug(...)` on lines 99 and 224

**config.py:**
- In `TimescaleSettings`, add a model_validator that logs a WARNING at startup if the URL contains `postgres:postgres` and `env != "development"`
- In `RedisSettings`, add a model_validator that logs a WARNING if the URL has no password and `env != "development"`

**server.py:**
- Add TLS configuration options to `run_server()`: accept `ssl_keyfile` and `ssl_certfile` parameters, pass them to uvicorn config
- Document in docstring that production deployments should use a reverse proxy for TLS

**Fixes:** MEDIUM-01, LOW-01, LOW-04, LOW-05

---

## Summary of All Files Changed

| File | Action | Findings Addressed |
|------|--------|--------------------|
| `polyglotlink/utils/config.py` | Edit | CRITICAL-01, LOW-01, MEDIUM-05 config |
| `polyglotlink/utils/middleware.py` | **Create** | CRITICAL-02, CRITICAL-03, LOW-05 |
| `polyglotlink/app/server.py` | Edit | CRITICAL-02, CRITICAL-03, MEDIUM-03, MEDIUM-05, LOW-04, LOW-05 |
| `polyglotlink/api/routes/v1.py` | Edit | MEDIUM-02, MEDIUM-03 |
| `polyglotlink/modules/semantic_translator_agent.py` | Edit | HIGH-02, HIGH-03 |
| `polyglotlink/utils/secrets.py` | Edit | MEDIUM-01 |
| `docker-compose.yml` | Edit | CRITICAL-01, CRITICAL-04, MEDIUM-06 |
| `docker-compose.prod.yml` | **Create** | CRITICAL-04, HIGH-05, LOW-03, LOW-04 |
| `Dockerfile` | Edit | MEDIUM-04 |
| `.env.example` | Edit | HIGH-01, CRITICAL-01 |
| `pyproject.toml` | Edit | HIGH-04 |
| `polyglotlink/requirements.txt` | Edit | HIGH-04, MEDIUM-04 |
| `.github/workflows/security.yml` | **Create** | LOW-02 |

**New files:** 3
**Modified files:** 10
**Total findings addressed:** 20/20
