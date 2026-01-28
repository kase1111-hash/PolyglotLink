# PolyglotLink Software Audit Report

**Audit Date:** 2026-01-28
**Auditor:** Claude Code (claude-opus-4-5-20251101)
**Version Audited:** 0.1.0 (Alpha)

## Executive Summary

PolyglotLink is a Semantic API Translator for IoT Device Ecosystems that transforms heterogeneous IoT device payloads across multiple protocols into normalized, semantically enriched JSON objects. This audit evaluated the software for **correctness** and **fitness for purpose**.

**Overall Assessment:** The software demonstrates solid architectural design and good engineering practices but contains several issues that would impact production reliability. **8 issues identified** require attention before production deployment.

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 2 |
| Medium | 4 |
| Low | 4 |

---

## Critical Issues

### 1. Embedding Vector Dimension Mismatch

**File:** `polyglotlink/modules/semantic_translator_agent.py`
**Lines:** 296-297, 301

**Description:** The fallback pseudo-embedding uses SHA256 hash which produces 32 bytes, but real OpenAI embeddings (`text-embedding-3-large`) have 3072 dimensions. When these are mixed, `zip(a, b, strict=True)` on line 301 will raise a `ValueError` because the vectors have different lengths.

**Impact:** The semantic translation system completely fails when OpenAI API is unavailable or when mixing cached embeddings with new ones.

**Current Code:**
```python
# Fallback: simple hash-based pseudo-embedding
import hashlib
hash_bytes = hashlib.sha256(text.encode()).digest()
return [float(b) / 255.0 for b in hash_bytes]  # Returns 32 elements
```

**Recommendation:** Either:
1. Pad fallback embeddings to match the expected dimension (3072)
2. Store embedding dimension metadata and validate before comparison
3. Use a deterministic algorithm that produces the correct dimension

---

### 2. SQL Injection Vulnerability in TimescaleDB Output

**File:** `polyglotlink/modules/output_broker.py`
**Lines:** 412-418

**Description:** The table name is inserted into SQL queries via f-string interpolation without parameterization or validation.

**Current Code:**
```python
await conn.executemany(
    f"""
    INSERT INTO {self.config.timescale.table_name}
    (time, device_id, metric, value)
    VALUES ($1, $2, $3, $4)
    """,
    ...
)
```

**Impact:** If `table_name` configuration can be influenced by external input (e.g., environment variables from untrusted sources), an attacker could execute arbitrary SQL.

**Recommendation:**
1. Validate table name against a strict pattern (alphanumeric and underscores only)
2. Use `psycopg2.sql.Identifier` or equivalent for safe identifier quoting
3. Consider a whitelist approach for allowed table names

---

## High Priority Issues

### 3. Flawed Payload Encoding Detection

**File:** `polyglotlink/modules/protocol_listener.py`
**Lines:** 109-115

**Description:** The encoding detection logic incorrectly identifies any even-length binary payload as `MODBUS_REGISTERS` before falling back to `BINARY`. This will cause legitimate binary files, images, or compressed data to be incorrectly parsed.

**Current Code:**
```python
# Check for Modbus register format
if len(payload) % 2 == 0 and len(payload) > 0:
    # Could be Modbus registers (16-bit values)
    return PayloadEncoding.MODBUS_REGISTERS

# Binary fallback
return PayloadEncoding.BINARY
```

**Impact:** Corrupted data when processing binary payloads that happen to have even length.

**Recommendation:** Only return `MODBUS_REGISTERS` when the message actually came from the Modbus protocol handler or has explicit protocol metadata indicating Modbus origin.

---

### 4. MQTT Wildcard Pattern Matching Bug

**File:** `polyglotlink/modules/protocol_listener.py`
**Lines:** 357-358

**Description:** MQTT wildcards (`#` and `+`) are converted to fnmatch patterns, but the conversion is incorrect. MQTT `#` matches multiple levels but fnmatch `**` is not a valid glob pattern in Python's `fnmatch` module.

**Current Code:**
```python
mqtt_pattern = pattern.replace("+", "*").replace("#", "**")
if fnmatch.fnmatch(topic, mqtt_pattern):
```

**Impact:** Topic pattern matching fails for multi-level wildcards, affecting message routing and metadata.

**Recommendation:** Implement proper MQTT topic matching using a dedicated function:
```python
def mqtt_topic_matches(topic: str, pattern: str) -> bool:
    topic_parts = topic.split('/')
    pattern_parts = pattern.split('/')
    # Implement proper MQTT wildcard matching
```

---

## Medium Priority Issues

### 5. Deprecated `datetime.utcnow()` Usage

**Files:** Multiple files throughout the codebase
**Locations:**
- `protocol_listener.py:332, 397, 475, 586, 680`
- `schema_extractor.py:422, 437`
- `semantic_translator_agent.py:611, 637`
- `normalization_engine.py:547, 555`
- `output_broker.py:460, 529`
- `server.py:56, 253, 307`

**Description:** `datetime.utcnow()` is deprecated in Python 3.12 and will be removed in future versions.

**Impact:** Code will emit deprecation warnings and eventually break in future Python versions.

**Recommendation:** Replace all occurrences with:
```python
from datetime import datetime, timezone
datetime.now(timezone.utc)
```

---

### 6. Deprecated `asyncio.get_event_loop()` Usage

**Files:** `output_broker.py:532`, `server.py:363`

**Description:** `asyncio.get_event_loop()` is deprecated for getting the running loop from coroutines.

**Recommendation:** Use `asyncio.get_running_loop()` when inside a coroutine.

---

### 7. Type Annotation Error in `apply_conversion()`

**File:** `polyglotlink/modules/normalization_engine.py`
**Line:** 186

**Description:** The function `apply_conversion()` is annotated to return `float` but returns `None` when the input value is `None`.

**Current Code:**
```python
def apply_conversion(value: Any, formula: str) -> float:
    if value is None:
        return None  # Returns None, not float
```

**Recommendation:** Change return type to `float | None`.

---

### 8. Timezone-Naive Datetime Conversion

**File:** `polyglotlink/modules/normalization_engine.py`
**Line:** 245

**Description:** `datetime.fromtimestamp()` uses the local timezone, which can cause inconsistent behavior in distributed systems or when servers are in different timezones.

**Current Code:**
```python
return datetime.fromtimestamp(value)
```

**Recommendation:**
```python
return datetime.fromtimestamp(value, tz=timezone.utc)
```

---

## Low Priority Issues

### 9. Missing Protobuf Parser

**File:** `polyglotlink/modules/protocol_listener.py`
**Lines:** 169-179

**Description:** Protobuf encoding detection exists (`is_likely_protobuf()`) but no parser is registered in `ENCODING_PARSERS`. Messages detected as Protobuf cannot be parsed.

**Recommendation:** Either add a Protobuf parser or remove the detection to avoid confusion.

---

### 10. Weak Protobuf Detection Heuristic

**File:** `polyglotlink/modules/protocol_listener.py`
**Lines:** 43-50

**Description:** The wire type check (`wire_type in (0, 1, 2, 5)`) is too simplistic and will produce false positives for random binary data.

**Recommendation:** Require additional heuristics or protocol metadata for Protobuf detection.

---

### 11. Unix Timestamp Range Limitations

**File:** `polyglotlink/modules/schema_extractor.py`
**Lines:** 189-192

**Description:** The Unix timestamp detection misses timestamps before September 9, 2001 (value < 1,000,000,000 seconds).

**Recommendation:** Expand the range or use a more sophisticated timestamp detection approach.

---

### 12. HTTP Server Task Not Tracked

**File:** `polyglotlink/modules/protocol_listener.py`
**Line:** 421

**Description:** The HTTP server task is created with `asyncio.create_task()` but the reference is not stored, making proper cleanup uncertain.

**Recommendation:** Store the task reference and await it during shutdown.

---

## Fitness for Purpose Assessment

### Strengths

1. **Well-Structured Architecture:** Clear separation of concerns with modular pipeline stages (Protocol Listener → Schema Extractor → Semantic Translator → Normalization Engine → Output Broker).

2. **Comprehensive Protocol Support:** Supports 6 protocols (MQTT, CoAP, Modbus, OPC-UA, HTTP, WebSocket) and 7 payload formats.

3. **Secure Formula Evaluation:** The `SafeExpressionEvaluator` class properly uses AST-based evaluation instead of `eval()`, preventing code injection.

4. **Robust Configuration:** Pydantic-based settings with validation, environment variable support, and proper defaults.

5. **Good Test Coverage:** Comprehensive test suite covering unit tests, integration tests, validation tests, security tests, and fuzzing.

6. **Proper Error Handling:** Custom exception types, structured logging with structlog, and Sentry integration for error tracking.

### Weaknesses

1. **Fallback Mode Broken:** The system cannot function correctly without OpenAI API due to the embedding dimension mismatch issue.

2. **Encoding Detection Fragile:** Binary payloads may be misidentified, leading to data corruption.

3. **Security Gaps:** SQL injection vulnerability in TimescaleDB output.

4. **Inconsistent Datetime Handling:** Mixed use of timezone-aware and timezone-naive datetimes could cause issues.

### Verdict

**The software is partially fit for purpose.** The core architecture and design are sound, but the identified issues prevent reliable production deployment. After addressing the Critical and High priority issues, the software would be suitable for production use in IoT semantic translation scenarios.

---

## Recommendations Summary

| Priority | Issue | Effort |
|----------|-------|--------|
| Critical | Fix embedding dimension mismatch | Medium |
| Critical | Parameterize SQL table name | Low |
| High | Fix encoding detection for binary payloads | Low |
| High | Implement proper MQTT wildcard matching | Medium |
| Medium | Replace deprecated datetime.utcnow() | Low |
| Medium | Replace deprecated asyncio.get_event_loop() | Low |
| Medium | Fix apply_conversion() return type | Low |
| Medium | Use timezone-aware datetime conversion | Low |

**Estimated Total Effort:** 2-3 days of focused development work

---

## Files Audited

| File | Lines | Issues Found |
|------|-------|--------------|
| `modules/protocol_listener.py` | 848 | 4 |
| `modules/schema_extractor.py` | 456 | 1 |
| `modules/semantic_translator_agent.py` | 718 | 1 |
| `modules/normalization_engine.py` | 607 | 2 |
| `modules/output_broker.py` | 603 | 2 |
| `models/schemas.py` | 373 | 0 |
| `utils/config.py` | 269 | 0 |
| `app/server.py` | 381 | 2 |

---

*Report generated by Claude Code audit session*
