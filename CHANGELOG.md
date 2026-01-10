# Changelog

All notable changes to PolyglotLink will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance benchmarks for core components
- Load testing with Locust for stress testing
- Prometheus metrics integration for observability

## [0.1.0] - 2024-01-15

### Added

#### Core Features
- **Protocol Listener Module**: Multi-protocol IoT message ingestion
  - MQTT support with topic pattern subscriptions
  - CoAP (Constrained Application Protocol) support
  - Modbus TCP/RTU support
  - OPC-UA support
  - HTTP/REST webhook ingestion
  - WebSocket real-time connections
  - Automatic payload encoding detection (JSON, XML, CBOR, Binary)

- **Schema Extractor Module**: Automatic schema detection and extraction
  - Zero-configuration schema inference from payloads
  - Nested JSON flattening with dot notation
  - Type detection (string, integer, float, boolean, datetime)
  - Unit inference from field names (celsius, fahrenheit, percent, etc.)
  - Semantic hint detection (temperature, humidity, pressure, location)
  - Schema signature caching for performance

- **Semantic Translator Agent**: Intelligent field mapping
  - Embedding-based semantic matching
  - LLM fallback for complex mappings
  - Canonical ontology with extensible concepts
  - Confidence scoring for mappings
  - Cache-first resolution strategy

- **Normalization Engine**: Value conversion and standardization
  - Safe formula execution for unit conversions
  - Type coercion with validation
  - Range validation for known concepts
  - Metadata enrichment
  - Device registry tracking

- **Output Broker Module**: Multi-destination publishing
  - Apache Kafka output with configurable partitioning
  - MQTT republishing
  - HTTP/REST POST endpoints
  - WebSocket broadcast
  - TimescaleDB time-series storage
  - JSON and JSON-LD output formats

#### Infrastructure
- Docker and docker-compose for containerized deployment
- Multi-stage Dockerfile for optimized images
- Environment-specific configurations (dev/staging/production)
- Redis for caching
- Neo4j for ontology storage
- Weaviate for vector embeddings
- TimescaleDB for time-series data

#### Developer Experience
- Full CLI with serve, check, and version commands
- Comprehensive logging with structlog
- Sentry integration for error tracking
- Secrets management (env vars, AWS Secrets Manager, Vault)
- Pydantic-based configuration validation

#### Testing & Quality
- Unit tests for all core modules
- Integration tests for end-to-end pipeline
- System/acceptance tests for specification compliance
- Performance benchmarks
- Load testing with Locust
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline

#### Documentation
- MIT License
- Semantic versioning with bump2version
- Makefile for common operations

### Security
- Input validation and sanitization
- Malicious pattern detection
- Safe formula execution (no arbitrary code)
- TLS support for all protocols
- API key authentication support

### Performance
- Schema caching with signature matching
- Embedding result caching
- Async processing throughout
- Connection pooling for databases
- Configurable rate limiting

## [0.0.1] - 2024-01-01

### Added
- Initial project structure
- README with project specification
- Basic data models and schemas

---

## Release Notes

### Version Naming
- **Major** (X.0.0): Breaking changes to API or data formats
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Upgrade Guide

#### From 0.0.x to 0.1.0
This is the first feature-complete release. If upgrading from an earlier development version:

1. Update configuration to use new Pydantic settings format
2. Migrate any custom schemas to new ExtractedSchema format
3. Update Docker images to use new multi-stage build
4. Review new environment variables in `.env.example`

### Known Issues
- LLM translation requires OpenAI API key
- Some unit conversions may need manual ontology additions
- WebSocket output requires async event loop

### Deprecations
None in this release.

[Unreleased]: https://github.com/polyglotlink/polyglotlink/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/polyglotlink/polyglotlink/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/polyglotlink/polyglotlink/releases/tag/v0.0.1
