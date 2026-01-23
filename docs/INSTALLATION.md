# Installation Guide

This guide covers the installation and setup of PolyglotLink for different environments.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
  - [Using pip](#using-pip)
  - [Using Docker](#using-docker)
  - [From Source](#from-source)
- [Configuration](#configuration)
- [Infrastructure Setup](#infrastructure-setup)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Disk Space**: 2GB minimum

### Software Requirements

- **Python**: 3.10 or higher
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Docker Compose**: 2.0+ (optional)

### Optional Services

PolyglotLink integrates with the following services (can be run via Docker Compose):

- **Redis**: Caching and message queuing
- **MQTT Broker** (Mosquitto): For MQTT protocol support
- **Apache Kafka**: For message streaming output
- **Neo4j**: Ontology graph storage
- **Weaviate**: Vector embeddings storage
- **TimescaleDB**: Time-series metrics storage

## Quick Start

The fastest way to get started is with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/polyglotlink/polyglotlink.git
cd polyglotlink

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings (especially OPENAI_API_KEY)

# Start all services
docker-compose up -d

# Check the status
docker-compose ps
```

PolyglotLink will be available at:
- **HTTP API**: http://localhost:8080
- **WebSocket**: ws://localhost:8081
- **CoAP**: coap://localhost:5683

## Installation Methods

### Using pip

Install PolyglotLink from PyPI:

```bash
# Install the package
pip install polyglotlink

# Or with a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install polyglotlink
```

Verify the installation:

```bash
polyglotlink version
```

### Using Docker

Pull the official Docker image:

```bash
# Pull the latest image
docker pull polyglotlink/polyglotlink:latest

# Run the container
docker run -d \
  --name polyglotlink \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 5683:5683/udp \
  -v $(pwd)/.env:/app/.env \
  polyglotlink/polyglotlink:latest
```

Or use Docker Compose for the full stack:

```bash
# Clone the repository
git clone https://github.com/polyglotlink/polyglotlink.git
cd polyglotlink

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d
```

### From Source

For development or to get the latest changes:

```bash
# Clone the repository
git clone https://github.com/polyglotlink/polyglotlink.git
cd polyglotlink

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# LLM Configuration (Required)
OPENAI_API_KEY=your-api-key-here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=INFO

# Protocol Configuration
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_TOPIC_PATTERNS=devices/#,sensors/#

COAP_HOST=0.0.0.0
COAP_PORT=5683

# Storage
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
WEAVIATE_URL=http://localhost:8085

# Output
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
OUTPUT_KAFKA_TOPIC=polyglotlink.normalized
```

See `.env.example` for the complete list of configuration options.

### Configuration Files

Environment-specific configurations are in the `config/` directory:

- `config/development.yaml`: Development settings
- `config/staging.yaml`: Staging environment settings
- `config/production.yaml`: Production settings

## Infrastructure Setup

### Using Docker Compose (Recommended)

Start all infrastructure services:

```bash
docker-compose up -d redis mosquitto kafka zookeeper
```

### Manual Setup

#### Redis

```bash
# Using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install locally (Ubuntu/Debian)
sudo apt install redis-server
sudo systemctl start redis
```

#### Mosquitto (MQTT Broker)

```bash
# Using Docker
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:2

# Or install locally (Ubuntu/Debian)
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

#### Kafka

```bash
# Using Docker Compose (includes Zookeeper)
docker-compose up -d zookeeper kafka
```

## Verifying Installation

### Check CLI

```bash
# Show version
polyglotlink version

# Run health check
polyglotlink check
```

### Start the Server

```bash
# Start in verbose mode
polyglotlink serve --verbose
```

### Test the API

```bash
# Health check endpoint
curl http://localhost:8080/health

# API documentation (Swagger UI)
open http://localhost:8080/docs
```

### Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

## Troubleshooting

### Common Issues

#### OpenAI API Key Not Set

```
Error: OPENAI_API_KEY environment variable not set
```

**Solution**: Set your OpenAI API key in the `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
```

#### Connection Refused to Redis

```
Error: Connection refused to localhost:6379
```

**Solution**: Start Redis:
```bash
docker-compose up -d redis
# or
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### MQTT Connection Failed

```
Error: MQTT connection failed to localhost:1883
```

**Solution**: Start the MQTT broker:
```bash
docker-compose up -d mosquitto
# or
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:2
```

#### Port Already in Use

```
Error: Address already in use: 8080
```

**Solution**: Stop the conflicting service or use a different port:
```bash
SERVER_PORT=8081 polyglotlink serve
```

### Getting Help

- Check the [FAQ](FAQ.md) for common questions
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging
- Open an issue on [GitHub](https://github.com/polyglotlink/polyglotlink/issues)

## Next Steps

After installation, you can:

1. **Explore the API**: Visit http://localhost:8080/docs for interactive documentation
2. **Configure protocols**: Set up MQTT topics, CoAP resources, or other protocol handlers
3. **Set up outputs**: Configure Kafka, MQTT output, or webhook destinations
4. **Review architecture**: See [ARCHITECTURE.md](architecture/ARCHITECTURE.md) for system design
