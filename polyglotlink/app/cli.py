"""
PolyglotLink Command Line Interface

Provides CLI commands for running and managing the PolyglotLink service.
"""

import argparse
import asyncio
import signal
import sys
from typing import Optional

from polyglotlink.utils.config import get_settings, reload_settings
from polyglotlink.utils.error_logging import flush as flush_errors, init_sentry
from polyglotlink.utils.logging import configure_logging, get_logger


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="polyglotlink",
        description="Semantic API Translator for IoT Device Ecosystems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  polyglotlink serve                    Start the server with default settings
  polyglotlink serve --port 8080        Start on specific port
  polyglotlink serve --mqtt-only        Only enable MQTT listener
  polyglotlink check                    Check configuration
  polyglotlink version                  Show version information

Environment Variables:
  POLYGLOTLINK_ENV                      Environment (development/staging/production)
  LOG_LEVEL                             Logging level (DEBUG/INFO/WARNING/ERROR)
  OPENAI_API_KEY                        OpenAI API key for semantic translation
  MQTT_BROKER_HOST                      MQTT broker hostname
  REDIS_URL                             Redis connection URL

For more information, visit: https://github.com/polyglotlink/polyglotlink
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the PolyglotLink server",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP server host (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080)",
    )
    serve_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    serve_parser.add_argument(
        "--mqtt-only",
        action="store_true",
        help="Only enable MQTT listener",
    )
    serve_parser.add_argument(
        "--http-only",
        action="store_true",
        help="Only enable HTTP listener",
    )
    serve_parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable all output brokers (dry run mode)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check configuration and connectivity",
    )
    check_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full connectivity checks",
    )

    # Version command
    subparsers.add_parser(
        "version",
        help="Show version information",
    )

    # Schema command
    schema_parser = subparsers.add_parser(
        "schema",
        help="Schema management commands",
    )
    schema_subparsers = schema_parser.add_subparsers(dest="schema_command")

    schema_subparsers.add_parser(
        "list",
        help="List cached schemas",
    )
    schema_subparsers.add_parser(
        "clear",
        help="Clear schema cache",
    )
    schema_export = schema_subparsers.add_parser(
        "export",
        help="Export ontology",
    )
    schema_export.add_argument(
        "--format",
        choices=["json", "json-ld", "rdf"],
        default="json",
        help="Export format (default: json)",
    )
    schema_export.add_argument(
        "-o", "--output",
        type=str,
        help="Output file (default: stdout)",
    )

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test message processing",
    )
    test_parser.add_argument(
        "input",
        type=str,
        help="Input message (JSON string or @filename)",
    )
    test_parser.add_argument(
        "--protocol",
        choices=["mqtt", "http", "coap"],
        default="mqtt",
        help="Simulate protocol (default: mqtt)",
    )
    test_parser.add_argument(
        "--topic",
        type=str,
        default="devices/test/telemetry",
        help="Simulate topic/path",
    )

    return parser


def get_log_level(verbose: int, quiet: bool) -> str:
    """Determine log level from verbosity flags."""
    if quiet:
        return "ERROR"
    if verbose >= 2:
        return "DEBUG"
    if verbose >= 1:
        return "INFO"
    return get_settings().log_level


async def cmd_serve(args: argparse.Namespace) -> int:
    """Run the serve command."""
    logger = get_logger("cli")
    settings = get_settings()

    logger.info(
        "Starting PolyglotLink server",
        host=args.host,
        port=args.port,
        environment=settings.env,
    )

    # Import here to avoid circular imports
    from polyglotlink.app.server import create_app, run_server

    # Build configuration overrides
    overrides = {}
    if args.mqtt_only:
        overrides["http_enabled"] = False
        overrides["coap_enabled"] = False
        overrides["websocket_enabled"] = False
    if args.http_only:
        overrides["mqtt_enabled"] = False
        overrides["coap_enabled"] = False
        overrides["websocket_enabled"] = False
    if args.no_output:
        overrides["output_enabled"] = False

    try:
        await run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            **overrides,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        return 0
    except Exception as e:
        logger.error("Server error", error=str(e))
        return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Run the check command."""
    logger = get_logger("cli")
    settings = get_settings()

    print("PolyglotLink Configuration Check")
    print("=" * 40)
    print()

    errors = []
    warnings = []

    # Check environment
    print(f"Environment: {settings.env}")
    print(f"Log Level: {settings.log_level}")
    print()

    # Check LLM configuration
    print("LLM Configuration:")
    if settings.llm.openai_api_key:
        print("  OpenAI API Key: [configured]")
    else:
        warnings.append("OpenAI API key not configured - semantic translation will use fallback")
        print("  OpenAI API Key: [not configured]")
    print(f"  Model: {settings.llm.model}")
    print()

    # Check protocol listeners
    print("Protocol Listeners:")
    print(f"  MQTT: {'enabled' if settings.mqtt.enabled else 'disabled'}")
    if settings.mqtt.enabled:
        print(f"    Broker: {settings.mqtt.broker_host}:{settings.mqtt.broker_port}")
    print(f"  HTTP: {'enabled' if settings.http.enabled else 'disabled'}")
    if settings.http.enabled:
        print(f"    Address: {settings.http.host}:{settings.http.port}")
    print(f"  CoAP: {'enabled' if settings.coap.enabled else 'disabled'}")
    print(f"  Modbus: {'enabled' if settings.modbus.enabled else 'disabled'}")
    print(f"  OPC-UA: {'enabled' if settings.opcua.enabled else 'disabled'}")
    print(f"  WebSocket: {'enabled' if settings.websocket.enabled else 'disabled'}")
    print()

    # Check storage
    print("Storage:")
    print(f"  Redis: {settings.redis.url}")
    print(f"  Neo4j: {settings.neo4j.uri}")
    print(f"  Weaviate: {settings.weaviate.url}")
    print(f"  TimescaleDB: {settings.timescale.url.split('@')[-1] if '@' in settings.timescale.url else settings.timescale.url}")
    print()

    # Check outputs
    print("Output Brokers:")
    print(f"  Kafka: {'enabled' if settings.kafka.enabled else 'disabled'}")
    if settings.kafka.enabled:
        print(f"    Servers: {settings.kafka.bootstrap_servers}")
    print()

    # Full connectivity check
    if args.full:
        print("Connectivity Checks:")
        print("  (Running connectivity checks...)")

        # Check Redis
        try:
            import redis
            r = redis.from_url(settings.redis.url)
            r.ping()
            print("  Redis: OK")
        except Exception as e:
            errors.append(f"Redis connection failed: {e}")
            print(f"  Redis: FAILED - {e}")

        # Check MQTT
        if settings.mqtt.enabled:
            try:
                import paho.mqtt.client as mqtt
                client = mqtt.Client()
                client.connect(settings.mqtt.broker_host, settings.mqtt.broker_port, 5)
                client.disconnect()
                print("  MQTT: OK")
            except Exception as e:
                errors.append(f"MQTT connection failed: {e}")
                print(f"  MQTT: FAILED - {e}")

        print()

    # Summary
    print("=" * 40)
    if errors:
        print(f"ERRORS: {len(errors)}")
        for err in errors:
            print(f"  - {err}")
    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        for warn in warnings:
            print(f"  - {warn}")
    if not errors and not warnings:
        print("All checks passed!")

    return 1 if errors else 0


def cmd_version(args: argparse.Namespace) -> int:
    """Run the version command."""
    print("PolyglotLink v0.1.0")
    print()
    print("Semantic API Translator for IoT Device Ecosystems")
    print()
    print("Python:", sys.version.split()[0])
    print("Platform:", sys.platform)
    return 0


async def cmd_test(args: argparse.Namespace) -> int:
    """Run the test command."""
    import json

    logger = get_logger("cli")

    # Load input
    if args.input.startswith("@"):
        with open(args.input[1:], "r") as f:
            input_data = f.read()
    else:
        input_data = args.input

    try:
        payload = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input - {e}")
        return 1

    print("Input Message:")
    print(json.dumps(payload, indent=2))
    print()

    # Import processing modules
    from polyglotlink.models.schemas import Protocol, RawMessage
    from polyglotlink.modules.protocol_listener import detect_encoding, generate_uuid
    from polyglotlink.modules.schema_extractor import SchemaExtractor
    from polyglotlink.modules.semantic_translator_agent import SemanticTranslator
    from polyglotlink.modules.normalization_engine import NormalizationEngine

    # Create raw message
    payload_bytes = json.dumps(payload).encode()
    raw = RawMessage(
        message_id=generate_uuid(),
        device_id="test-device",
        protocol=Protocol[args.protocol.upper()],
        topic=args.topic,
        payload_raw=payload_bytes,
        payload_encoding=detect_encoding(payload_bytes),
    )

    print(f"Protocol: {raw.protocol.value}")
    print(f"Topic: {raw.topic}")
    print(f"Encoding: {raw.payload_encoding.value}")
    print()

    # Extract schema
    extractor = SchemaExtractor()
    schema = extractor.extract_schema(raw)

    print("Extracted Schema:")
    print(f"  Signature: {schema.schema_signature}")
    print(f"  Fields: {len(schema.fields)}")
    for field in schema.fields:
        print(f"    - {field.key}: {field.value_type}", end="")
        if field.inferred_unit:
            print(f" ({field.inferred_unit})", end="")
        if field.inferred_semantic:
            print(f" -> {field.inferred_semantic}", end="")
        print()
    print()

    # Translate schema
    translator = SemanticTranslator()
    mapping = await translator.translate_schema(schema)

    print("Semantic Mapping:")
    print(f"  Confidence: {mapping.confidence:.2f}")
    print(f"  LLM Used: {mapping.llm_generated}")
    print(f"  Mappings:")
    for m in mapping.field_mappings:
        print(f"    - {m.source_field} -> {m.target_field} ({m.resolution_method.value})")
    print()

    # Normalize
    engine = NormalizationEngine()
    normalized = engine.normalize_message(schema, mapping)

    print("Normalized Message:")
    print(json.dumps(normalized.data, indent=2, default=str))

    if normalized.conversions:
        print()
        print("Conversions Applied:")
        for conv in normalized.conversions:
            print(f"  - {conv.field}: {conv.original_value} {conv.from_unit} -> {conv.converted_value} {conv.to_unit}")

    if normalized.validation_errors:
        print()
        print("Validation Errors:")
        for err in normalized.validation_errors:
            print(f"  - {err.field}: {err.error.value} - {err.details}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Set up logging
    log_level = get_log_level(args.verbose, args.quiet)
    settings = get_settings()
    configure_logging(
        log_level=log_level,
        json_logs=settings.env == "production",
        development=settings.env == "development",
    )

    # Initialize error tracking
    init_sentry()

    # Handle commands
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "serve":
            return asyncio.run(cmd_serve(args))
        elif args.command == "check":
            return cmd_check(args)
        elif args.command == "version":
            return cmd_version(args)
        elif args.command == "test":
            return asyncio.run(cmd_test(args))
        elif args.command == "schema":
            print("Schema commands not yet implemented")
            return 1
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        flush_errors()


if __name__ == "__main__":
    sys.exit(main())
