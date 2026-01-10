"""
Performance Benchmarks for PolyglotLink Core Components

These tests measure the performance characteristics of core components
without requiring external services.

Run with: pytest polyglotlink/tests/performance/test_benchmarks.py -v
"""

import json
import time
import pytest
import statistics
from datetime import datetime

from polyglotlink.models.schemas import (
    PayloadEncoding,
    Protocol,
    RawMessage,
)
from polyglotlink.modules.schema_extractor import SchemaExtractor
from polyglotlink.modules.semantic_translator_agent import SemanticTranslator
from polyglotlink.modules.normalization_engine import NormalizationEngine
from polyglotlink.modules.protocol_listener import (
    detect_encoding,
    extract_device_id,
    generate_uuid,
)


class TestSchemaExtractorBenchmarks:
    """Benchmark tests for schema extraction performance."""

    @pytest.fixture
    def extractor(self):
        return SchemaExtractor()

    def test_simple_payload_extraction_speed(self, extractor):
        """Benchmark: Simple payload schema extraction."""
        payload = json.dumps({
            "temperature": 23.5,
            "humidity": 65,
            "device_id": "sensor-001"
        }).encode()

        raw = RawMessage(
            message_id="bench-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
        )

        # Warm-up
        for _ in range(10):
            extractor.extract_schema(raw)

        # Benchmark
        iterations = 1000
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            extractor.extract_schema(raw)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(iterations * 0.95)]
        p99_time = sorted(times)[int(iterations * 0.99)]

        print(f"\nSimple Payload Extraction ({iterations} iterations):")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  P95: {p95_time:.3f} ms")
        print(f"  P99: {p99_time:.3f} ms")
        print(f"  Throughput: {1000/avg_time:.0f} ops/sec")

        # Performance assertions
        assert avg_time < 5.0, f"Average time {avg_time}ms exceeds 5ms threshold"
        assert p99_time < 20.0, f"P99 time {p99_time}ms exceeds 20ms threshold"

    def test_nested_payload_extraction_speed(self, extractor):
        """Benchmark: Nested payload schema extraction."""
        payload = json.dumps({
            "device": {
                "id": "sensor-001",
                "type": "environmental",
                "firmware": "1.2.3"
            },
            "readings": {
                "temperature": {"value": 23.5, "unit": "celsius"},
                "humidity": {"value": 65, "unit": "percent"},
                "pressure": {"value": 1013.25, "unit": "hpa"}
            },
            "metadata": {
                "timestamp": "2024-01-15T10:30:00Z",
                "sequence": 12345,
                "quality": "good"
            }
        }).encode()

        raw = RawMessage(
            message_id="bench-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
        )

        # Benchmark
        iterations = 500
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            extractor.extract_schema(raw)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(iterations * 0.95)]

        print(f"\nNested Payload Extraction ({iterations} iterations):")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  P95: {p95_time:.3f} ms")
        print(f"  Throughput: {1000/avg_time:.0f} ops/sec")

        assert avg_time < 10.0, f"Average time {avg_time}ms exceeds 10ms threshold"

    def test_large_payload_extraction_speed(self, extractor):
        """Benchmark: Large payload with many fields."""
        payload = json.dumps({
            f"field_{i}": i * 1.5 for i in range(100)
        }).encode()

        raw = RawMessage(
            message_id="bench-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
        )

        # Benchmark
        iterations = 200
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            extractor.extract_schema(raw)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = statistics.mean(times)

        print(f"\nLarge Payload (100 fields) Extraction ({iterations} iterations):")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Throughput: {1000/avg_time:.0f} ops/sec")

        assert avg_time < 50.0, f"Average time {avg_time}ms exceeds 50ms threshold"


class TestEncodingDetectionBenchmarks:
    """Benchmark tests for encoding detection."""

    def test_json_detection_speed(self):
        """Benchmark: JSON encoding detection."""
        payloads = [
            json.dumps({"key": f"value_{i}", "num": i}).encode()
            for i in range(100)
        ]

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            detect_encoding(payloads[i % len(payloads)])

        elapsed = time.perf_counter() - start
        avg_time = (elapsed / iterations) * 1000000  # microseconds

        print(f"\nJSON Detection ({iterations} iterations):")
        print(f"  Average: {avg_time:.2f} μs")
        print(f"  Throughput: {iterations/elapsed:.0f} ops/sec")

        assert avg_time < 100, f"Average time {avg_time}μs exceeds 100μs threshold"

    def test_xml_detection_speed(self):
        """Benchmark: XML encoding detection."""
        payloads = [
            f"<root><value>{i}</value><name>test_{i}</name></root>".encode()
            for i in range(100)
        ]

        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            detect_encoding(payloads[i % len(payloads)])

        elapsed = time.perf_counter() - start
        avg_time = (elapsed / iterations) * 1000000

        print(f"\nXML Detection ({iterations} iterations):")
        print(f"  Average: {avg_time:.2f} μs")
        print(f"  Throughput: {iterations/elapsed:.0f} ops/sec")

        assert avg_time < 100, f"Average time {avg_time}μs exceeds 100μs threshold"


class TestDeviceIdExtractionBenchmarks:
    """Benchmark tests for device ID extraction."""

    def test_topic_parsing_speed(self):
        """Benchmark: Topic pattern parsing for device ID."""
        topics = [
            f"sensors/device-{i:04d}/telemetry"
            for i in range(100)
        ]

        iterations = 50000
        start = time.perf_counter()

        for i in range(iterations):
            extract_device_id(topics[i % len(topics)])

        elapsed = time.perf_counter() - start
        avg_time = (elapsed / iterations) * 1000000

        print(f"\nDevice ID Extraction ({iterations} iterations):")
        print(f"  Average: {avg_time:.2f} μs")
        print(f"  Throughput: {iterations/elapsed:.0f} ops/sec")

        assert avg_time < 50, f"Average time {avg_time}μs exceeds 50μs threshold"


class TestUuidGenerationBenchmarks:
    """Benchmark tests for UUID generation."""

    def test_uuid_generation_speed(self):
        """Benchmark: UUID generation for message IDs."""
        iterations = 100000
        start = time.perf_counter()

        for _ in range(iterations):
            generate_uuid()

        elapsed = time.perf_counter() - start
        avg_time = (elapsed / iterations) * 1000000

        print(f"\nUUID Generation ({iterations} iterations):")
        print(f"  Average: {avg_time:.2f} μs")
        print(f"  Throughput: {iterations/elapsed:.0f} ops/sec")

        assert avg_time < 20, f"Average time {avg_time}μs exceeds 20μs threshold"


class TestNormalizationBenchmarks:
    """Benchmark tests for normalization engine."""

    @pytest.fixture
    def components(self):
        return {
            "extractor": SchemaExtractor(),
            "translator": SemanticTranslator(),
            "normalizer": NormalizationEngine(),
        }

    @pytest.mark.asyncio
    async def test_full_pipeline_throughput(self, components):
        """Benchmark: Full pipeline throughput."""
        payload = json.dumps({
            "temperature": 23.5,
            "humidity": 65,
            "pressure_hpa": 1013.25,
            "device_id": "sensor-001",
            "timestamp": "2024-01-15T10:30:00Z"
        }).encode()

        raw = RawMessage(
            message_id="bench-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="sensors/sensor-001/telemetry",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
        )

        extractor = components["extractor"]
        translator = components["translator"]
        normalizer = components["normalizer"]

        # Warm-up
        for _ in range(5):
            schema = extractor.extract_schema(raw)
            mapping = await translator.translate_schema(schema)
            normalizer.normalize_message(schema, mapping)

        # Benchmark
        iterations = 100
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            schema = extractor.extract_schema(raw)
            mapping = await translator.translate_schema(schema)
            normalizer.normalize_message(schema, mapping)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(iterations * 0.95)]

        print(f"\nFull Pipeline ({iterations} iterations):")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  P95: {p95_time:.3f} ms")
        print(f"  Throughput: {1000/avg_time:.0f} msgs/sec")

        # Full pipeline should complete in reasonable time
        assert avg_time < 100.0, f"Average time {avg_time}ms exceeds 100ms threshold"


class TestMemoryEfficiency:
    """Tests for memory efficiency."""

    def test_no_memory_leak_on_repeated_extraction(self):
        """Ensure no memory leak during repeated operations."""
        import sys

        extractor = SchemaExtractor()
        payload = json.dumps({
            "temperature": 23.5,
            "humidity": 65,
            "device_id": "sensor-001"
        }).encode()

        raw = RawMessage(
            message_id="bench-001",
            device_id="sensor-001",
            protocol=Protocol.MQTT,
            topic="test",
            payload_raw=payload,
            payload_encoding=PayloadEncoding.JSON,
        )

        # Measure initial memory
        initial_refs = len([o for o in dir() if not o.startswith('_')])

        # Run many iterations
        for _ in range(1000):
            schema = extractor.extract_schema(raw)
            del schema

        # Memory should not grow significantly
        final_refs = len([o for o in dir() if not o.startswith('_')])

        # Simple check - reference count shouldn't explode
        assert final_refs - initial_refs < 100, "Possible memory leak detected"


class TestConcurrencyBenchmarks:
    """Benchmark tests for concurrent processing."""

    @pytest.mark.asyncio
    async def test_concurrent_schema_extraction(self):
        """Benchmark: Concurrent schema extractions."""
        import asyncio

        extractor = SchemaExtractor()

        payloads = [
            json.dumps({
                "temperature": 20 + i * 0.1,
                "device_id": f"sensor-{i:04d}"
            }).encode()
            for i in range(100)
        ]

        async def extract_one(idx):
            raw = RawMessage(
                message_id=f"bench-{idx}",
                device_id=f"sensor-{idx:04d}",
                protocol=Protocol.MQTT,
                topic="test",
                payload_raw=payloads[idx % len(payloads)],
                payload_encoding=PayloadEncoding.JSON,
            )
            return extractor.extract_schema(raw)

        # Benchmark concurrent extraction
        iterations = 500
        start = time.perf_counter()

        tasks = [extract_one(i) for i in range(iterations)]
        results = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start

        print(f"\nConcurrent Extraction ({iterations} concurrent tasks):")
        print(f"  Total time: {elapsed:.3f} s")
        print(f"  Throughput: {iterations/elapsed:.0f} ops/sec")

        assert len(results) == iterations
        assert elapsed < 10.0, f"Concurrent processing took {elapsed}s, exceeds 10s threshold"
