"""
Pytest configuration and shared fixtures.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any

from polyglotlink.models.schemas import (
    ExtractedField,
    ExtractedSchema,
    FieldMapping,
    PayloadEncoding,
    Protocol,
    RawMessage,
    ResolutionMethod,
    SemanticMapping,
)


@pytest.fixture
def sample_json_payload() -> bytes:
    """Sample JSON payload for testing."""
    return json.dumps({
        "temperature": 23.5,
        "humidity": 65,
        "pressure": 1013.25,
        "device_id": "sensor-001",
        "timestamp": "2024-01-15T10:30:00Z"
    }).encode()


@pytest.fixture
def sample_nested_payload() -> bytes:
    """Sample nested JSON payload for testing."""
    return json.dumps({
        "sensor": {
            "readings": {
                "temperature": 23.5,
                "humidity": 65
            },
            "metadata": {
                "location": "room-a",
                "floor": 1
            }
        },
        "device_id": "sensor-001"
    }).encode()


@pytest.fixture
def sample_raw_message(sample_json_payload) -> RawMessage:
    """Sample RawMessage for testing."""
    return RawMessage(
        message_id="test-msg-001",
        device_id="sensor-001",
        protocol=Protocol.MQTT,
        topic="sensors/sensor-001/telemetry",
        payload_raw=sample_json_payload,
        payload_encoding=PayloadEncoding.JSON,
        timestamp=datetime.utcnow(),
        metadata={"broker": "localhost"}
    )


@pytest.fixture
def sample_extracted_fields() -> list[ExtractedField]:
    """Sample extracted fields for testing."""
    return [
        ExtractedField(
            key="temperature",
            original_key="temperature",
            value=23.5,
            value_type="float",
            inferred_unit="celsius",
            inferred_semantic="temperature",
            is_timestamp=False,
            is_identifier=False,
        ),
        ExtractedField(
            key="humidity",
            original_key="humidity",
            value=65,
            value_type="integer",
            inferred_unit="percent",
            inferred_semantic="humidity",
            is_timestamp=False,
            is_identifier=False,
        ),
        ExtractedField(
            key="device_id",
            original_key="device_id",
            value="sensor-001",
            value_type="string",
            inferred_semantic="identifier",
            is_timestamp=False,
            is_identifier=True,
        ),
        ExtractedField(
            key="timestamp",
            original_key="timestamp",
            value="2024-01-15T10:30:00Z",
            value_type="datetime",
            inferred_semantic="timestamp",
            is_timestamp=True,
            is_identifier=False,
        ),
    ]


@pytest.fixture
def sample_extracted_schema(sample_extracted_fields) -> ExtractedSchema:
    """Sample ExtractedSchema for testing."""
    return ExtractedSchema(
        message_id="test-msg-001",
        device_id="sensor-001",
        protocol=Protocol.MQTT,
        topic="sensors/sensor-001/telemetry",
        fields=sample_extracted_fields,
        schema_signature="abc123def456",
        cached_mapping=None,
        payload_decoded={
            "temperature": 23.5,
            "humidity": 65,
            "device_id": "sensor-001",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        extracted_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_field_mappings() -> list[FieldMapping]:
    """Sample field mappings for testing."""
    return [
        FieldMapping(
            source_field="temperature",
            target_concept="temperature_celsius",
            target_field="temperature_celsius",
            source_unit="celsius",
            target_unit="celsius",
            conversion_formula=None,
            confidence=0.95,
            resolution_method=ResolutionMethod.EMBEDDING,
        ),
        FieldMapping(
            source_field="humidity",
            target_concept="humidity_percent",
            target_field="humidity_percent",
            source_unit="percent",
            target_unit="percent",
            conversion_formula=None,
            confidence=0.92,
            resolution_method=ResolutionMethod.EMBEDDING,
        ),
        FieldMapping(
            source_field="device_id",
            target_concept="_identifier",
            target_field="device_id",
            confidence=1.0,
            resolution_method=ResolutionMethod.PASSTHROUGH,
        ),
        FieldMapping(
            source_field="timestamp",
            target_concept="_timestamp",
            target_field="timestamp",
            confidence=1.0,
            resolution_method=ResolutionMethod.PASSTHROUGH,
        ),
    ]


@pytest.fixture
def sample_semantic_mapping(sample_field_mappings) -> SemanticMapping:
    """Sample SemanticMapping for testing."""
    return SemanticMapping(
        message_id="test-msg-001",
        device_id="sensor-001",
        schema_signature="abc123def456",
        field_mappings=sample_field_mappings,
        device_context="Environmental sensor",
        confidence=0.93,
        llm_generated=False,
        translated_at=datetime.utcnow(),
    )


@pytest.fixture
def iot_payloads() -> Dict[str, Dict[str, Any]]:
    """Collection of various IoT device payloads for testing."""
    return {
        "environmental_sensor": {
            "temperature_c": 22.5,
            "humidity_pct": 55,
            "pressure_hpa": 1013.25,
            "co2_ppm": 450,
            "device_id": "env-001",
            "ts": 1705312200
        },
        "power_meter": {
            "voltage_v": 230.5,
            "current_a": 2.3,
            "power_w": 530.15,
            "energy_kwh": 1234.5,
            "device": "meter-001"
        },
        "gps_tracker": {
            "lat": 40.7128,
            "lng": -74.0060,
            "alt_m": 10.5,
            "speed_kmh": 45.2,
            "heading": 180,
            "timestamp": "2024-01-15T10:30:00Z"
        },
        "industrial_plc": {
            "registers": [100, 200, 300],
            "status": 1,
            "alarm": False,
            "setpoint": 75.0,
            "process_value": 74.5
        }
    }


# Async fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
