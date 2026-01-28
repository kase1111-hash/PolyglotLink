"""
PolyglotLink Pydantic Models

This module defines all data models used throughout the PolyglotLink system
for IoT message processing and semantic translation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Enums
# ============================================================================


class Protocol(str, Enum):
    """Supported IoT protocols."""

    MQTT = "MQTT"
    COAP = "CoAP"
    MODBUS = "Modbus"
    OPCUA = "OPC-UA"
    HTTP = "HTTP"
    WEBSOCKET = "WebSocket"
    SDR = "SDR"  # Software Defined Radio


class PayloadEncoding(str, Enum):
    """Detected payload encoding types."""

    JSON = "json"
    XML = "xml"
    CBOR = "cbor"
    CSV = "csv"
    PROTOBUF = "protobuf"
    MODBUS_REGISTERS = "modbus_registers"
    BINARY = "binary"
    # SDR protocol decodings
    SDR_ADSB = "sdr_adsb"
    SDR_POCSAG = "sdr_pocsag"
    SDR_APRS = "sdr_aprs"
    SDR_ACARS = "sdr_acars"
    SDR_RDS = "sdr_rds"
    SDR_FLEX = "sdr_flex"
    SDR_IQ = "sdr_iq"


class ResolutionMethod(str, Enum):
    """Methods used to resolve semantic mappings."""

    CACHE = "cache"
    EMBEDDING = "embedding"
    LLM = "llm"
    PASSTHROUGH = "passthrough"


class MappingSource(str, Enum):
    """Source of a cached mapping."""

    LLM = "llm"
    MANUAL = "manual"
    LEARNED = "learned"


class ValidationErrorType(str, Enum):
    """Types of validation errors."""

    CONVERSION_FAILED = "conversion_failed"
    TYPE_MISMATCH = "type_mismatch"
    OUT_OF_RANGE = "out_of_range"


# ============================================================================
# Protocol Listener Models
# ============================================================================


class RawMessage(BaseModel):
    """
    Raw message received from a protocol listener.
    Contains the unprocessed payload and protocol-specific metadata.
    """

    message_id: str = Field(..., description="UUID for this message")
    device_id: str = Field(..., description="Extracted device identifier")
    protocol: Protocol = Field(..., description="Source protocol")
    topic: str = Field(..., description="Topic/path/node identifier")
    payload_raw: bytes = Field(..., description="Original payload bytes")
    payload_encoding: PayloadEncoding = Field(..., description="Detected format")
    timestamp: datetime = Field(..., description="Receive timestamp")
    qos: int | None = Field(None, description="MQTT QoS level")
    retained: bool | None = Field(None, description="MQTT retained flag")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Protocol-specific metadata")


# ============================================================================
# Schema Extractor Models
# ============================================================================


class ExtractedField(BaseModel):
    """
    A single field extracted from a payload.
    Contains type inference and semantic hints.
    """

    key: str = Field(..., description="Flattened key (e.g., 'sensor.temp')")
    original_key: str = Field(..., description="Original key name")
    value: Any = Field(..., description="Actual value")
    value_type: str = Field(..., description="Detected type")
    inferred_unit: str | None = Field(None, description="Heuristic unit guess")
    inferred_semantic: str | None = Field(None, description="Semantic category hint")
    is_timestamp: bool = Field(False, description="Likely a timestamp field")
    is_identifier: bool = Field(False, description="Likely an ID field")


class FieldMapping(BaseModel):
    """
    Mapping from a source field to a target ontology concept.
    """

    source_field: str = Field(..., description="Original field name")
    target_concept: str = Field(..., description="Ontology concept ID")
    target_field: str = Field(..., description="Standardized field name")
    source_unit: str | None = Field(None, description="Detected source unit")
    target_unit: str | None = Field(None, description="Target standard unit")
    conversion_formula: str | None = Field(None, description="Unit conversion formula")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Mapping confidence")
    resolution_method: ResolutionMethod = Field(..., description="How the mapping was resolved")
    reasoning: str | None = Field(None, description="LLM explanation")


class CachedMapping(BaseModel):
    """
    A cached schema-to-concept mapping for fast lookup.
    """

    schema_signature: str = Field(..., description="Hash of the schema structure")
    field_mappings: list[FieldMapping] = Field(..., description="List of field mappings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    created_at: datetime = Field(..., description="When the mapping was created")
    source: MappingSource = Field(..., description="Source of the mapping")
    hit_count: int = Field(0, description="Number of cache hits")


class ExtractedSchema(BaseModel):
    """
    Schema extracted from a raw message payload.
    Ready for semantic translation.
    """

    message_id: str = Field(..., description="Original message ID")
    device_id: str = Field(..., description="Device identifier")
    protocol: Protocol = Field(..., description="Source protocol")
    topic: str = Field(..., description="Topic/path/node identifier")
    fields: list[ExtractedField] = Field(..., description="Extracted fields")
    schema_signature: str = Field(..., description="Hash for caching")
    cached_mapping: CachedMapping | None = Field(None, description="Pre-existing mapping if found")
    payload_decoded: dict[str, Any] = Field(..., description="Original decoded payload")
    extracted_at: datetime = Field(..., description="Extraction timestamp")


# ============================================================================
# Semantic Translator Models
# ============================================================================


class SuggestedConcept(BaseModel):
    """
    A concept suggested by the LLM for addition to the ontology.
    """

    concept_id: str = Field(..., description="Proposed concept identifier")
    description: str = Field(..., description="Human-readable description")
    unit: str = Field(..., description="Standard unit")
    datatype: str = Field(..., description="Data type")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")


class SemanticMapping(BaseModel):
    """
    Complete semantic mapping for a message.
    Maps all fields to ontology concepts.
    """

    message_id: str = Field(..., description="Original message ID")
    device_id: str = Field(..., description="Device identifier")
    schema_signature: str = Field(..., description="Schema hash")
    field_mappings: list[FieldMapping] = Field(..., description="All field mappings")
    device_context: str | None = Field(None, description="Inferred device type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall mapping confidence")
    llm_generated: bool = Field(False, description="True if LLM was used")
    translated_at: datetime = Field(..., description="Translation timestamp")


# ============================================================================
# Normalization Engine Models
# ============================================================================


class ConversionRecord(BaseModel):
    """
    Record of a unit conversion performed during normalization.
    """

    field: str = Field(..., description="Field name")
    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    original_value: Any = Field(..., description="Value before conversion")
    converted_value: Any = Field(..., description="Value after conversion")


class ValidationError(BaseModel):
    """
    Record of a validation error during normalization.
    """

    field: str = Field(..., description="Field name")
    error: ValidationErrorType = Field(..., description="Error type")
    details: str | None = Field(None, description="Error details")
    expected: str | None = Field(None, description="Expected value/type")
    actual: str | None = Field(None, description="Actual value/type")
    value: Any | None = Field(None, description="The problematic value")
    min: float | None = Field(None, description="Minimum allowed value")
    max: float | None = Field(None, description="Maximum allowed value")


class NormalizedMessage(BaseModel):
    """
    Fully normalized and enriched message ready for output.
    """

    message_id: str = Field(..., description="Original message ID")
    device_id: str = Field(..., description="Device identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: dict[str, Any] = Field(..., description="Normalized field values")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Enriched metadata")
    context: str | None = Field(None, description="Inferred device context")
    schema_signature: str = Field(..., description="Schema hash")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    conversions: list[ConversionRecord] = Field(
        default_factory=list, description="Conversion records"
    )
    validation_errors: list[ValidationError] = Field(
        default_factory=list, description="Validation errors"
    )
    normalized_at: datetime = Field(..., description="Normalization timestamp")


# ============================================================================
# Configuration Models
# ============================================================================


class MQTTConfig(BaseModel):
    """MQTT listener configuration."""

    enabled: bool = True
    broker_host: str = "localhost"
    broker_port: int = 1883
    client_id: str = "polyglotlink-listener"
    username: str | None = None
    password: str | None = None
    tls_enabled: bool = False
    ca_cert: str | None = None
    topic_patterns: list[str] = Field(default_factory=lambda: ["sensors/#", "devices/+/telemetry"])
    qos: int = 1


class CoAPConfig(BaseModel):
    """CoAP listener configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"  # nosec B104 - binding to all interfaces is intentional for server
    port: int = 5683


class ModbusDeviceConfig(BaseModel):
    """Configuration for a single Modbus device."""

    device_id: str
    slave_id: int
    register_blocks: list[dict[str, Any]]


class ModbusConfig(BaseModel):
    """Modbus listener configuration."""

    enabled: bool = False
    host: str = "192.168.1.100"
    port: int = 502
    poll_interval_seconds: int = 5
    devices: list[ModbusDeviceConfig] = Field(default_factory=list)


class OPCUAConfig(BaseModel):
    """OPC-UA listener configuration."""

    enabled: bool = False
    endpoint_url: str = "opc.tcp://localhost:4840"
    security_policy: str | None = None
    subscription_interval_ms: int = 1000
    monitored_nodes: list[str] = Field(default_factory=list)


class HTTPConfig(BaseModel):
    """HTTP webhook listener configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"  # nosec B104 - binding to all interfaces is intentional for server
    port: int = 8080
    path_prefix: str = "/ingest"


class WebSocketConfig(BaseModel):
    """WebSocket listener configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"  # nosec B104 - binding to all interfaces is intentional for server
    port: int = 8081


class SDRDeviceConfig(BaseModel):
    """Configuration for a single SDR device."""

    device_type: str = "rtlsdr"  # "rtlsdr" or "hackrf"
    frequency_hz: float = 433.92e6  # Default to 433.92 MHz (ISM band)
    sample_rate: int = 2_400_000  # 2.4 MHz default
    gain: float = 40.0  # dB
    ppm_correction: int = 0
    agc_enabled: bool = True


class SDRDecoderConfig(BaseModel):
    """Configuration for SDR protocol decoders."""

    adsb_enabled: bool = False  # ADS-B aircraft tracking (1090 MHz)
    adsb_frequency: float = 1090e6
    pocsag_enabled: bool = False  # Pager decoding
    pocsag_frequency: float = 152.84e6
    aprs_enabled: bool = False  # Amateur packet radio
    aprs_frequency: float = 144.39e6
    acars_enabled: bool = False  # Aircraft communications
    acars_frequencies: list[float] = Field(default_factory=lambda: [129.125e6, 130.025e6])
    rds_enabled: bool = False  # FM broadcast RDS
    flex_enabled: bool = False  # FLEX pager
    custom_frequency: float | None = None  # Custom frequency for raw IQ


class SDRConfig(BaseModel):
    """SDR (Software Defined Radio) listener configuration."""

    enabled: bool = False
    rtlsdr: SDRDeviceConfig = Field(default_factory=SDRDeviceConfig)
    hackrf: SDRDeviceConfig = Field(
        default_factory=lambda: SDRDeviceConfig(device_type="hackrf", sample_rate=8_000_000)
    )
    decoders: SDRDecoderConfig = Field(default_factory=SDRDecoderConfig)
    buffer_size: int = 262144  # IQ sample buffer size
    spectrum_enabled: bool = True  # Enable spectrum analysis
    signal_classification: bool = True  # Auto-classify detected signals


class ProtocolListenerConfig(BaseModel):
    """Complete protocol listener configuration."""

    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    coap: CoAPConfig = Field(default_factory=CoAPConfig)
    modbus: ModbusConfig = Field(default_factory=ModbusConfig)
    opcua: OPCUAConfig = Field(default_factory=OPCUAConfig)
    http: HTTPConfig = Field(default_factory=HTTPConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    sdr: SDRConfig = Field(default_factory=SDRConfig)


class SchemaExtractorConfig(BaseModel):
    """Schema extractor configuration."""

    max_nesting_depth: int = 10
    max_array_sample_size: int = 5
    cache_ttl_days: int = 30
    enable_unit_inference: bool = True
    enable_semantic_hints: bool = True
    flatten_arrays: bool = True
    preserve_null_fields: bool = False


class SemanticTranslatorConfig(BaseModel):
    """Semantic translator configuration."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    embedding_model: str = "text-embedding-3-large"
    embedding_threshold: float = 0.85
    max_llm_retries: int = 3
    timeout_seconds: int = 30
    enable_concept_learning: bool = True
    min_confidence_threshold: float = 0.6
    cache_embeddings: bool = True
    vector_store: str = "weaviate"


class NormalizationConfig(BaseModel):
    """Normalization engine configuration."""

    null_strategy: str = "preserve"  # "preserve" | "omit" | "default"
    default_values: dict[str, Any] = Field(
        default_factory=lambda: {"float": 0.0, "integer": 0, "string": "", "boolean": False}
    )
    precision: dict[str, int] = Field(
        default_factory=lambda: {"float": 6, "temperature": 2, "percentage": 1}
    )
    include_unmapped: bool = True
    include_invalid: bool = True
    timestamp_field_names: list[str] = Field(
        default_factory=lambda: ["timestamp", "ts", "time", "datetime", "created_at"]
    )
    device_registry_enabled: bool = True
