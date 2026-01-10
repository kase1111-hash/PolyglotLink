"""
PolyglotLink Core Modules

This package contains the core processing modules for the PolyglotLink system:
- protocol_listener: Multi-protocol IoT message ingestion
- schema_extractor: Schema extraction and type inference
- semantic_translator_agent: LLM-based semantic mapping
- normalization_engine: Unit conversion and value normalization
- output_broker: Multi-destination message publishing
"""

from polyglotlink.modules.protocol_listener import (
    BaseProtocolHandler,
    CoAPHandler,
    HTTPHandler,
    ModbusHandler,
    MQTTHandler,
    OPCUAHandler,
    ProtocolListener,
    WebSocketHandler,
    detect_encoding,
    extract_device_id,
    generate_uuid,
)
from polyglotlink.modules.schema_extractor import (
    SchemaCache,
    SchemaExtractor,
    UnsupportedEncodingError,
    detect_type,
    flatten_dict,
    generate_schema_hash,
    infer_semantic_hint,
    infer_unit_from_key,
    is_identifier_field,
    is_timestamp_field,
)
from polyglotlink.modules.semantic_translator_agent import (
    DEFAULT_ONTOLOGY_CONCEPTS,
    EmbeddingResolver,
    LLMTranslator,
    SemanticTranslator,
    build_fields_table,
    build_ontology_context,
)
from polyglotlink.modules.normalization_engine import (
    ConversionError,
    DeviceInfo,
    DeviceRegistry,
    NormalizationEngine,
    UnsafeFormulaError,
    apply_conversion,
    enforce_type,
    enrich_metadata,
    extract_timestamp,
    get_unit_conversion,
    validate_value,
)
from polyglotlink.modules.output_broker import (
    HTTPOutputConfig,
    KafkaOutputConfig,
    MQTTOutputConfig,
    OutputBroker,
    OutputBrokerConfig,
    OutputRouting,
    PublishResult,
    TimescaleOutputConfig,
    TopicMapper,
    WebSocketManager,
    WebSocketOutputConfig,
)

__all__ = [
    # Protocol Listener
    "BaseProtocolHandler",
    "CoAPHandler",
    "HTTPHandler",
    "ModbusHandler",
    "MQTTHandler",
    "OPCUAHandler",
    "ProtocolListener",
    "WebSocketHandler",
    "detect_encoding",
    "extract_device_id",
    "generate_uuid",
    # Schema Extractor
    "SchemaCache",
    "SchemaExtractor",
    "UnsupportedEncodingError",
    "detect_type",
    "flatten_dict",
    "generate_schema_hash",
    "infer_semantic_hint",
    "infer_unit_from_key",
    "is_identifier_field",
    "is_timestamp_field",
    # Semantic Translator
    "DEFAULT_ONTOLOGY_CONCEPTS",
    "EmbeddingResolver",
    "LLMTranslator",
    "SemanticTranslator",
    "build_fields_table",
    "build_ontology_context",
    # Normalization Engine
    "ConversionError",
    "DeviceInfo",
    "DeviceRegistry",
    "NormalizationEngine",
    "UnsafeFormulaError",
    "apply_conversion",
    "enforce_type",
    "enrich_metadata",
    "extract_timestamp",
    "get_unit_conversion",
    "validate_value",
    # Output Broker
    "HTTPOutputConfig",
    "KafkaOutputConfig",
    "MQTTOutputConfig",
    "OutputBroker",
    "OutputBrokerConfig",
    "OutputRouting",
    "PublishResult",
    "TimescaleOutputConfig",
    "TopicMapper",
    "WebSocketManager",
    "WebSocketOutputConfig",
]
