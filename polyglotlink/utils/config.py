"""
PolyglotLink Configuration Module

Centralized configuration management with validation using Pydantic Settings.
Loads configuration from environment variables and .env files.
"""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM and embedding configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for LLM and embeddings",
    )
    model: str = Field(default="gpt-4o", description="LLM model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    # Embedding settings
    embedding_model: str = Field(
        default="text-embedding-3-large", validation_alias="EMBEDDING_MODEL"
    )
    embedding_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, validation_alias="EMBEDDING_THRESHOLD"
    )


class MQTTListenerSettings(BaseSettings):
    """MQTT listener configuration."""

    model_config = SettingsConfigDict(env_prefix="MQTT_")

    enabled: bool = Field(default=True)
    broker_host: str = Field(default="localhost")
    broker_port: int = Field(default=1883, ge=1, le=65535)
    client_id: str = Field(default="polyglotlink-listener")
    username: str | None = Field(default=None)
    password: str | None = Field(default=None)
    tls_enabled: bool = Field(default=False)
    ca_cert: str | None = Field(default=None)
    topic_patterns: list[str] = Field(default=["sensors/#", "devices/+/telemetry"])
    qos: int = Field(default=1, ge=0, le=2)

    @field_validator("topic_patterns", mode="before")
    @classmethod
    def parse_topic_patterns(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @model_validator(mode="after")
    def validate_tls(self):
        if self.tls_enabled and not self.ca_cert:
            raise ValueError("ca_cert is required when tls_enabled is True")
        return self


class HTTPListenerSettings(BaseSettings):
    """HTTP listener configuration."""

    model_config = SettingsConfigDict(env_prefix="HTTP_")

    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")  # nosec B104 - intentional for server
    port: int = Field(default=8080, ge=1, le=65535)
    path_prefix: str = Field(default="/ingest")

    @field_validator("path_prefix")
    @classmethod
    def validate_path_prefix(cls, v: str) -> str:
        if not v.startswith("/"):
            return f"/{v}"
        return v


class CoAPListenerSettings(BaseSettings):
    """CoAP listener configuration."""

    model_config = SettingsConfigDict(env_prefix="COAP_")

    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")  # nosec B104 - intentional for server
    port: int = Field(default=5683, ge=1, le=65535)


class ModbusListenerSettings(BaseSettings):
    """Modbus listener configuration."""

    model_config = SettingsConfigDict(env_prefix="MODBUS_")

    enabled: bool = Field(default=False)
    host: str = Field(default="192.168.1.100")
    port: int = Field(default=502, ge=1, le=65535)
    poll_interval_seconds: int = Field(default=5, ge=1, le=3600)


class OPCUAListenerSettings(BaseSettings):
    """OPC-UA listener configuration."""

    model_config = SettingsConfigDict(env_prefix="OPCUA_")

    enabled: bool = Field(default=False)
    endpoint_url: str = Field(default="opc.tcp://localhost:4840")
    security_policy: str | None = Field(default=None)
    subscription_interval_ms: int = Field(default=1000, ge=100, le=60000)


class WebSocketListenerSettings(BaseSettings):
    """WebSocket listener configuration."""

    model_config = SettingsConfigDict(env_prefix="WEBSOCKET_")

    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")  # nosec B104 - intentional for server
    port: int = Field(default=8081, ge=1, le=65535)


class RedisSettings(BaseSettings):
    """Redis configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=10, ge=1, le=100)


class Neo4jSettings(BaseSettings):
    """Neo4j configuration."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_")

    uri: str = Field(default="neo4j://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="polyglotlink")


class WeaviateSettings(BaseSettings):
    """Weaviate configuration."""

    model_config = SettingsConfigDict(env_prefix="WEAVIATE_")

    url: str = Field(default="http://localhost:8085")


class TimescaleSettings(BaseSettings):
    """TimescaleDB configuration."""

    model_config = SettingsConfigDict(env_prefix="TIMESCALE_")

    url: str = Field(default="postgresql://postgres:postgres@localhost:5432/iot")
    pool_min_size: int = Field(default=1, ge=1, le=10)
    pool_max_size: int = Field(default=10, ge=1, le=100)


class KafkaOutputSettings(BaseSettings):
    """Kafka output configuration."""

    model_config = SettingsConfigDict(env_prefix="KAFKA_")

    enabled: bool = Field(default=True)
    bootstrap_servers: str = Field(default="localhost:9092")
    topic_prefix: str = Field(default="iot.normalized")
    acks: str = Field(default="all")
    compression_type: str = Field(default="gzip")


class MetricsSettings(BaseSettings):
    """Prometheus metrics configuration."""

    model_config = SettingsConfigDict(env_prefix="METRICS_")

    enabled: bool = Field(default=True)
    port: int = Field(default=9090, ge=1, le=65535)


class SentrySettings(BaseSettings):
    """Sentry error tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="SENTRY_")

    dsn: str | None = Field(default=None)
    environment: str = Field(default="development")
    traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    env: str = Field(default="development", validation_alias="POLYGLOTLINK_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    debug: bool = Field(default=False, validation_alias="DEBUG")

    # Sub-configurations
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mqtt: MQTTListenerSettings = Field(default_factory=MQTTListenerSettings)
    http: HTTPListenerSettings = Field(default_factory=HTTPListenerSettings)
    coap: CoAPListenerSettings = Field(default_factory=CoAPListenerSettings)
    modbus: ModbusListenerSettings = Field(default_factory=ModbusListenerSettings)
    opcua: OPCUAListenerSettings = Field(default_factory=OPCUAListenerSettings)
    websocket: WebSocketListenerSettings = Field(default_factory=WebSocketListenerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    timescale: TimescaleSettings = Field(default_factory=TimescaleSettings)
    kafka: KafkaOutputSettings = Field(default_factory=KafkaOutputSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    sentry: SentrySettings = Field(default_factory=SentrySettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production", "test"]
        if v.lower() not in valid_envs:
            raise ValueError(f"env must be one of {valid_envs}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        return self.env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    Settings are loaded once and cached for performance.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings (clears cache).
    Use this when environment variables have changed.
    """
    get_settings.cache_clear()
    return get_settings()
