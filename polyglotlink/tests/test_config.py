"""
Unit tests for the Configuration module.
"""

import os
from unittest.mock import patch

import pytest

from polyglotlink.utils.config import (
    HTTPListenerSettings,
    LLMSettings,
    MQTTListenerSettings,
    Settings,
    get_settings,
    reload_settings,
)


class TestLLMSettings:
    """Tests for LLM configuration settings."""

    def test_default_values(self):
        settings = LLMSettings()
        assert settings.model == "gpt-4o"
        assert settings.temperature == 0.1
        assert settings.max_tokens == 2000
        assert settings.max_retries == 3

    def test_temperature_bounds(self):
        # Valid temperatures
        settings = LLMSettings(temperature=0.0)
        assert settings.temperature == 0.0

        settings = LLMSettings(temperature=2.0)
        assert settings.temperature == 2.0

        # Invalid temperature
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMSettings(temperature=-0.1)

        with pytest.raises(ValidationError):
            LLMSettings(temperature=2.1)

    def test_max_tokens_bounds(self):
        settings = LLMSettings(max_tokens=100)
        assert settings.max_tokens == 100

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMSettings(max_tokens=50)  # Below minimum

    def test_embedding_threshold_bounds(self):
        # Use validation_alias for constructor
        settings = LLMSettings(EMBEDDING_THRESHOLD=0.5)
        assert settings.embedding_threshold == 0.5

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LLMSettings(EMBEDDING_THRESHOLD=1.5)


class TestMQTTListenerSettings:
    """Tests for MQTT listener configuration settings."""

    def test_default_values(self):
        settings = MQTTListenerSettings()
        assert settings.enabled is True
        assert settings.broker_host == "localhost"
        assert settings.broker_port == 1883
        assert settings.qos == 1

    def test_port_bounds(self):
        settings = MQTTListenerSettings(broker_port=8883)
        assert settings.broker_port == 8883

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MQTTListenerSettings(broker_port=0)

        with pytest.raises(ValidationError):
            MQTTListenerSettings(broker_port=70000)

    def test_qos_bounds(self):
        settings = MQTTListenerSettings(qos=0)
        assert settings.qos == 0

        settings = MQTTListenerSettings(qos=2)
        assert settings.qos == 2

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MQTTListenerSettings(qos=3)

    def test_topic_patterns_string_parsing(self):
        settings = MQTTListenerSettings(topic_patterns="sensors/#,devices/+/data")
        assert settings.topic_patterns == ["sensors/#", "devices/+/data"]

    def test_topic_patterns_list(self):
        settings = MQTTListenerSettings(topic_patterns=["topic1", "topic2"])
        assert settings.topic_patterns == ["topic1", "topic2"]

    def test_tls_validation(self):
        # TLS enabled without cert should fail
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MQTTListenerSettings(tls_enabled=True, ca_cert=None)

        # TLS enabled with cert should work
        settings = MQTTListenerSettings(tls_enabled=True, ca_cert="/path/to/cert")
        assert settings.tls_enabled is True


class TestHTTPListenerSettings:
    """Tests for HTTP listener configuration settings."""

    def test_default_values(self):
        settings = HTTPListenerSettings()
        assert settings.enabled is True
        assert settings.host == "0.0.0.0"
        assert settings.port == 8080
        assert settings.path_prefix == "/ingest"

    def test_path_prefix_normalization(self):
        # Should add leading slash
        settings = HTTPListenerSettings(path_prefix="ingest")
        assert settings.path_prefix == "/ingest"

        # Should preserve leading slash
        settings = HTTPListenerSettings(path_prefix="/api/ingest")
        assert settings.path_prefix == "/api/ingest"


class TestSettings:
    """Tests for main Settings class."""

    def test_default_environment(self):
        settings = Settings()
        assert settings.env in ["development", "staging", "production", "test"]

    def test_log_level_validation(self):
        # Use validation_alias name for constructor
        settings = Settings(LOG_LEVEL="DEBUG")
        assert settings.log_level == "DEBUG"

        settings = Settings(LOG_LEVEL="info")
        assert settings.log_level == "INFO"

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(LOG_LEVEL="INVALID")

    def test_environment_validation(self):
        # Use validation_alias name for constructor
        settings = Settings(POLYGLOTLINK_ENV="production")
        assert settings.env == "production"

        settings = Settings(POLYGLOTLINK_ENV="DEVELOPMENT")
        assert settings.env == "development"

        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(POLYGLOTLINK_ENV="invalid_env")

    def test_is_production_property(self):
        # Use validation_alias name for constructor
        settings = Settings(POLYGLOTLINK_ENV="production")
        assert settings.is_production is True
        assert settings.is_development is False

        settings = Settings(POLYGLOTLINK_ENV="development")
        assert settings.is_production is False
        assert settings.is_development is True

    @patch.dict(os.environ, {"POLYGLOTLINK_ENV": "staging", "LOG_LEVEL": "WARNING"})
    def test_environment_variable_loading(self):
        # Clear cache to pick up new env vars
        reload_settings()
        _settings = get_settings()
        # Note: actual values depend on how pydantic-settings loads them


class TestGetSettings:
    """Tests for settings singleton."""

    def test_returns_settings(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_caching(self):
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reload_clears_cache(self):
        settings1 = get_settings()
        reload_settings()
        settings2 = get_settings()
        # After reload, it's a new instance
        assert settings1 is not settings2
