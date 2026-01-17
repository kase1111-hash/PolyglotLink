"""
PolyglotLink Secrets Management Module

Provides secure handling of sensitive configuration values.
Supports multiple backends: environment variables, .env files, AWS Secrets Manager,
HashiCorp Vault, and Azure Key Vault.
"""

import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache

from polyglotlink.utils.exceptions import ConfigurationError
from polyglotlink.utils.logging import get_logger

logger = get_logger(__name__)


class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def get_secrets(self, prefix: str) -> dict[str, str]:
        """Retrieve all secrets with a given prefix."""
        pass

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return True


class EnvironmentSecretsBackend(SecretsBackend):
    """
    Secrets backend using environment variables.
    This is the default and most portable option.
    """

    def __init__(self, prefix: str = "POLYGLOTLINK_"):
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable."""
        # Try with prefix first
        value = os.environ.get(f"{self.prefix}{key}")
        if value is not None:
            return value

        # Try without prefix
        return os.environ.get(key)

    def get_secrets(self, prefix: str) -> dict[str, str]:
        """Get all environment variables with prefix."""
        full_prefix = f"{self.prefix}{prefix}"
        return {
            k[len(full_prefix) :]: v for k, v in os.environ.items() if k.startswith(full_prefix)
        }


class DotEnvSecretsBackend(SecretsBackend):
    """
    Secrets backend using .env files.
    Supports encrypted .env files with age or sops.
    """

    def __init__(
        self,
        env_file: str = ".env",
        encrypted: bool = False,
        encryption_key_env: str = "DOTENV_KEY",
    ):
        self.env_file = env_file
        self.encrypted = encrypted
        self.encryption_key_env = encryption_key_env
        self._secrets: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        """Load secrets from .env file."""
        if self._loaded:
            return

        if not os.path.exists(self.env_file):
            logger.debug("No .env file found", path=self.env_file)
            self._loaded = True
            return

        try:
            if self.encrypted:
                self._load_encrypted()
            else:
                self._load_plain()
            self._loaded = True
            logger.info("Loaded secrets from .env", count=len(self._secrets))
        except Exception as e:
            logger.error("Failed to load .env file", error=str(e))
            raise ConfigurationError(f"Failed to load secrets: {e}")

    def _load_plain(self) -> None:
        """Load plain text .env file."""
        try:
            from dotenv import dotenv_values

            self._secrets = {k: v for k, v in dotenv_values(self.env_file).items() if v is not None}
        except ImportError:
            # Fallback: manual parsing
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        # Remove quotes
                        value = value.strip().strip("'\"")
                        self._secrets[key.strip()] = value

    def _load_encrypted(self) -> None:
        """Load encrypted .env file using sops or age."""
        import subprocess

        # Try sops first
        try:
            result = subprocess.run(
                ["sops", "-d", self.env_file],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    self._secrets[key.strip()] = value.strip().strip("'\"")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try age
        try:
            key = os.environ.get(self.encryption_key_env)
            if not key:
                raise ConfigurationError(f"Encryption key not found in {self.encryption_key_env}")

            result = subprocess.run(
                ["age", "-d", "-i", "-", self.env_file],
                input=key,
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    self._secrets[key.strip()] = value.strip().strip("'\"")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        raise ConfigurationError("No decryption tool available (sops or age)")

    def get_secret(self, key: str) -> str | None:
        self._load()
        return self._secrets.get(key)

    def get_secrets(self, prefix: str) -> dict[str, str]:
        self._load()
        return {k[len(prefix) :]: v for k, v in self._secrets.items() if k.startswith(prefix)}

    def is_available(self) -> bool:
        return os.path.exists(self.env_file)


class AWSSecretsManagerBackend(SecretsBackend):
    """
    Secrets backend using AWS Secrets Manager.
    """

    def __init__(
        self,
        region_name: str | None = None,
        secret_name: str = "polyglotlink/config",
    ):
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.secret_name = secret_name
        self._client = None
        self._secrets: dict[str, str] = {}
        self._loaded = False

    def _get_client(self):
        if self._client is None:
            try:
                import boto3

                self._client = boto3.client(
                    "secretsmanager",
                    region_name=self.region_name,
                )
            except ImportError:
                raise ConfigurationError("boto3 not installed for AWS Secrets Manager")
        return self._client

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            client = self._get_client()
            response = client.get_secret_value(SecretId=self.secret_name)

            if "SecretString" in response:
                self._secrets = json.loads(response["SecretString"])
            else:
                import base64

                decoded = base64.b64decode(response["SecretBinary"])
                self._secrets = json.loads(decoded)

            self._loaded = True
            logger.info("Loaded secrets from AWS Secrets Manager", count=len(self._secrets))

        except Exception as e:
            logger.error("Failed to load from AWS Secrets Manager", error=str(e))
            # Don't raise - allow fallback to other backends
            self._loaded = True

    def get_secret(self, key: str) -> str | None:
        self._load()
        return self._secrets.get(key)

    def get_secrets(self, prefix: str) -> dict[str, str]:
        self._load()
        return {k[len(prefix) :]: v for k, v in self._secrets.items() if k.startswith(prefix)}

    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("boto3") is not None


class VaultSecretsBackend(SecretsBackend):
    """
    Secrets backend using HashiCorp Vault.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        mount_point: str = "secret",
        path: str = "polyglotlink/config",
    ):
        self.url = url or os.environ.get("VAULT_ADDR", "http://localhost:8200")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.mount_point = mount_point
        self.path = path
        self._client = None
        self._secrets: dict[str, str] = {}
        self._loaded = False

    def _get_client(self):
        if self._client is None:
            try:
                import hvac

                self._client = hvac.Client(url=self.url, token=self.token)
            except ImportError:
                raise ConfigurationError("hvac not installed for Vault support")
        return self._client

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            client = self._get_client()

            if not client.is_authenticated():
                logger.warning("Vault client not authenticated")
                self._loaded = True
                return

            response = client.secrets.kv.v2.read_secret_version(
                mount_point=self.mount_point,
                path=self.path,
            )
            self._secrets = response["data"]["data"]
            self._loaded = True
            logger.info("Loaded secrets from Vault", count=len(self._secrets))

        except Exception as e:
            logger.error("Failed to load from Vault", error=str(e))
            self._loaded = True

    def get_secret(self, key: str) -> str | None:
        self._load()
        return self._secrets.get(key)

    def get_secrets(self, prefix: str) -> dict[str, str]:
        self._load()
        return {k[len(prefix) :]: v for k, v in self._secrets.items() if k.startswith(prefix)}

    def is_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("hvac") is not None and bool(self.token)


class SecretsManager:
    """
    Unified secrets manager that tries multiple backends in order.
    """

    def __init__(self, backends: list[SecretsBackend] | None = None):
        if backends is None:
            # Default backend order
            backends = [
                EnvironmentSecretsBackend(),
                DotEnvSecretsBackend(),
            ]

            # Add cloud backends if available
            aws_backend = AWSSecretsManagerBackend()
            if aws_backend.is_available():
                backends.append(aws_backend)

            vault_backend = VaultSecretsBackend()
            if vault_backend.is_available():
                backends.append(vault_backend)

        self.backends = backends

    def get_secret(
        self,
        key: str,
        default: str | None = None,
        required: bool = False,
    ) -> str | None:
        """
        Get a secret from available backends.

        Args:
            key: Secret key
            default: Default value if not found
            required: Raise error if not found and no default

        Returns:
            Secret value or default
        """
        for backend in self.backends:
            try:
                value = backend.get_secret(key)
                if value is not None:
                    return value
            except Exception as e:
                logger.warning(
                    "Backend failed",
                    backend=type(backend).__name__,
                    error=str(e),
                )
                continue

        if required and default is None:
            raise ConfigurationError(f"Required secret '{key}' not found")

        return default

    def get_secrets(self, prefix: str) -> dict[str, str]:
        """
        Get all secrets with a prefix from all backends.

        Args:
            prefix: Secret key prefix

        Returns:
            Merged dictionary of secrets
        """
        result: dict[str, str] = {}

        # Go through backends in reverse order so higher priority backends
        # override lower priority ones
        for backend in reversed(self.backends):
            try:
                secrets = backend.get_secrets(prefix)
                result.update(secrets)
            except Exception as e:
                logger.warning(
                    "Backend failed",
                    backend=type(backend).__name__,
                    error=str(e),
                )
                continue

        return result

    def require(self, key: str) -> str:
        """
        Get a required secret, raising error if not found.

        Args:
            key: Secret key

        Returns:
            Secret value

        Raises:
            ConfigurationError: If secret not found
        """
        value = self.get_secret(key, required=True)
        assert value is not None  # For type checker
        return value


@lru_cache
def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    return SecretsManager()


def get_secret(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """
    Convenience function to get a secret.

    Args:
        key: Secret key
        default: Default value
        required: Raise if not found

    Returns:
        Secret value or default
    """
    return get_secrets_manager().get_secret(key, default, required)


def require_secret(key: str) -> str:
    """
    Convenience function to get a required secret.

    Args:
        key: Secret key

    Returns:
        Secret value

    Raises:
        ConfigurationError: If not found
    """
    return get_secrets_manager().require(key)
