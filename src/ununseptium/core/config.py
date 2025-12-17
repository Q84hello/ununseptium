"""Configuration management using Pydantic Settings.

Provides centralized configuration with environment variable support,
YAML/TOML file loading, and validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "console"
    include_timestamp: bool = True
    include_caller: bool = False


class SecuritySettings(BaseModel):
    """Security-related configuration."""

    enable_audit_logging: bool = True
    audit_log_path: Path = Path("audit.log")
    hash_algorithm: Literal["sha256", "sha384", "sha512"] = "sha256"
    enable_pii_masking: bool = True
    encryption_key_env: str = "UNUNSEPTIUM_ENCRYPTION_KEY"


class ModelSettings(BaseModel):
    """AI model configuration."""

    cache_dir: Path = Path.home() / ".cache" / "ununseptium" / "models"
    download_timeout: int = Field(default=300, ge=30, le=3600)
    verify_checksums: bool = True
    default_device: Literal["cpu", "cuda", "mps", "auto"] = "auto"


class MonitoringSettings(BaseModel):
    """Transaction monitoring configuration."""

    alert_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    enable_drift_detection: bool = True
    drift_threshold: float = Field(default=0.05, ge=0.01, le=0.5)


class Settings(BaseSettings):
    """Main application settings.

    Configuration is loaded from environment variables with the prefix
    `UNUNSEPTIUM_`. Nested settings use double underscores.

    Example:
        ```bash
        export UNUNSEPTIUM_LOGGING__LEVEL=DEBUG
        export UNUNSEPTIUM_SECURITY__ENABLE_AUDIT_LOGGING=true
        ```

    Attributes:
        environment: Deployment environment (development, staging, production).
        debug: Enable debug mode with verbose output.
        logging: Logging configuration.
        security: Security settings.
        models: AI model settings.
        monitoring: Transaction monitoring settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="UNUNSEPTIUM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)


def load_config(
    config_path: Path | str | None = None,
    *,
    env_file: Path | str | None = None,
) -> Settings:
    """Load configuration from file and/or environment variables.

    Args:
        config_path: Path to YAML or TOML configuration file.
        env_file: Path to .env file for environment variables.

    Returns:
        Loaded and validated Settings instance.

    Raises:
        ConfigurationError: If configuration file cannot be loaded or validated.

    Example:
        ```python
        from ununseptium.core import load_config

        # Load from environment only
        settings = load_config()

        # Load from YAML file
        settings = load_config("config.yaml")

        # Load with .env file
        settings = load_config(env_file=".env")
        ```
    """
    from ununseptium.core.errors import ConfigurationError

    config_data: dict[str, Any] = {}

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise ConfigurationError(msg)

        try:
            if config_path.suffix in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as err:
                    msg = "PyYAML is required to load YAML configuration files"
                    raise ConfigurationError(msg) from err
                with config_path.open() as f:
                    config_data = yaml.safe_load(f) or {}
            elif config_path.suffix == ".toml":
                try:
                    import tomllib
                except ImportError:
                    try:
                        import tomli as tomllib  # type: ignore[import-not-found, no-redef]
                    except ImportError as err:
                        msg = "tomli is required to load TOML configuration files on Python < 3.11"
                        raise ConfigurationError(msg) from err
                with config_path.open("rb") as f:
                    config_data = tomllib.load(f)
            else:
                msg = f"Unsupported configuration file format: {config_path.suffix}"
                raise ConfigurationError(msg)
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            msg = f"Failed to load configuration file: {e}"
            raise ConfigurationError(msg) from e

    # Handle env file if specified
    if env_file is not None:
        env_file = Path(env_file)
        if env_file.exists():
            _load_env_file(env_file)

    try:
        return Settings(**config_data)
    except Exception as e:
        msg = f"Configuration validation failed: {e}"
        raise ConfigurationError(msg) from e


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file.
    """
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = value
