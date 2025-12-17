"""Structured logging using structlog.

Provides consistent, machine-parseable logging across the library
with support for JSON output in production and colored console output
for development.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ununseptium.core.config import Settings

# Module-level logger cache
_loggers: dict[str, structlog.stdlib.BoundLogger] = {}
_configured = False


def setup_logging(settings: Settings | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        settings: Application settings. If None, uses default configuration.

    Example:
        ```python
        from ununseptium.core import Settings, setup_logging

        settings = Settings(logging={"level": "DEBUG", "format": "json"})
        setup_logging(settings)
        ```
    """
    global _configured  # noqa: PLW0603

    if settings is None:
        from ununseptium.core.config import Settings

        settings = Settings()

    log_level = getattr(logging, settings.logging.level)
    is_json = settings.logging.format == "json"

    # Configure shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    if settings.logging.include_timestamp:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if settings.logging.include_caller:
        shared_processors.append(structlog.processors.CallsiteParameterAdder())

    shared_processors.append(structlog.stdlib.PositionalArgumentsFormatter())
    shared_processors.append(structlog.processors.StackInfoRenderer())
    shared_processors.append(structlog.processors.UnicodeDecoder())

    if is_json:
        # JSON output for production
        shared_processors.append(structlog.processors.format_exc_info)
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Colored console output for development
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["httpx", "httpcore", "urllib3", "filelock"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name. Defaults to the calling module's name.

    Returns:
        A bound logger instance with the specified name.

    Example:
        ```python
        from ununseptium.core import get_logger

        logger = get_logger(__name__)
        logger.info("Processing transaction", transaction_id="TXN-123")
        ```
    """
    global _configured  # noqa: PLW0603

    if not _configured:
        setup_logging()

    if name is None:
        # Get the caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            name = frame.f_back.f_globals.get("__name__", "ununseptium")
        else:
            name = "ununseptium"

    if name not in _loggers:
        _loggers[name] = structlog.stdlib.get_logger(name)

    return _loggers[name]


class LogContext:
    """Context manager for adding temporary context to log messages.

    Example:
        ```python
        from ununseptium.core.logging import LogContext, get_logger

        logger = get_logger(__name__)

        with LogContext(request_id="REQ-123", user_id="USER-456"):
            logger.info("Processing request")  # Includes request_id and user_id
        ```
    """

    def __init__(self, **context: Any) -> None:
        """Initialize log context.

        Args:
            **context: Key-value pairs to add to log context.
        """
        self._context = context
        self._token: Any = None

    def __enter__(self) -> LogContext:
        """Enter the context manager."""
        self._token = structlog.contextvars.bind_contextvars(**self._context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context manager and restore previous context."""
        structlog.contextvars.unbind_contextvars(*self._context.keys())


def bind_context(**context: Any) -> None:
    """Bind values to the current log context.

    These values will be included in all subsequent log messages
    until explicitly unbound.

    Args:
        **context: Key-value pairs to add to log context.

    Example:
        ```python
        from ununseptium.core.logging import bind_context, get_logger

        bind_context(session_id="SESS-789")
        logger = get_logger(__name__)
        logger.info("User logged in")  # Includes session_id
        ```
    """
    structlog.contextvars.bind_contextvars(**context)


def unbind_context(*keys: str) -> None:
    """Remove values from the current log context.

    Args:
        *keys: Keys to remove from the log context.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all values from the current log context."""
    structlog.contextvars.clear_contextvars()
