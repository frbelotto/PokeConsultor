"""Logging service using Singleton pattern.

Provides centralized logging configuration as a single service for the entire application.
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import validate_call

from pokeconsultor.config import settings


class LogLevel(Enum):
    """Supported log level names accepted from configuration."""

    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET

    @staticmethod
    @validate_call(validate_return=True)
    def get_log_level(level_str: str) -> int:
        """Convert string to logging level value."""
        try:
            return int(LogLevel[level_str.upper()].value)
        except KeyError:
            raise ValueError(f"Invalid log level: {level_str}")


class LoggerService:
    """Singleton service for centralized logging management.
    Ensures a single logging configuration exists throughout the application lifecycle.
    """

    _instance: LoggerService | None = None
    _logger: logging.Logger | None = None  # explicit None initial state

    def __new__(cls) -> LoggerService:
        """Ensure a single instance is created for the lifetime of the process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger with basic configuration."""
        if LoggerService._logger is None:
            logger_instance = logging.getLogger(__name__)
            log_level = LogLevel.get_log_level(settings.LOG_LEVEL)
            logger_instance.setLevel(log_level)

            # Disable propagation to prevent duplicate messages from parent logger
            logger_instance.propagate = False

            # Clear any existing handlers to prevent duplicates
            logger_instance.handlers.clear()

            # Add our custom handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)

            LoggerService._logger = logger_instance

    @property
    def logger(self) -> logging.Logger:
        """Return the configured logger instance.

        This property guarantees a non-None logger after initialization.
        """
        assert self._logger is not None, "Logger not initialized"
        return self._logger


logger = LoggerService().logger
