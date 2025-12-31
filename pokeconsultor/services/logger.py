"""Logging service using Singleton pattern.

Provides centralized logging configuration as a single service for the entire application.
"""

from __future__ import annotations

import logging
from enum import Enum

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
    def get_log_level(level_str: str) -> int:
        """Convert string to logging level value."""
        try:
            return LogLevel[level_str.upper()].value
        except KeyError:
            raise ValueError(f"Invalid log level: {level_str}")


class LoggerService:
    """Singleton service for centralized logging management.

    Ensures a single logging configuration exists throughout the application lifecycle.
    """

    _instance: LoggerService | None = None
    _logger: logging.Logger | None = None

    def __new__(cls) -> "LoggerService":
        """Ensure a single instance is created for the lifetime of the process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger with basic configuration."""
        if LoggerService._logger is None:
            LoggerService._logger = logging.getLogger(__name__)
            log_level = settings.LOG_LEVEL
            LoggerService._logger.setLevel(LogLevel.get_log_level(log_level))

            # Avoid adding handlers multiple times
            if not LoggerService._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                LoggerService._logger.addHandler(handler)


logger = LoggerService()._logger
