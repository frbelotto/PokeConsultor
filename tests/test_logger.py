"""Unit tests for logger service.

Validates that the logger service implements singleton pattern correctly
and provides expected logging functionality.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from pokeconsultor.services.logger import LoggerService, LogLevel, logger


class TestLogLevel:
    """Test cases for LogLevel enumeration."""

    def test_get_log_level_valid_levels(self) -> None:
        """Verify that valid log level strings are converted correctly."""
        assert LogLevel.get_log_level("DEBUG") == logging.DEBUG
        assert LogLevel.get_log_level("INFO") == logging.INFO
        assert LogLevel.get_log_level("WARNING") == logging.WARNING
        assert LogLevel.get_log_level("ERROR") == logging.ERROR
        assert LogLevel.get_log_level("CRITICAL") == logging.CRITICAL

    def test_get_log_level_case_insensitive(self) -> None:
        """Verify that log level strings are case-insensitive."""
        assert LogLevel.get_log_level("debug") == logging.DEBUG
        assert LogLevel.get_log_level("DeBuG") == logging.DEBUG
        assert LogLevel.get_log_level("info") == logging.INFO

    def test_get_log_level_invalid_level(self) -> None:
        """Verify that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level: INVALID"):
            LogLevel.get_log_level("INVALID")


class TestLoggerServiceSingleton:
    """Test cases for LoggerService singleton pattern implementation."""

    def test_singleton_same_instance(self) -> None:
        """Verify that multiple instantiations return the same instance."""
        instance1 = LoggerService()
        instance2 = LoggerService()
        instance3 = LoggerService()

        assert instance1 is instance2
        assert instance2 is instance3
        assert instance1 is instance3

    def test_singleton_same_logger(self) -> None:
        """Verify that all instances share the same logger."""
        instance1 = LoggerService()
        instance2 = LoggerService()

        assert instance1._logger is instance2._logger

    @patch("pokeconsultor.services.logger.settings")
    def test_logger_initialized_once(self, mock_settings: MagicMock) -> None:
        """Verify that logger is initialized only once even with multiple instances."""
        # Reset the singleton state for this test
        original_instance = LoggerService._instance
        original_logger = LoggerService._logger

        try:
            LoggerService._instance = None
            LoggerService._logger = None
            mock_settings.LOG_LEVEL = "INFO"

            # Create multiple instances
            instance1 = LoggerService()
            instance2 = LoggerService()

            # Logger should only have one handler (added once)
            assert len(instance1._logger.handlers) == 1
            assert instance1._logger is instance2._logger

        finally:
            # Restore original state
            LoggerService._instance = original_instance
            LoggerService._logger = original_logger


class TestLoggerServiceFunctionality:
    """Test cases for LoggerService logging functionality."""

    @patch("pokeconsultor.services.logger.settings")
    def test_logger_respects_log_level(self, mock_settings: MagicMock) -> None:
        """Verify that logger respects configured log level."""
        original_instance = LoggerService._instance
        original_logger = LoggerService._logger

        try:
            LoggerService._instance = None
            LoggerService._logger = None
            mock_settings.LOG_LEVEL = "WARNING"

            service = LoggerService()

            assert service._logger.level == logging.WARNING

        finally:
            LoggerService._instance = original_instance
            LoggerService._logger = original_logger

    def test_exported_logger_is_logger_instance(self) -> None:
        """Verify that exported logger is a Logger instance."""
        assert isinstance(logger, logging.Logger)

    def test_logger_has_stream_handler(self) -> None:
        """Verify that logger has at least one StreamHandler."""
        handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(handlers) > 0

    @patch("pokeconsultor.services.logger.LoggerService._logger")
    def test_logger_can_log_messages(self, mock_logger: MagicMock) -> None:
        """Verify that logger can log messages at different levels."""
        # Use the mock logger
        mock_logger.debug("Debug message")
        mock_logger.info("Info message")
        mock_logger.warning("Warning message")
        mock_logger.error("Error message")
        mock_logger.critical("Critical message")

        # Verify all log methods were called
        mock_logger.debug.assert_called_once_with("Debug message")
        mock_logger.info.assert_called_once_with("Info message")
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_logger.error.assert_called_once_with("Error message")
        mock_logger.critical.assert_called_once_with("Critical message")


class TestLoggerServiceIntegration:
    """Integration tests for LoggerService."""

    def test_logger_integration_with_real_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify that logger actually logs messages that can be captured."""
        with caplog.at_level(logging.INFO, logger="pokeconsultor.services.logger"):
            logger.info("Test integration message")

        assert "Test integration message" in caplog.text

    def test_logger_respects_level_filtering(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify that logger filters messages below its configured level."""
        # Store original level
        original_level = logger.level

        try:
            logger.setLevel(logging.WARNING)

            with caplog.at_level(logging.DEBUG):
                logger.debug("Debug message - should not appear")
                logger.info("Info message - should not appear")
                logger.warning("Warning message - should appear")

            assert "Debug message - should not appear" not in caplog.text
            assert "Info message - should not appear" not in caplog.text
            assert "Warning message - should appear" in caplog.text

        finally:
            # Restore original level
            logger.setLevel(original_level)
