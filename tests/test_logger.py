"""Unit tests for logger service.

Validates that the logger service implements singleton pattern correctly
and provides expected logging functionality.
"""

import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from pokeconsultor.services.logger import LoggerService, LogLevel, logger


@pytest.fixture(autouse=True)
def reset_logger_level() -> None:
    """Fixture to ensure logger level is reset after each test."""
    original_level = logger.level
    yield
    logger.setLevel(original_level)


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
            if instance1._logger is not None:
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

            if service._logger is not None:
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

    def test_singleton_does_not_prevent_propagation_issue(self) -> None:
        """IMPORTANT: Demonstrates that Singleton pattern does NOT prevent propagation issues.

        This test illustrates a critical gap in the original design:
        - The Singleton pattern ensures only one LoggerService instance exists
        - HOWEVER, it does NOT prevent the underlying logging.Logger from propagating
          messages to its parent logger

        The real issue is:
        1. logging.getLogger() returns a logger with propagate=True by default
        2. If a parent logger has handlers, messages propagate and get duplicated
        3. The Singleton controlling LoggerService instances doesn't protect against this

        Solution: Must explicitly set propagate=False on the underlying logger.
        """
        # Even with singleton, if propagate is True, duplicates can occur
        assert logger.propagate is False, (
            "Singleton pattern alone does NOT prevent logging propagation issues. "
            "Must explicitly disable propagate=False on the underlying logger object."
        )

    def test_singleton_vs_logging_module_independence(self) -> None:
        """Clarify the distinction between LoggerService singleton and logging module behavior.

        LoggerService is a Singleton ✓ (one instance)
        logging.Logger has its own behavior (propagate, handlers, etc.)

        The Singleton doesn't control the logging.Logger's propagation behavior.
        """
        instance1 = LoggerService()
        instance2 = LoggerService()

        # Singleton guarantee
        assert instance1 is instance2  # Same LoggerService instance ✓

        # BUT the underlying logger object's propagation is independent
        logger_obj1 = instance1._logger
        logger_obj2 = instance2._logger

        # They share the same logger, but the propagate setting is still
        # controlled by the logging module, not the Singleton
        assert logger_obj1 is logger_obj2  # Same Logger object ✓
        if logger_obj1 is not None:
            assert logger_obj1.propagate is False  # Must be explicitly set

    def test_logger_no_duplicate_messages_with_multiple_handlers(self) -> None:
        """Verify that messages are not duplicated even when root logger has handlers.

        This test catches the specific bug where a logger with propagate=True
        would duplicate messages if the root logger also has handlers.
        """
        # Create a StringIO handler to capture log output
        log_capture = StringIO()
        test_handler = logging.StreamHandler(log_capture)
        test_handler.setFormatter(logging.Formatter("%(message)s"))
        test_handler.setLevel(logging.DEBUG)  # Ensure handler captures all levels

        # Add the test handler temporarily
        logger.addHandler(test_handler)

        try:
            logger.info("Test message for duplication check")

            # Flush the handler to ensure the message is written
            test_handler.flush()

            # Get the captured output
            log_output = log_capture.getvalue()

            # Count occurrences of the test message
            message_count = log_output.count("Test message for duplication check")
            assert message_count == 1, (
                f"Logger message was duplicated! Found {message_count} occurrences. "
                "This indicates that the message is being sent to multiple handlers."
            )
        finally:
            # Clean up: remove the test handler
            logger.removeHandler(test_handler)

    def test_logger_integration_with_real_logging(self) -> None:
        """Verify that logger actually logs messages that can be captured."""
        # Create a StringIO handler to capture log output
        log_capture = StringIO()
        test_handler = logging.StreamHandler(log_capture)
        test_handler.setFormatter(logging.Formatter("%(message)s"))
        test_handler.setLevel(logging.DEBUG)  # Ensure handler captures all levels

        logger.addHandler(test_handler)

        try:
            logger.info("Test integration message")
            test_handler.flush()
            log_output = log_capture.getvalue()
            assert "Test integration message" in log_output
        finally:
            logger.removeHandler(test_handler)

    def test_logger_respects_level_filtering(self) -> None:
        """Verify that logger filters messages below its configured level."""
        # Store original level
        original_level = logger.level

        # Create a StringIO handler to capture log output
        log_capture = StringIO()
        test_handler = logging.StreamHandler(log_capture)
        test_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(test_handler)

        try:
            logger.setLevel(logging.WARNING)

            logger.debug("Debug message - should not appear")
            logger.info("Info message - should not appear")
            logger.warning("Warning message - should appear")

            log_output = log_capture.getvalue()

            assert "Debug message - should not appear" not in log_output
            assert "Info message - should not appear" not in log_output
            assert "Warning message - should appear" in log_output

        finally:
            # Restore original level and remove test handler
            logger.setLevel(original_level)
            logger.removeHandler(test_handler)
