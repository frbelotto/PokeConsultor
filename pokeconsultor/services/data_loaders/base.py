"""Abstract base class for data loaders."""

from abc import ABC, abstractmethod
from pathlib import Path


class DataLoader(ABC):
    """Abstract base class for loading data from various file formats.

    This class defines the interface for all concrete data loaders.
    Each loader is responsible for reading a specific file format
    and converting it into a list of document strings.
    """

    @staticmethod
    @abstractmethod
    def supports(file_path: Path) -> bool:
        """Check if this loader supports the given file.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if this loader can handle the file format, False otherwise.
        """

    @abstractmethod
    def load(self, file_path: Path) -> list[str]:
        """Load data from a file and return as a list of documents.

        Each document is a formatted string representation of the data.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of formatted document strings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
