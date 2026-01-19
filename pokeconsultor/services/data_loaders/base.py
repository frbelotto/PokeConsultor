"""Abstract base class for data loaders."""

from abc import ABC, abstractmethod
from pathlib import Path


from langchain_core.documents import Document


class DataLoader(ABC):
    """Abstract base class for loading data from various file formats.

    This class defines the interface for all concrete data loaders.
    Each loader is responsible for reading a specific file format
    and converting it into a list of Document objects.
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
    def load(self, file_path: Path) -> list[Document]:
        """Load data from a file and return as a list of documents.

        Each document is a LangChain Document object containing content and metadata.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
