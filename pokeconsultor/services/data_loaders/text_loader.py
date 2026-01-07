"""Text and Markdown data loader."""

from pathlib import Path

from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.logger import logger


class TextLoader(DataLoader):
    """Loader for text and markdown files."""

    @staticmethod
    def supports(file_path: Path) -> bool:
        """Check if file is a text or markdown file."""
        return file_path.suffix.lower() in {".txt", ".md", ".markdown"}

    def load(self, file_path: Path) -> list[str]:
        """Load text data as documents.

        Reads the entire file content and splits on double newlines to
        create separate documents. If no double newlines exist, the
        entire file becomes a single document.

        Args:
            file_path: Path to the text file.

        Returns:
            List of formatted document strings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"Text file is empty: {file_path}")
                return []

            # Split on double newlines to create separate documents
            # If no double newlines, treat entire file as one document
            documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]

            logger.info(
                f"Loaded {len(documents)} documents from text: {file_path.name}"
            )
            return documents

        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise ValueError(f"Failed to read text file: {e}") from e
