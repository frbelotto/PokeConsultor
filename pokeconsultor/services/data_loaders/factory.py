"""Factory for automatically detecting and creating appropriate data loaders."""

from pathlib import Path

from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.data_loaders.csv_loader import CSVLoader
from pokeconsultor.services.data_loaders.pdf_loader import PDFLoader
from pokeconsultor.services.data_loaders.text_loader import TextLoader
from pokeconsultor.services.logger import logger


class LoaderFactory:
    """Factory for creating appropriate data loaders based on file type.

    This factory implements the Strategy pattern, providing a single point
    of responsibility for instantiating the correct loader for a given file.
    """

    # Loaders in order of precedence
    _loaders: list[type[DataLoader]] = [
        PDFLoader,
        CSVLoader,
        TextLoader,
    ]

    @classmethod
    def get_loader(cls, file_path: Path) -> DataLoader:
        """Get an appropriate data loader for the given file.

        Args:
            file_path: Path to the file to load.

        Returns:
            An instance of a DataLoader that can handle the file.

        Raises:
            ValueError: If no loader can handle the file format.
        """
        file_path = Path(file_path)

        for loader_class in cls._loaders:
            if loader_class.supports(file_path):
                logger.debug(f"Using {loader_class.__name__} for {file_path.name}")
                return loader_class()

        supported_formats = cls.get_supported_formats()
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported file extensions (e.g., ['.csv', '.xlsx']).
        """
        formats = set()

        # Map loader classes to their common extensions
        extension_map = {
            CSVLoader: [".csv"],
            TextLoader: [".txt", ".md", ".markdown"],
            PDFLoader: [".pdf"],
        }

        for loader_class in cls._loaders:
            if loader_class in extension_map:
                formats.update(extension_map[loader_class])

        return sorted(list(formats))

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if a file format is supported.

        Args:
            file_path: Path to check.

        Returns:
            True if the file format is supported, False otherwise.
        """
        file_path = Path(file_path)
        return any(loader_class.supports(file_path) for loader_class in cls._loaders)
