"""PDF data loader."""

from pathlib import Path

import pdfplumber

from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.logger import logger


class PDFLoader(DataLoader):
    """Loader for PDF files."""

    @staticmethod
    def supports(file_path: Path) -> bool:
        """Check if file is a PDF file."""
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> list[str]:
        """Load PDF data with each page as a separate document.

        Extracts text from each page in the PDF and returns them as
        individual documents for better chunking control.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of document strings, one per page with content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the PDF cannot be read.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            documents: list[str] = []

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(
                    f"Processing PDF with {total_pages} pages: {file_path.name}"
                )

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text and text.strip():
                        # Each page is a separate document with page reference
                        documents.append(f"[Page {page_num}] {text}")
                    else:
                        logger.debug(f"No text extracted from page {page_num}")

            if not documents:
                logger.warning(f"No text content found in PDF: {file_path}")
                return []

            logger.info(
                f"Loaded PDF with {total_pages} pages → {len(documents)} documents: {file_path.name}"
            )

            return documents

        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            raise ValueError(f"Failed to read PDF file: {e}") from e
