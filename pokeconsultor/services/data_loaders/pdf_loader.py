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
        """Load PDF data as a single document.

        Extracts text from all pages in the PDF and combines them into
        a single document string. Page breaks are preserved for clarity.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List containing a single document string with all PDF content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the PDF cannot be read.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            pages_text = []

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(
                    f"Processing PDF with {total_pages} pages: {file_path.name}"
                )

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text:
                        # Preserve page information
                        pages_text.append(f"[Page {page_num}]\n{text}")
                    else:
                        logger.debug(f"No text extracted from page {page_num}")

            if not pages_text:
                logger.warning(f"No text content found in PDF: {file_path}")
                return []

            # Combine all pages into a single document
            full_document = "\n\n".join(pages_text)
            logger.info(f"Loaded PDF with {total_pages} pages: {file_path.name}")

            return [full_document]

        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            raise ValueError(f"Failed to read PDF file: {e}") from e
