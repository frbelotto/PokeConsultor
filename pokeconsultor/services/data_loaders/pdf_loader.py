"""PDF data loader."""

from pathlib import Path

import pdfplumber

from langchain_core.documents import Document
from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.logger import logger


class PDFLoader(DataLoader):
    """Loader for PDF files."""

    @staticmethod
    def supports(file_path: Path) -> bool:
        """Check if file is a PDF file."""
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> list[Document]:
        """Load PDF data with each page as a separate document.

        Extracts text from each page in the PDF and returns them as
        individual Document objects with page metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the PDF cannot be read.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            documents: list[Document] = []

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Extract PDF metadata
                pdf_metadata = {}
                if pdf.metadata:
                    # Clean and map common metadata fields
                    for key, val in pdf.metadata.items():
                        if val and isinstance(val, (str, int)):
                            pdf_metadata[f"pdf_{key.lower()}"] = val

                logger.debug(
                    f"Processing PDF with {total_pages} pages: {file_path.name}"
                )

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()

                    if text and text.strip():
                        # Create Document with metadata
                        metadata = {
                            "source": file_path.name,
                            "page_number": page_num,
                            "total_pages": total_pages,
                            **pdf_metadata
                        }
                        documents.append(Document(page_content=text, metadata=metadata))
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
