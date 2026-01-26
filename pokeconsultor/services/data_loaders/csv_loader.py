"""CSV data loader."""

import csv
from pathlib import Path

from langchain_core.documents import Document
from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.logger import logger


class CSVLoader(DataLoader):
    """Loader for CSV files with robust validation.

    Features:
    - Validates CSV structure and detects malformed rows
    - Checks for missing column values (None values)
    - Verifies all expected columns are present
    - Provides detailed error messages with line numbers

    Common CSV errors detected:
    - Improperly quoted fields
    - Missing delimiters between columns
    - Extra or missing columns in specific rows
    """

    @staticmethod
    def supports(file_path: Path) -> bool:
        """Check if file is a CSV file."""
        return file_path.suffix.lower() == ".csv"

    def load(self, file_path: Path) -> list[Document]:
        """Load CSV data and format as Document objects.

        Each row in the CSV is converted to a formatted Document
        with field names and values in content, and row metadata.

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of Document objects (one per row).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the CSV cannot be parsed or has malformed rows.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        documents: list[Document] = []

        try:
            with open(file_path, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                if not reader.fieldnames:
                    logger.warning(f"CSV file is empty: {file_path}")
                    return documents

                expected_columns = set(reader.fieldnames)
                row_number = 1  # Header is line 0

                for row in reader:
                    row_number += 1

                    # Validate row structure (skip validation errors, just warn)
                    try:
                        self._validate_row(row, expected_columns, row_number, file_path)
                    except ValueError as e:
                        # Log warning but continue processing
                        logger.warning(
                            f"Skipping malformed row at line {row_number}: {str(e)}"
                        )
                        continue

                    doc_content = self._format_row(row)
                    if doc_content.strip():  # Only add non-empty documents
                        metadata = {
                            "source": file_path.name,
                            "row_number": row_number,
                        }
                        documents.append(
                            Document(page_content=doc_content, metadata=metadata)
                        )

            logger.info(f"Loaded {len(documents)} records from CSV: {file_path.name}")
            return documents

        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}") from e

    @staticmethod
    def _validate_row(
        row: dict[str, str],
        expected_columns: set[str],
        row_number: int,
        file_path: Path,
    ) -> None:
        """Validate that a CSV row has the expected structure.

        Args:
            row: Dictionary with CSV row data.
            expected_columns: Set of expected column names.
            row_number: Line number in the file (for error reporting).
            file_path: Path to the CSV file (for error reporting).

        Raises:
            ValueError: If the row has missing columns or unexpected None keys.
        """
        # Check for None keys (indicates malformed CSV row with extra columns)
        if None in row:
            error_msg = (
                f"Malformed CSV row at line {row_number} in {file_path.name}: "
                f"Row has unexpected data that doesn't match column headers. "
                f"This usually indicates improperly quoted fields or missing delimiters."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for None values (indicates missing required columns)
        none_columns = [col for col, val in row.items() if val is None]
        if none_columns:
            error_msg = (
                f"Malformed CSV row at line {row_number} in {file_path.name}: "
                f"Missing values for columns: {', '.join(none_columns)}. "
                f"This usually indicates improperly quoted fields or missing delimiters."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for missing columns (all expected columns should be present)
        row_columns = set(row.keys())
        missing_columns = expected_columns - row_columns

        if missing_columns:
            error_msg = (
                f"Invalid CSV row at line {row_number} in {file_path.name}: "
                f"Missing expected columns: {', '.join(missing_columns)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check for completely empty rows (all values are empty strings)
        if all(not value or value.strip() == "" for value in row.values()):
            logger.warning(
                f"Empty row detected at line {row_number} in {file_path.name} - skipping"
            )

    @staticmethod
    def _format_row(row: dict[str, str]) -> str:
        """Format a CSV row into a readable document.

        Args:
            row: Dictionary with CSV row data.

        Returns:
            Formatted document string.
        """
        parts = []

        for key, value in row.items():
            if value:
                clean_key = key.replace("_", " ").title()
                parts.append(f"{clean_key}: {value}")

        return "\n".join(parts)
