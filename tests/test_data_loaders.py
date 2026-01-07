"""Unit tests for data loaders."""

import csv
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
from faker import Faker

from pokeconsultor.services.data_loaders.base import DataLoader
from pokeconsultor.services.data_loaders.csv_loader import CSVLoader
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.data_loaders.pdf_loader import PDFLoader
from pokeconsultor.services.data_loaders.text_loader import TextLoader

# Initialize Faker for generating random test data
fake = Faker()


# ============================================================================
# FIXTURES: Temporary test files generated with random data
# ============================================================================


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file with random data using Faker."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        # Generate random headers
        headers = ["name", "email", "city", "company"]
        writer.writerow(headers)

        # Generate 5 rows of random data
        for _ in range(5):
            writer.writerow(
                [
                    fake.name(),
                    fake.email(),
                    fake.city(),
                    fake.company(),
                ]
            )

        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_csv_with_empty_rows() -> Generator[Path, None, None]:
    """Create a CSV file with empty rows for testing edge cases."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        headers = ["name", "email", "age"]
        writer.writerow(headers)

        # Valid row
        writer.writerow([fake.name(), fake.email(), fake.random_int(min=18, max=80)])
        # Empty row
        writer.writerow(["", "", ""])
        # Another valid row
        writer.writerow([fake.name(), fake.email(), fake.random_int(min=18, max=80)])

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_malformed_csv() -> Generator[Path, None, None]:
    """Create a malformed CSV file for error testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,email,city\n")
        f.write(f"{fake.name()},{fake.email()},{fake.city()}\n")
        # Malformed row with extra unquoted comma
        f.write(f"{fake.name()},invalid,email@test.com,{fake.city()}\n")
        f.write(f"{fake.name()},{fake.email()},{fake.city()}\n")

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_empty_csv() -> Generator[Path, None, None]:
    """Create an empty CSV file (no header)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # Write nothing - completely empty
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_csv_missing_columns() -> Generator[Path, None, None]:
    """Create a CSV file with a row missing columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,email,age\n")
        f.write(f"{fake.name()},{fake.email()},{fake.random_int(min=18, max=80)}\n")
        # Row with missing column - only 2 values for 3 headers
        f.write(f"{fake.name()},{fake.email()}\n")  # age missing

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_text_file() -> Generator[Path, None, None]:
    """Create a temporary text file with random content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Generate multiple paragraphs separated by double newlines
        paragraphs = [fake.paragraph(nb_sentences=5) for _ in range(3)]
        f.write("\n\n".join(paragraphs))

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_markdown_file() -> Generator[Path, None, None]:
    """Create a temporary markdown file with random content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        # Generate markdown content
        title = fake.sentence(nb_words=4)
        f.write(f"# {title}\n\n")
        f.write(f"{fake.paragraph(nb_sentences=3)}\n\n")
        f.write(f"## {fake.sentence(nb_words=3)}\n\n")
        f.write(f"{fake.paragraph(nb_sentences=4)}\n")

        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_empty_text_file() -> Generator[Path, None, None]:
    """Create an empty text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Write nothing - empty file
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_pdf_mock(mocker: Any) -> Generator[Path, None, None]:
    """Mock PDF file and pdfplumber for testing.

    Since we're not creating real PDF files, we mock pdfplumber behavior.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = Path(f.name)

    # Mock pdfplumber.open
    mock_pdf = mocker.MagicMock()
    mock_page1 = mocker.MagicMock()
    mock_page2 = mocker.MagicMock()

    # Generate random text for pages
    mock_page1.extract_text.return_value = fake.paragraph(nb_sentences=5)
    mock_page2.extract_text.return_value = fake.paragraph(nb_sentences=5)

    mock_pdf.pages = [mock_page1, mock_page2]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mocker.patch("pdfplumber.open", return_value=mock_pdf)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_pdf_empty_mock(mocker: Any) -> Generator[Path, None, None]:
    """Mock PDF file with no text content."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = Path(f.name)

    # Mock pdfplumber.open with pages that have no text
    mock_pdf = mocker.MagicMock()
    mock_page1 = mocker.MagicMock()
    mock_page2 = mocker.MagicMock()

    # Return None or empty string to simulate pages with no text
    mock_page1.extract_text.return_value = None
    mock_page2.extract_text.return_value = ""

    mock_pdf.pages = [mock_page1, mock_page2]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mocker.patch("pdfplumber.open", return_value=mock_pdf)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_text_single_paragraph() -> Generator[Path, None, None]:
    """Create a text file with a single paragraph (no double newlines)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        # Single paragraph without double newlines
        f.write(fake.paragraph(nb_sentences=10))
        temp_path = Path(f.name)

    yield temp_path

    if temp_path.exists():
        temp_path.unlink()


# ============================================================================
# TESTS: CSVLoader
# ============================================================================


class TestCSVLoader:
    """Test suite for CSVLoader class."""

    def test_supports_csv_extension(self) -> None:
        """Test that CSVLoader recognizes CSV files."""
        loader = CSVLoader()
        assert loader.supports(Path("test.csv"))
        assert loader.supports(Path("test.CSV"))
        assert not loader.supports(Path("test.txt"))
        assert not loader.supports(Path("test.pdf"))

    def test_load_valid_csv(self, temp_csv_file: Path) -> None:
        """Test loading a valid CSV file with random data."""
        loader = CSVLoader()
        documents = loader.load(temp_csv_file)

        assert len(documents) == 5  # 5 rows of data
        assert all(isinstance(doc, str) for doc in documents)
        assert all("Name:" in doc for doc in documents)
        assert all("Email:" in doc for doc in documents)

    def test_load_csv_with_empty_rows(self, temp_csv_with_empty_rows: Path) -> None:
        """Test CSV loading skips empty rows properly."""
        loader = CSVLoader()
        documents = loader.load(temp_csv_with_empty_rows)

        # Should only load the 2 valid rows, skipping the empty one
        assert len(documents) == 2

    def test_load_malformed_csv_continues_processing(self, temp_malformed_csv: Path) -> None:
        """Test that malformed rows are skipped but processing continues."""
        loader = CSVLoader()
        documents = loader.load(temp_malformed_csv)

        # Should load 2 valid rows (skipping the malformed one)
        assert len(documents) == 2

    def test_load_nonexistent_csv_raises_error(self) -> None:
        """Test that loading non-existent file raises FileNotFoundError."""
        loader = CSVLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.csv"))

    def test_format_row_converts_to_readable_text(self) -> None:
        """Test the _format_row method."""
        loader = CSVLoader()
        row = {
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": fake.email(),
        }

        formatted = loader._format_row(row)

        assert "First Name:" in formatted
        assert "Last Name:" in formatted
        assert "Email:" in formatted

    def test_load_empty_csv_returns_empty_list(self, temp_empty_csv: Path) -> None:
        """Test that empty CSV files return empty document list."""
        loader = CSVLoader()
        documents = loader.load(temp_empty_csv)

        assert documents == []

    def test_validate_row_with_none_key(self) -> None:
        """Test validation fails when row has None as key (extra columns)."""
        loader = CSVLoader()
        row = {"name": "Test", "email": "test@test.com", None: "extra"}
        expected_columns = {"name", "email"}

        with pytest.raises(ValueError, match="Malformed CSV row"):
            loader._validate_row(row, expected_columns, 2, Path("test.csv"))

    def test_validate_row_with_none_value(self) -> None:
        """Test validation fails when row has None values."""
        loader = CSVLoader()
        row = {"name": "Test", "email": None}
        expected_columns = {"name", "email"}

        with pytest.raises(ValueError, match="Malformed CSV row"):
            loader._validate_row(row, expected_columns, 2, Path("test.csv"))

    def test_validate_row_with_missing_columns(self) -> None:
        """Test validation fails when expected columns are missing."""
        loader = CSVLoader()
        row = {"name": "Test"}
        expected_columns = {"name", "email", "age"}

        with pytest.raises(ValueError, match="Missing expected columns"):
            loader._validate_row(row, expected_columns, 2, Path("test.csv"))

    def test_format_row_with_empty_values(self) -> None:
        """Test _format_row skips empty values."""
        loader = CSVLoader()
        row = {
            "name": "John Doe",
            "email": "",
            "city": "New York",
            "phone": None,
        }

        formatted = loader._format_row(row)

        assert "Name: John Doe" in formatted
        assert "City: New York" in formatted
        assert "Email" not in formatted  # Empty value should be skipped
        assert "Phone" not in formatted  # None value should be skipped

    def test_load_csv_with_general_error(self, mocker: Any) -> None:
        """Test handling of general errors when reading CSV files."""
        loader = CSVLoader()

        # Create a temporary CSV file path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,email\n")
            f.write("John,john@test.com\n")
            temp_path = Path(f.name)

        # Mock the open function to raise a generic error
        mocker.patch("builtins.open", side_effect=PermissionError("Access denied"))

        with pytest.raises(ValueError, match="Failed to parse CSV file"):
            loader.load(temp_path)

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# TESTS: TextLoader
# ============================================================================


class TestTextLoader:
    """Test suite for TextLoader class."""

    def test_supports_text_extensions(self) -> None:
        """Test that TextLoader recognizes text file extensions."""
        loader = TextLoader()
        assert loader.supports(Path("test.txt"))
        assert loader.supports(Path("test.md"))
        assert loader.supports(Path("test.markdown"))
        assert loader.supports(Path("test.TXT"))
        assert not loader.supports(Path("test.csv"))
        assert not loader.supports(Path("test.pdf"))

    def test_load_valid_text_file(self, temp_text_file: Path) -> None:
        """Test loading a valid text file with multiple paragraphs."""
        loader = TextLoader()
        documents = loader.load(temp_text_file)

        assert len(documents) == 3  # 3 paragraphs separated by double newlines
        assert all(isinstance(doc, str) for doc in documents)
        assert all(len(doc) > 0 for doc in documents)

    def test_load_markdown_file(self, temp_markdown_file: Path) -> None:
        """Test loading a markdown file."""
        loader = TextLoader()
        documents = loader.load(temp_markdown_file)

        assert len(documents) > 0
        # Check for markdown formatting
        content = "\n\n".join(documents)
        assert "#" in content or "##" in content

    def test_load_empty_text_file(self, temp_empty_text_file: Path) -> None:
        """Test that empty text files return empty list."""
        loader = TextLoader()
        documents = loader.load(temp_empty_text_file)

        assert documents == []

    def test_load_nonexistent_text_raises_error(self) -> None:
        """Test that loading non-existent file raises FileNotFoundError."""
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.txt"))

    def test_load_text_single_paragraph(self, temp_text_single_paragraph: Path) -> None:
        """Test loading text file with single paragraph (no double newlines)."""
        loader = TextLoader()
        documents = loader.load(temp_text_single_paragraph)

        # Should return single document since there are no double newlines
        assert len(documents) == 1

    def test_load_text_with_encoding_error(self, mocker: Any) -> None:
        """Test handling of encoding errors when reading text files."""
        loader = TextLoader()

        # Create a temporary file path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        # Mock the open function to raise an error
        mocker.patch(
            "builtins.open",
            side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
        )

        with pytest.raises(ValueError, match="Failed to read text file"):
            loader.load(temp_path)

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# TESTS: PDFLoader
# ============================================================================


class TestPDFLoader:
    """Test suite for PDFLoader class."""

    def test_supports_pdf_extension(self) -> None:
        """Test that PDFLoader recognizes PDF files."""
        loader = PDFLoader()
        assert loader.supports(Path("test.pdf"))
        assert loader.supports(Path("test.PDF"))
        assert not loader.supports(Path("test.txt"))
        assert not loader.supports(Path("test.csv"))

    def test_load_valid_pdf(self, temp_pdf_mock: Path) -> None:
        """Test loading a PDF file (mocked)."""
        loader = PDFLoader()
        documents = loader.load(temp_pdf_mock)

        assert len(documents) == 1  # PDF content combined into one document
        assert "[Page 1]" in documents[0]
        assert "[Page 2]" in documents[0]

    def test_load_nonexistent_pdf_raises_error(self) -> None:
        """Test that loading non-existent PDF raises FileNotFoundError."""
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.pdf"))

    def test_load_pdf_with_empty_pages(self, temp_pdf_empty_mock: Path) -> None:
        """Test loading PDF where pages have no text content."""
        loader = PDFLoader()
        documents = loader.load(temp_pdf_empty_mock)

        # Should return empty list when no text is extracted
        assert documents == []

    def test_load_pdf_with_error(self, mocker: Any) -> None:
        """Test handling of errors when reading PDF files."""
        loader = PDFLoader()

        # Create a temporary PDF file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            temp_path = Path(f.name)

        # Mock pdfplumber.open to raise an error
        mocker.patch("pdfplumber.open", side_effect=Exception("PDF read error"))

        with pytest.raises(ValueError, match="Failed to read PDF file"):
            loader.load(temp_path)

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


# ============================================================================
# TESTS: LoaderFactory
# ============================================================================


class TestLoaderFactory:
    """Test suite for LoaderFactory class."""

    def test_get_loader_for_csv(self) -> None:
        """Test factory returns CSVLoader for CSV files."""
        loader = LoaderFactory.get_loader(Path("test.csv"))
        assert isinstance(loader, CSVLoader)

    def test_get_loader_for_text(self) -> None:
        """Test factory returns TextLoader for text files."""
        loader = LoaderFactory.get_loader(Path("test.txt"))
        assert isinstance(loader, TextLoader)

        loader = LoaderFactory.get_loader(Path("test.md"))
        assert isinstance(loader, TextLoader)

    def test_get_loader_for_pdf(self) -> None:
        """Test factory returns PDFLoader for PDF files."""
        loader = LoaderFactory.get_loader(Path("test.pdf"))
        assert isinstance(loader, PDFLoader)

    def test_get_loader_unsupported_format_raises_error(self) -> None:
        """Test that unsupported file format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            LoaderFactory.get_loader(Path("test.docx"))

    def test_get_supported_formats(self) -> None:
        """Test getting list of supported formats."""
        formats = LoaderFactory.get_supported_formats()

        assert ".csv" in formats
        assert ".txt" in formats
        assert ".md" in formats
        assert ".pdf" in formats
        assert all(fmt.startswith(".") for fmt in formats)

    def test_is_supported(self) -> None:
        """Test checking if file format is supported."""
        assert LoaderFactory.is_supported(Path("test.csv"))
        assert LoaderFactory.is_supported(Path("test.txt"))
        assert LoaderFactory.is_supported(Path("test.pdf"))
        assert not LoaderFactory.is_supported(Path("test.docx"))
        assert not LoaderFactory.is_supported(Path("test.xlsx"))


# ============================================================================
# TESTS: Base DataLoader Abstract Class
# ============================================================================


class TestDataLoaderBase:
    """Test suite for DataLoader abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that DataLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataLoader()

    def test_concrete_loader_must_implement_supports(self) -> None:
        """Test that concrete loaders must implement supports method."""

        class IncompleteLoader(DataLoader):
            def load(self, file_path: Path) -> list[str]:
                return []

        with pytest.raises(TypeError):
            IncompleteLoader()

    def test_concrete_loader_must_implement_load(self) -> None:
        """Test that concrete loaders must implement load method."""

        class IncompleteLoader(DataLoader):
            @staticmethod
            def supports(file_path: Path) -> bool:
                return True

        with pytest.raises(TypeError):
            IncompleteLoader()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestDataLoaderIntegration:
    """Integration tests for the complete data loader system."""

    def test_end_to_end_csv_loading(self, temp_csv_file: Path) -> None:
        """Test complete workflow: factory -> loader -> documents."""
        loader = LoaderFactory.get_loader(temp_csv_file)
        documents = loader.load(temp_csv_file)

        assert isinstance(loader, CSVLoader)
        assert len(documents) > 0
        assert all(isinstance(doc, str) for doc in documents)

    def test_end_to_end_text_loading(self, temp_text_file: Path) -> None:
        """Test complete workflow for text files."""
        loader = LoaderFactory.get_loader(temp_text_file)
        documents = loader.load(temp_text_file)

        assert isinstance(loader, TextLoader)
        assert len(documents) > 0

    def test_end_to_end_pdf_loading(self, temp_pdf_mock: Path) -> None:
        """Test complete workflow for PDF files."""
        loader = LoaderFactory.get_loader(temp_pdf_mock)
        documents = loader.load(temp_pdf_mock)

        assert isinstance(loader, PDFLoader)
        assert len(documents) > 0

    def test_multiple_file_types_in_sequence(
        self, temp_csv_file: Path, temp_text_file: Path, temp_pdf_mock: Path
    ) -> None:
        """Test processing multiple file types consecutively."""
        files = [temp_csv_file, temp_text_file, temp_pdf_mock]
        all_documents = []

        for file_path in files:
            loader = LoaderFactory.get_loader(file_path)
            documents = loader.load(file_path)
            all_documents.extend(documents)

        assert len(all_documents) > 0
        assert all(isinstance(doc, str) for doc in all_documents)
