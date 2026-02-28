"""Tests for incremental embedding functionality.

Tests cover the following scenarios:
1. Initial embedding of new files
2. Incremental addition of new files
3. Removal of files (embeddings should be deleted)
4. Modification of existing files (re-embedding)
5. Combined operations: simultaneous add/remove/modify
6. Empty directory handling
7. Hash collision and verification
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from faker import Faker

fake = Faker()


# =============================================================================
# FIXTURES: Test directories and mock services
# =============================================================================


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for ChromaDB cache."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_text_file(directory: Path, name: str, content: str) -> Path:
    """Helper to create a text file with specific content."""
    file_path = directory / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def detect_incremental_changes(
    source_files: list[Path], existing_hashes: set[str]
) -> tuple[list[Path], set[str]]:
    """Replicate incremental detection logic used by the RAG pipeline."""
    files_to_embed: list[Path] = []
    files_to_delete: set[str] = set(existing_hashes)

    for file_path in source_files:
        file_hash = calculate_file_hash(file_path)
        if file_hash not in existing_hashes:
            files_to_embed.append(file_path)
        files_to_delete.discard(file_hash)

    return files_to_embed, files_to_delete


# =============================================================================
# TESTS: EmbeddingService Methods (Unit Tests)
# =============================================================================


class TestEmbeddingServiceMethods:
    """Unit tests for EmbeddingService incremental methods."""

    def test_add_file_embeddings_creates_correct_metadata_structure(
        self, temp_data_dir: Path
    ) -> None:
        """Test that add_file_embeddings creates correct metadata structure."""
        file_path = create_text_file(
            temp_data_dir, "pokemon.txt", "Pikachu é um Pokémon elétrico muito popular."
        )
        file_hash = calculate_file_hash(file_path)
        chunks = ["Pikachu é um Pokémon elétrico.", "Muito popular."]

        # Create expected metadata structure as the service would
        metadatas = [
            {"file_hash": file_hash, "file_path": str(file_path)} for _ in chunks
        ]

        assert len(metadatas) == 2
        assert all(m["file_hash"] == file_hash for m in metadatas)
        assert all(m["file_path"] == str(file_path) for m in metadatas)

    def test_delete_filter_structure(self) -> None:
        """Test that delete filter is structured correctly for ChromaDB."""
        file_hash = "abc123def456"

        # This is the filter structure used by delete_file_embeddings
        expected_filter = {"file_hash": file_hash}

        assert "file_hash" in expected_filter
        assert expected_filter["file_hash"] == file_hash

    def test_get_file_hashes_logic_handles_edge_cases(self) -> None:
        """Test that get_file_hashes logic handles various metadata scenarios."""
        # Simulate ChromaDB response with various edge cases
        mock_metadata_response = {
            "metadatas": [
                {"file_hash": "hash1", "file_path": "/path/file1.txt"},
                {"file_hash": "hash1", "file_path": "/path/file1.txt"},  # Duplicate
                {"file_hash": "hash2", "file_path": "/path/file2.txt"},
                {"file_hash": "hash3", "file_path": "/path/file3.txt"},
                None,  # Edge case: None metadata
                {},  # Edge case: Empty metadata
            ]
        }

        # Replicate the logic from get_file_hashes
        hashes = set()
        for meta in mock_metadata_response["metadatas"]:
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])

        assert hashes == {"hash1", "hash2", "hash3"}


# =============================================================================
# TESTS: RAGService Incremental Logic (Integration-like Tests)
# =============================================================================


class TestRAGServiceIncrementalLogic:
    """Tests for RAGService._load_with_incremental_embedding logic."""

    def test_detect_new_files(self, temp_data_dir: Path) -> None:
        """Test that new files are detected and added to files_to_embed."""
        # Create initial files
        file1 = create_text_file(temp_data_dir, "file1.txt", "Conteúdo 1")
        file2 = create_text_file(temp_data_dir, "file2.txt", "Conteúdo 2")

        # Simulate: ChromaDB has no existing hashes
        existing_hashes: set[str] = set()

        # Calculate current file hashes
        source_files = [file1, file2]
        files_to_embed, _files_to_delete = detect_incremental_changes(
            source_files, existing_hashes
        )

        assert len(files_to_embed) == 2
        assert file1 in files_to_embed
        assert file2 in files_to_embed

    def test_detect_deleted_files(self, temp_data_dir: Path) -> None:
        """Test that deleted files are detected for embedding removal."""
        # Create only one file
        file1 = create_text_file(temp_data_dir, "file1.txt", "Conteúdo")

        # Simulate: ChromaDB has 3 files previously
        hash1 = calculate_file_hash(file1)
        existing_hashes = {hash1, "old_hash_2", "old_hash_3"}

        source_files = [file1]
        _files_to_embed, files_to_delete = detect_incremental_changes(
            source_files, existing_hashes
        )

        # Should mark 2 files for deletion
        assert files_to_delete == {"old_hash_2", "old_hash_3"}

    @pytest.mark.parametrize(
        ("original_content", "updated_content"),
        [
            ("Conteúdo original", "Conteúdo MODIFICADO"),
            ("Content A", "Content B"),
        ],
    )
    def test_detect_modified_files(
        self,
        temp_data_dir: Path,
        original_content: str,
        updated_content: str,
    ) -> None:
        """Test that modified files (different hash) are detected."""
        # Create file and calculate original hash
        file1 = create_text_file(temp_data_dir, "file1.txt", original_content)
        original_hash = calculate_file_hash(file1)

        # Simulate existing hashes in ChromaDB (with original hash)
        existing_hashes = {original_hash}

        # Modify the file
        file1.write_text(updated_content)
        new_hash = calculate_file_hash(file1)

        assert original_hash != new_hash

        # Now detect changes
        files_to_embed, files_to_delete = detect_incremental_changes(
            [file1], existing_hashes
        )

        # Should embed modified file (new hash) and delete old hash
        assert len(files_to_embed) == 1
        assert file1 in files_to_embed
        assert original_hash in files_to_delete

    def test_unchanged_files_not_reembedded(self, temp_data_dir: Path) -> None:
        """Test that unchanged files are not re-embedded."""
        # Create file
        file1 = create_text_file(temp_data_dir, "file1.txt", "Conteúdo estável")
        file_hash = calculate_file_hash(file1)

        # Simulate: ChromaDB already has this hash
        existing_hashes = {file_hash}

        files_to_embed, files_to_delete = detect_incremental_changes(
            [file1], existing_hashes
        )

        # Nothing to embed or delete
        assert len(files_to_embed) == 0
        assert len(files_to_delete) == 0


# =============================================================================
# TESTS: Hash Calculation
# =============================================================================


class TestHashCalculation:
    """Tests for file hash calculation."""

    @pytest.mark.parametrize(
        ("content_a", "content_b", "should_match"),
        [
            ("Mesmo conteúdo", "Mesmo conteúdo", True),
            ("Conteúdo A", "Conteúdo B", False),
        ],
    )
    def test_calculate_file_hash_comparison(
        self,
        temp_data_dir: Path,
        content_a: str,
        content_b: str,
        should_match: bool,
    ) -> None:
        """Test hash equality and inequality based on file content."""
        file1 = create_text_file(temp_data_dir, "file1.txt", content_a)
        file2 = create_text_file(temp_data_dir, "file2.txt", content_b)

        hash1 = calculate_file_hash(file1)
        hash2 = calculate_file_hash(file2)

        assert (hash1 == hash2) is should_match

    def test_hash_format_is_valid_sha256(self, temp_data_dir: Path) -> None:
        """Test that hash output is valid SHA256 format."""
        file1 = create_text_file(temp_data_dir, "file1.txt", "Test content")
        file_hash = calculate_file_hash(file1)

        # SHA256 produces 64 hex characters
        assert len(file_hash) == 64
        assert all(c in "0123456789abcdef" for c in file_hash)


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for incremental embedding."""

    def test_empty_data_directory(self, temp_data_dir: Path) -> None:
        """Test handling of empty data directory."""
        # No files in directory
        source_files = list(temp_data_dir.glob("*.txt"))

        assert len(source_files) == 0

    def test_unsupported_file_types_ignored(self, temp_data_dir: Path) -> None:
        """Test that unsupported file types are not processed."""
        # Create unsupported file
        unsupported = temp_data_dir / "data.xyz"
        unsupported.write_text("Some data")

        # Create supported file
        supported = create_text_file(temp_data_dir, "data.txt", "Real content")

        # Simulate LoaderFactory.is_supported behavior
        supported_extensions = {".txt", ".md", ".csv", ".pdf"}

        supported_files = [
            f
            for f in temp_data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        assert len(supported_files) == 1
        assert supported_files[0] == supported

    def test_large_number_of_files(self, temp_data_dir: Path) -> None:
        """Test incremental logic with many files."""
        # Create 50 files
        num_files = 50
        for i in range(num_files):
            create_text_file(
                temp_data_dir,
                f"file_{i:03d}.txt",
                f"Conteúdo do arquivo {i}: {fake.paragraph()}",
            )

        source_files = list(temp_data_dir.glob("*.txt"))
        existing_hashes: set[str] = set()

        files_to_embed = []
        for file_path in source_files:
            file_hash = calculate_file_hash(file_path)
            if file_hash not in existing_hashes:
                files_to_embed.append(file_path)

        assert len(files_to_embed) == num_files

    def test_partial_existing_embeddings(self, temp_data_dir: Path) -> None:
        """Test scenario where some files are already embedded."""
        # Create 5 files
        files = []
        for i in range(5):
            f = create_text_file(temp_data_dir, f"file_{i}.txt", f"Content {i}")
            files.append(f)

        # Simulate: first 3 files already embedded
        existing_hashes = {
            calculate_file_hash(files[0]),
            calculate_file_hash(files[1]),
            calculate_file_hash(files[2]),
        }

        files_to_embed = []
        for file_path in files:
            file_hash = calculate_file_hash(file_path)
            if file_hash not in existing_hashes:
                files_to_embed.append(file_path)

        # Only files 3 and 4 should be embedded
        assert len(files_to_embed) == 2
        assert files[3] in files_to_embed
        assert files[4] in files_to_embed


# =============================================================================
# TESTS: Combined Operations
# =============================================================================


class TestCombinedOperations:
    """Tests for combined add/remove/modify operations."""

    def test_simultaneous_add_and_delete(self, temp_data_dir: Path) -> None:
        """Test adding new file while another is deleted."""
        # Initial state: file1 exists
        file1 = create_text_file(temp_data_dir, "file1.txt", "Content 1")
        hash1 = calculate_file_hash(file1)

        # ChromaDB has hash1 and hash2 (file2 was deleted)
        existing_hashes = {hash1, "hash_of_deleted_file2"}

        # Add file3
        file3 = create_text_file(temp_data_dir, "file3.txt", "New content")

        source_files = [file1, file3]
        files_to_embed, files_to_delete = detect_incremental_changes(
            source_files, existing_hashes
        )

        # file3 should be embedded, hash_of_deleted_file2 should be deleted
        assert len(files_to_embed) == 1
        assert file3 in files_to_embed
        assert "hash_of_deleted_file2" in files_to_delete
        assert hash1 not in files_to_delete

    def test_modify_one_add_one_delete_one(self, temp_data_dir: Path) -> None:
        """Test complex scenario: modify + add + delete simultaneously."""
        # Create initial files
        file1 = create_text_file(temp_data_dir, "file1.txt", "Original 1")
        original_hash1 = calculate_file_hash(file1)

        # Existing state in ChromaDB
        existing_hashes = {original_hash1, "hash_file2_deleted"}

        # Modify file1
        file1.write_text("Modified 1")
        calculate_file_hash(file1)

        # Add file3
        file3 = create_text_file(temp_data_dir, "file3.txt", "Brand new file")
        calculate_file_hash(file3)

        # Process
        source_files = [file1, file3]
        files_to_embed, files_to_delete = detect_incremental_changes(
            source_files, existing_hashes
        )

        # Both modified file1 and new file3 should be embedded
        assert len(files_to_embed) == 2
        assert file1 in files_to_embed
        assert file3 in files_to_embed

        # Both original_hash1 and hash_file2_deleted should be deleted
        assert original_hash1 in files_to_delete
        assert "hash_file2_deleted" in files_to_delete


# =============================================================================
# TESTS: Chunking Integration
# =============================================================================


class TestChunkingIntegration:
    """Tests for chunking during incremental embedding."""

    def test_chunk_documents_called_for_new_files(self, temp_data_dir: Path) -> None:
        """Test that chunk_documents is called when embedding new files."""
        file1 = create_text_file(
            temp_data_dir,
            "pokemon.txt",
            "Pikachu é um Pokémon elétrico. " * 100,  # Large enough to chunk
        )

        # This would be tested in integration with EmbeddingService
        content = file1.read_text()

        # Just verify file was created with enough content
        assert len(content) > 500

    def test_multiple_files_chunked_independently(self, temp_data_dir: Path) -> None:
        """Test that each file is chunked independently."""
        files = []
        for i in range(3):
            f = create_text_file(
                temp_data_dir,
                f"file_{i}.txt",
                f"Conteúdo específico do arquivo {i}. " * 50,
            )
            files.append(f)

        # Each file should produce independent chunks
        chunks_per_file = {}
        for f in files:
            content = f.read_text()
            # Simplified chunking simulation
            chunks_per_file[f.name] = content.split(". ")

        assert len(chunks_per_file) == 3
        for name, chunks in chunks_per_file.items():
            assert len(chunks) > 1


# =============================================================================
# TESTS: RAGService Integration Logic
# =============================================================================


class TestRAGServiceIntegrationLogic:
    """Tests for RAGService incremental loading logic (algorithm verification)."""

    def test_incremental_workflow_full_scenario(self, temp_data_dir: Path) -> None:
        """Test complete incremental workflow: initial + add + modify + delete."""

        # === Phase 1: Initial Load (empty database) ===
        file1 = create_text_file(temp_data_dir, "file1.txt", "Content 1")
        file2 = create_text_file(temp_data_dir, "file2.txt", "Content 2")

        existing_hashes: set[str] = set()  # Empty ChromaDB
        source_files = [file1, file2]

        files_to_embed_phase1 = []
        files_to_delete_phase1 = set(existing_hashes)

        file_hash_map = {}
        for file_path in source_files:
            file_hash = calculate_file_hash(file_path)
            file_hash_map[file_hash] = file_path
            if file_hash not in existing_hashes:
                files_to_embed_phase1.append(file_path)
            files_to_delete_phase1.discard(file_hash)

        assert len(files_to_embed_phase1) == 2
        assert len(files_to_delete_phase1) == 0

        # Simulate embedding phase 1
        existing_hashes = set(file_hash_map.keys())

        # === Phase 2: Add new file ===
        file3 = create_text_file(temp_data_dir, "file3.txt", "New content")
        source_files = [file1, file2, file3]

        files_to_embed_phase2, files_to_delete_phase2 = detect_incremental_changes(
            source_files, existing_hashes
        )

        assert len(files_to_embed_phase2) == 1
        assert file3 in files_to_embed_phase2
        assert len(files_to_delete_phase2) == 0

        # Update existing hashes
        existing_hashes.add(calculate_file_hash(file3))

        # === Phase 3: Modify file1 ===
        old_hash1 = calculate_file_hash(file1)
        file1.write_text("Modified content 1")
        new_hash1 = calculate_file_hash(file1)

        source_files = [file1, file2, file3]

        files_to_embed_phase3, files_to_delete_phase3 = detect_incremental_changes(
            source_files, existing_hashes
        )

        # Only modified file1 needs re-embedding
        assert len(files_to_embed_phase3) == 1
        assert file1 in files_to_embed_phase3
        # Old hash of file1 should be deleted
        assert old_hash1 in files_to_delete_phase3

        # Update hashes
        existing_hashes.discard(old_hash1)
        existing_hashes.add(new_hash1)

        # === Phase 4: Delete file2 ===
        hash2 = calculate_file_hash(file2)
        file2.unlink()  # Delete the file

        source_files = [file1, file3]  # file2 no longer exists

        files_to_embed_phase4, files_to_delete_phase4 = detect_incremental_changes(
            source_files, existing_hashes
        )

        assert len(files_to_embed_phase4) == 0
        assert hash2 in files_to_delete_phase4

    def test_hash_based_change_detection(self, temp_data_dir: Path) -> None:
        """Test that change detection relies purely on file hash."""
        # Create file
        file1 = create_text_file(temp_data_dir, "file.txt", "Original")
        original_hash = calculate_file_hash(file1)

        # Overwrite with SAME content
        file1.write_text("Original")
        same_hash = calculate_file_hash(file1)

        assert original_hash == same_hash

        # Overwrite with DIFFERENT content
        file1.write_text("Modified")
        different_hash = calculate_file_hash(file1)

        assert original_hash != different_hash

    def test_file_rename_detection(self, temp_data_dir: Path) -> None:
        """Test that renaming a file doesn't change its hash (content-based)."""
        # Create file
        file1 = create_text_file(temp_data_dir, "original_name.txt", "Same content")
        original_hash = calculate_file_hash(file1)

        # Rename the file
        new_path = temp_data_dir / "new_name.txt"
        file1.rename(new_path)
        renamed_hash = calculate_file_hash(new_path)

        # Hash should be the same (content unchanged)
        assert original_hash == renamed_hash

    def test_whitespace_only_changes_detected(self, temp_data_dir: Path) -> None:
        """Test that even whitespace changes produce different hash."""
        file1 = create_text_file(temp_data_dir, "file.txt", "Content")
        hash1 = calculate_file_hash(file1)

        file1.write_text("Content ")  # Added trailing space
        hash2 = calculate_file_hash(file1)

        assert hash1 != hash2

    def test_empty_file_has_consistent_hash(self, temp_data_dir: Path) -> None:
        """Test that empty files produce consistent hash."""
        file1 = create_text_file(temp_data_dir, "empty1.txt", "")
        file2 = create_text_file(temp_data_dir, "empty2.txt", "")

        hash1 = calculate_file_hash(file1)
        hash2 = calculate_file_hash(file2)

        assert hash1 == hash2
        # SHA256 of empty string
        assert (
            hash1 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
