"""Manifest file management for incremental embedding tracking.

This module tracks which files have been embedded and where their caches are stored,
enabling smart loading without reprocessing unchanged files.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pokeconsultor.services.logger import logger


class FileManifestEntry(BaseModel):
    """Entry for a file in the manifest."""

    relative_path: str = Field(description="Relative path from data directory")
    file_hash: str = Field(description="SHA256 hash of the file content")
    cache_key: str = Field(description="Cache key/directory name for this file's embeddings")
    timestamp: int = Field(description="Unix timestamp of last embedding")

    @staticmethod
    def calculate_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hasher = hashlib.sha256()
        hasher.update(file_path.read_bytes())
        return hasher.hexdigest()


class EmbeddingManifest(BaseModel):
    """Manifest tracking all embedded files and their cache locations."""

    version: str = Field(default="1", description="Manifest schema version")
    last_updated: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Unix timestamp of last update",
    )
    files: dict[str, FileManifestEntry] = Field(
        default_factory=dict, description="Map of relative paths to file entries"
    )


class ManifestManager:
    """Manager for manifest file operations.

    Tracks which files have been embedded and their cache locations,
    enabling detection of NEW, MODIFIED, and DELETED files.
    """

    def __init__(self, manifest_path: Path) -> None:
        """Initialize the manifest manager.

        Args:
            manifest_path: Path to the manifest JSON file
        """
        self.manifest_path = manifest_path
        self._manifest: EmbeddingManifest | None = None

    @property
    def manifest(self) -> EmbeddingManifest:
        """Get the current manifest, loading it if necessary."""
        if self._manifest is None:
            self._manifest = self.load()
        return self._manifest

    def load(self) -> EmbeddingManifest:
        """Load manifest from disk or create new if it doesn't exist."""
        if not self.manifest_path.exists():
            logger.info("Creating new manifest at %s", self.manifest_path)
            return EmbeddingManifest()

        try:
            data = json.loads(self.manifest_path.read_text())
            manifest = EmbeddingManifest(**data)
            logger.info("Loaded manifest: %d files tracked", len(manifest.files))
            return manifest
        except Exception:
            logger.exception("Failed to load manifest, creating new one")
            return EmbeddingManifest()

    def save(self) -> None:
        """Save manifest to disk."""
        if self._manifest is None:
            logger.warning("No manifest to save")
            return

        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self._manifest.last_updated = int(datetime.now().timestamp())
            self.manifest_path.write_text(
                self._manifest.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info("Saved manifest with %d files", len(self._manifest.files))
        except Exception:
            logger.exception("Failed to save manifest")
            raise

    def get_file_status(
        self, relative_path: str, actual_hash: str
    ) -> str:
        """Get status of a file compared to manifest.

        Args:
            relative_path: Relative path of the file
            actual_hash: SHA256 hash of current file content

        Returns:
            One of: "NEW", "MODIFIED", "UNCHANGED"
        """
        entry = self.manifest.files.get(relative_path)

        if entry is None:
            return "NEW"

        if entry.file_hash != actual_hash:
            return "MODIFIED"

        return "UNCHANGED"

    def get_deleted_files(self, current_files: set[str]) -> list[str]:
        """Get list of files that were in manifest but no longer exist.

        Args:
            current_files: Set of relative paths currently on disk

        Returns:
            List of relative paths that were deleted
        """
        return [path for path in self.manifest.files if path not in current_files]

    def add_entry(
        self, relative_path: str, file_hash: str, cache_key: str
    ) -> None:
        """Add or update a file entry in the manifest.

        Args:
            relative_path: Relative path of the file
            file_hash: SHA256 hash of the file
            cache_key: Cache key/directory name for embeddings
        """
        self.manifest.files[relative_path] = FileManifestEntry(
            relative_path=relative_path,
            file_hash=file_hash,
            cache_key=cache_key,
            timestamp=int(datetime.now().timestamp()),
        )

    def get_entry(self, relative_path: str) -> FileManifestEntry | None:
        """Get a manifest entry for a file.

        Args:
            relative_path: Relative path of the file

        Returns:
            FileManifestEntry if found, None otherwise
        """
        return self.manifest.files.get(relative_path)

    def remove_entry(self, relative_path: str) -> None:
        """Remove a file entry from the manifest.

        Args:
            relative_path: Relative path of the file to remove
        """
        if relative_path in self.manifest.files:
            self.manifest.files.pop(relative_path)
            logger.info("Removed manifest entry: %s", relative_path)

