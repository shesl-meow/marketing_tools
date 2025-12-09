"""Utility functions for persisting and retrieving string files."""

import uuid
from pathlib import Path


FILES_DIR = Path(__file__).resolve().parents[1] / "files"


def _ensure_files_dir() -> Path:
    """Create the files directory if it does not exist."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    return FILES_DIR


def save_file(content: str) -> str:
    """
    Persist the provided string content to disk and return a file identifier.
    """
    files_dir = _ensure_files_dir()
    file_id = uuid.uuid4().hex
    file_path = files_dir / f"{file_id}.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_id


def load_file(file_id: str) -> str:
    """
    Load a previously saved file string by its file identifier.
    """
    file_path = _ensure_files_dir() / f"{file_id}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"file with id '{file_id}' was not found.")
    return file_path.read_text(encoding="utf-8")
