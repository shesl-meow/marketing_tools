"""Utility functions for persisting and retrieving string files."""

import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


FILES_DIR = Path(__file__).resolve().parents[1] / "files"


def _ensure_files_dir() -> Path:
    """Create the files directory if it does not exist."""
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    return FILES_DIR


@tool
def read_text_file(file_path: str) -> str:
    """Agent tool: read text content from the provided path.

    Args:
        file_path: Absolute or relative path to the text file to read.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return path.read_text(encoding="utf-8")


@tool
def write_text_file(content: str, file_path: Optional[str]) -> str:
    """Agent tool: write text content to a path or generated id.

    Args:
        content: Text content to persist.
        file_path: Optional path to write to; when omitted a new file path is generated in `files/`.
    """
    if file_path:
        path = Path(file_path).expanduser()
    else:
        files_dir = _ensure_files_dir()
        file_id = uuid.uuid4().hex
        path = files_dir / f"{file_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)
