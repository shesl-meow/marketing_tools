"""File-backed demand extractor that uses the information_extract agent."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Union

from langchain_core.tools import tool

from src.tools.file_storage import read_text_file, write_text_file
from .agent import tool as information_extract_tool

logger = logging.getLogger(__name__)


@tool
def demand_extract_tool(input_file_path: str, output_file_path: Optional[str] = None) -> str:
    """
    Extract user demands from a comment json file.

    Args:
        input_file_path: File path of the input comment json. Example structure:
            `[{ "id": 1, "user": "CyberArtist", "content": "希望能增加4K导出选项", "likes": 234, "date": "2025-06-01" }]`
        output_file_path: Optional output path; defaults to alongside the input file.

    Returns:
        Output file path containing demand extraction results.
        Output structure example: `[ "改善画质" ]`
    """
    try:
        json_content = read_text_file.invoke({"file_path": input_file_path})
        comments: List[Mapping[str, Union[int, str]]] = json.loads(json_content)
    except Exception as exc:
        logger.error("Failed to read input file %s: %s", input_file_path, exc)
        raise

    texts = []
    for comment in comments:
        raw_text = comment.get("content") or ""
        texts.append(raw_text)

    demands = information_extract_tool.invoke(
        {
            "texts": texts,
            "information_type": "user demands, requests, or product requirements",
        }
    )

    if not isinstance(demands, list) or not demands:
        logger.error("Information extractor returned invalid demand results.")
        raise ValueError("Information extractor returned invalid demand results.")

    try:
        output_content = json.dumps(demands, ensure_ascii=False, indent=2)
        return write_text_file.invoke({"content": output_content, "file_path": output_file_path})
    except Exception as exc:
        logger.error("Failed to write demand file %s: %s", output_file_path, exc)
        raise
