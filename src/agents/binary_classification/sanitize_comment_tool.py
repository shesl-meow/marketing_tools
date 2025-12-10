"""File-backed sanitize tool that leverages the binary classification agent."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Union

from langchain_core.tools import tool

from .agent import tool as binary_classification_tool

logger = logging.getLogger(__name__)


@tool
def sanitize_comment_tool(input_file_path: str, output_file_path: Optional[str] = None) -> str:
    """
    Sanitize the input comment json file by filtering out sarcasm, excessive or extreme compliments, or meaningless/spammy filler content.

    Args:
        input_file_path: file path of the input comment json file. content structure example: `[{ "id": 1, "user": "CyberArtist", "content": "这光影效果真的绝绝子，比我手绘的快多了！", "likes": 234, "date": "2025-06-01" }]`
        output_file_path: optional output path; defaults to alongside the input file.

    Returns:
        sanitized comment json file path. content structure example: `[{ "id": 1, "user": "CyberArtist", "content": "这光影效果真的绝绝子，比我手绘的快多了！", "likes": 234, "date": "2025-06-01" }]`
    """
    input_path = Path(input_file_path)
    try:
        with input_path.open("r", encoding="utf-8") as infile:
            comments: List[Mapping[str, Union[int, str]]] = json.load(infile)
    except Exception as exc:
        logger.error("Failed to read input file %s: %s", input_file_path, exc)
        raise

    texts: List[str] = []
    for comment in comments:
        # Prefer 'content', fallback to 'comment' to stay compatible with previous schemas.
        raw_text = comment.get("content") or comment.get("comment") or ""
        texts.append(str(raw_text))

    criteria = (
        "Keep only meaningful, non-sarcastic, non-spam comments that provide genuine feedback, requests, or issues. "
        "Filter out sarcasm, over-the-top compliments, spam/ads, meaningless filler, malicious prompts, or irrelevant text."
    )
    positive_label = "keep"
    negative_label = "drop"

    labels = binary_classification_tool.invoke(
        {
            "texts": texts,
            "criteria": criteria,
            "positive_label": positive_label,
            "negative_label": negative_label,
        }
    )

    if not isinstance(labels, list) or not labels:
        logger.error("Binary classifier returned invalid labels.")
        raise ValueError("Binary classifier returned invalid labels.")

    if len(labels) != len(comments):
        logger.warning(
            "Label count (%s) does not match comment count (%s); truncating to shortest.",
            len(labels),
            len(comments),
        )

    filtered_comments = [
        comment for comment, label in zip(comments, labels) if str(label) == positive_label
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file_path:
        output_path = Path(output_file_path)
    else:
        output_path = input_path.with_name(f"sanitized_{input_path.stem}_{timestamp}.json")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(filtered_comments, outfile, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("Failed to write sanitized file %s: %s", output_path, exc)
        raise

    return str(output_path)
