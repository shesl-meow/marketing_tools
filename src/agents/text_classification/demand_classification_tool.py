"""File-backed demand classifier built on the text classification agent."""

import json
import logging
from typing import List, Mapping, Optional, Union

from langchain_core.tools import tool

from ...tools.file_storage import read_text_file, write_text_file
from .agent import tool as text_classification_tool

logger = logging.getLogger(__name__)


@tool
def demand_classification_tool(
    input_file_path: str,
    categories_file_path: str,
    output_file_path: Optional[str] = None,
) -> str:
    """
    Classify each comment in the input file into zero or more demand categories.

    Args:
        input_file_path: File path of the input comment json. Example:
            `[{"id": 1, "content": "希望能增加4K导出", ...}, {"id": 2, "content": "配音能否更自然", ...}]`
        categories_file_path: File path containing demand category list (JSON array of strings). Example:
            `["4K导出", "配音自然"]`
        output_file_path: Optional output path; defaults to alongside the input file.

    Returns:
        Output file path containing classification results. Example structure:
            `[{"id": 1, "content": "希望能增加4K导出", "demands": ["4K导出"]}]`
    """
    try:
        json_content = read_text_file.invoke({"file_path": input_file_path})
        comments: List[Mapping[str, Union[int, str]]] = json.loads(json_content)
    except Exception as exc:
        logger.error("Failed to read input file %s: %s", input_file_path, exc)
        raise

    try:
        categories_content = read_text_file.invoke({"file_path": categories_file_path})
        raw_categories = json.loads(categories_content)
        if not isinstance(raw_categories, list):
            raise ValueError("categories is not a list")
        categories = [str(item) for item in raw_categories if str(item)]
    except Exception as exc:
        logger.error("Failed to read categories file %s: %s", categories_file_path, exc)
        raise

    texts: List[str] = []
    for comment in comments:
        # Prefer 'content', fallback to 'comment' for compatibility with older schemas.
        raw_text = comment.get("content") or ""
        texts.append(str(raw_text))

    predictions = text_classification_tool.invoke({"texts": texts, "categories": categories})

    if not isinstance(predictions, list) or not predictions:
        logger.error("Text classifier returned invalid labels.")
        raise ValueError("Text classifier returned invalid labels.")

    if len(predictions) != len(comments):
        logger.warning(
            "Prediction count (%s) does not match comment count (%s); truncating to shortest.",
            len(predictions),
            len(comments),
        )

    results = []
    for idx, (comment, labels) in enumerate(zip(comments, predictions)):
        normalized_labels = labels if isinstance(labels, list) else [labels] if labels is not None else []
        results.append(
            {
                "id": comment.get("id", idx),
                "content": comment.get("content") or comment.get("comment") or "",
                "demands": [str(label) for label in normalized_labels if str(label)],
            }
        )

    try:
        output_content = json.dumps(results, ensure_ascii=False, indent=2)
        return write_text_file.invoke({"content": output_content, "file_path": output_file_path})
    except Exception as exc:
        logger.error("Failed to write demand classification file %s: %s", output_file_path, exc)
        raise
