"""File-backed sentiment classifier built on the text classification agent."""

import json
import logging
from typing import List, Mapping, Optional, Union

from langchain_core.tools import tool

from ...tools.file_storage import read_text_file, write_text_file
from .agent import tool as text_classification_tool

logger = logging.getLogger(__name__)


@tool
def sentiment_classification_tool(
    input_file_path: str,
    output_file_path: Optional[str] = None,
) -> str:
    """
    Classify each comment's sentiment as positive, negative, or neutral.

    Args:
        input_file_path: File path of the input comment json. Example:
            `[{"id": 1, "content": "很好用"}, {"id": 2, "content": "卡顿严重"}]`
        output_file_path: Optional output path; defaults to alongside the input file.

    Returns:
        Output file path containing sentiment results. Example structure:
            `[{"id": 1, "content": "...", "sentiment": "positive"}]`
    """
    sentiment_categories = ["positive", "negative", "neutral"]

    try:
        json_content = read_text_file.invoke({"file_path": input_file_path})
        comments: List[Mapping[str, Union[int, str]]] = json.loads(json_content)
    except Exception as exc:
        logger.error("Failed to read input file %s: %s", input_file_path, exc)
        raise

    texts: List[str] = []
    for comment in comments:
        raw_text = comment.get("content") or comment.get("comment") or ""
        texts.append(str(raw_text))

    predictions = text_classification_tool.invoke(
        {"texts": texts, "categories": sentiment_categories}
    )

    if not isinstance(predictions, list) or not predictions:
        logger.error("Text classifier returned invalid sentiment labels.")
        raise ValueError("Text classifier returned invalid sentiment labels.")

    if len(predictions) != len(comments):
        logger.warning(
            "Sentiment count (%s) does not match comment count (%s); truncating to shortest.",
            len(predictions),
            len(comments),
        )

    results = []
    for idx, (comment, labels) in enumerate(zip(comments, predictions)):
        normalized = []
        if isinstance(labels, list):
            normalized = [label for label in labels if label]
        elif labels is not None:
            normalized = [labels]

        sentiment = str(normalized[0]) if normalized else "neutral"
        results.append(
            {
                "id": comment.get("id", idx),
                "content": comment.get("content") or comment.get("comment") or "",
                "sentiment": sentiment,
            }
        )

    try:
        output_content = json.dumps(results, ensure_ascii=False, indent=2)
        return write_text_file.invoke({"content": output_content, "file_path": output_file_path})
    except Exception as exc:
        logger.error("Failed to write sentiment classification file %s: %s", output_file_path, exc)
        raise
