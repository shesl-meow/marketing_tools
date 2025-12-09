import json
import logging
from typing import List, Optional

import json_repair
from langchain_openai import ChatOpenAI

from .agent import agent

logger = logging.getLogger(__name__)


def tool(
    texts: List[str],
    categories: List[str],
    model: Optional[ChatOpenAI] = None,
) -> List[List[str]]:
    """
    Classify each text into one of the provided categories.

    Args:
        texts: List of text contents.
        categories: Allowed category labels to choose from (one-to-many).

    Returns:
        List of category lists aligned with the input texts.
    """
    input_json = json.dumps(
        {"texts": texts, "categories": categories},
        ensure_ascii=False,
    )
    cls_agent = agent(model=model)
    content = (
        "Classify each text into zero or more of the provided categories. "
        "Return only a JSON array of arrays, each containing the categories for the corresponding text.\n"
        f"{input_json}"
    )
    result = cls_agent.invoke({"messages": [{"role": "user", "content": content}]})
    response_content = result["messages"][-1].content

    try:
        predicted = json_repair.loads(response_content)
        if not isinstance(predicted, list):
            logger.error("Agent result is not a list: %s", type(predicted))
            return []

        normalized: List[List[str]] = []
        for item in predicted:
            if isinstance(item, list):
                normalized.append([str(label) for label in item])
            else:
                # If the agent returns a single label, wrap it to preserve alignment.
                normalized.append([str(item)])
        return normalized
    except Exception as exc:
        logger.error("Error parsing agent result: %s", exc)
        return []
