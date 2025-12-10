"""Agent for classifying texts into predefined categories."""

import json
import logging
from typing import List

import json_repair
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool

from ...llms.volcano import create_model

logger = logging.getLogger(__name__)


def system_prompt() -> str:
    return (
        "You are a text-classifier.\n"
        "You will receive JSON with 'texts' (list of strings) and 'categories' (allowed labels).\n"
        "Assign one or more categories from the provided list to each text. Use an empty array if no category applies.\n"
        "Return only a JSON array of arrays, matching the order of the input texts.\n"
        "Do not include explanations or any other text."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)


@lc_tool
def tool(
    texts: List[str],
    categories: List[str],
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
    cls_agent = agent()
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
