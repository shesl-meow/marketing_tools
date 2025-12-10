"""Agent for performing binary classification on texts."""

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
        "You are a binary text classifier.\n"
        "You will receive JSON containing 'texts' (list of strings), 'criteria' (description of the positive case), "
        "'positive_label', and 'negative_label'.\n"
        "For each text, decide whether it meets the criteria. Use exactly the provided labels.\n"
        "Return only a JSON array of labels in the same order as the input texts with no extra text."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)


@lc_tool
def tool(
    texts: List[str],
    criteria: str,
    positive_label: str,
    negative_label: str,
) -> List[str]:
    """
    Classify each text into one of two labels based on the provided criteria.

    Args:
        texts: List of text contents.
        criteria: Description of what qualifies as the positive case.
        positive_label: Label to use when the text meets the criteria.
        negative_label: Label to use when the text does not meet the criteria.

    Returns:
        List of labels aligned with the input texts.
    """
    input_json = json.dumps(
        {
            "texts": texts,
            "criteria": criteria,
            "positive_label": positive_label,
            "negative_label": negative_label,
        },
        ensure_ascii=False,
    )
    bin_agent = agent()
    content = (
        "For each text, decide if it satisfies the criteria. "
        "Use the positive_label when it matches; otherwise use the negative_label. "
        "Return only a JSON array of labels in order.\n"
        f"{input_json}"
    )
    result = bin_agent.invoke({"messages": [{"role": "user", "content": content}]})
    response_content = result["messages"][-1].content

    try:
        predicted = json_repair.loads(response_content)
        if not isinstance(predicted, list):
            logger.error("Agent result is not a list: %s", type(predicted))
            return []

        return [str(label) for label in predicted]
    except Exception as exc:
        logger.error("Error parsing agent result: %s", exc)
        return []
