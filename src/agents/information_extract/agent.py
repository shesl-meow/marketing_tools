"""Agent for extracting arbitrary information from free-form user texts."""

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
        "You are an information-extractor.\n"
        "You should identify specified information from a list of user texts and return the distinct items you find.\n"
        "Aggregate information as more as possible, avoid similar information and keep the most representative information.\n"
        "Return only a JSON array of unique strings.\n\n"
        "You will receive a list of texts in JSON format.\n"
        "Input example: `Extract sentiment from the below texts json:\\n[\"I love this\", \"This makes me angry\"]`\n"
        "Output example: `[\"joy\", \"anger\"]`\n"
        "Do not include duplicates or explanations—only the JSON array."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)


@lc_tool
def tool(
    texts: List[str],
    information_type: str,
) -> List[str]:
    """
    Extract specific information across a list of texts.

    Args:
        texts: List of text strings.
        information_type: Description of the information to extract (e.g., features, demands, sentiments).

    Returns:
        List of distinct information items found in the texts.
    """
    input_json = json.dumps(texts, ensure_ascii=False)
    in_agent = agent()
    content = f"Extract {information_type} from the below texts json:\n{input_json}"
    result = in_agent.invoke({"messages": [{"role": "user", "content": content}]})
    response_content = result["messages"][-1].content

    try:
        info_items = json_repair.loads(response_content)
        if isinstance(info_items, list):
            return info_items
        return []
    except Exception as exc:
        logger.error(f"Error parsing agent result: {exc}")
        return []
