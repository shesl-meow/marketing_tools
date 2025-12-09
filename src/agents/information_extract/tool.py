import json
import logging
from typing import Dict, List, Optional, Union

import json_repair
from langchain_openai import ChatOpenAI

from .agent import agent

logger = logging.getLogger(__name__)


def tool(
    texts: List[Dict[str, Union[int, str]]],
    information_type: str,
    model: Optional[ChatOpenAI] = None,
) -> List[str]:
    """
    Extract specific information across a list of texts.

    Args:
        texts: List of text dictionaries with 'id' and 'comment' keys.
        information_type: Description of the information to extract (e.g., ingredients, emotions).

    Returns:
        List of distinct information items found in the texts.
    """
    input_json = json.dumps(texts, ensure_ascii=False)
    in_agent = agent(model=model)
    content = f"Extra {information_type} from the below texts json:\n{input_json}"
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
