import json
from typing import List, Dict, Union, Optional
import json_repair
from langchain_openai import ChatOpenAI
from .agent import agent
import logging

logger = logging.getLogger(__name__)


def tool(
    comments: List[Dict[str, Union[int, str]]],
    model: Optional[ChatOpenAI] = None,
) -> List[Dict[str, Union[int, str]]]:
    """
    Sanitize comments by filtering out sarcasm, excessive compliments, and spammy content.

    Args:
        comments: List of comment dictionaries with 'id' and 'content' keys

    Returns:
        Filtered list of comment dictionaries
    """
    input_json = json.dumps(comments, ensure_ascii=False)
    sc_agent = agent()
    result = sc_agent.invoke({"messages": [{"role": "user", "content": input_json}]})
    response_content = result["messages"][-1].content

    try:
        filtered_comments = json_repair.loads(response_content)
        return filtered_comments
    except Exception as e:
        logger.error(f"Error parsing agent result: {e}")
        return []
