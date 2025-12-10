"""Agent for sanitizing free-form user comments using LangChain + OpenAI."""

import json
import logging
from typing import Dict, List, Union

import json_repair
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool

from ...llms.volcano import create_model

logger = logging.getLogger(__name__)


def system_prompt() -> str:
    return (
        "You are a comment-sanitizer.\n"
        "You should filter out and clean comments containing sarcasm, excessive or extreme compliments, or meaningless/spammy filler content.\n"
        "Return only the filtered comments.\n\n"
        "You will receive a list of comments in JSON format, you should return the filtered comments in JSON format.\n"
        "Input example: `[{\"id\": \"123\", \"comment\": \"This is a comment\"},{\"id\": \"456\", \"comment\": \"This is an another comment\"}]`\n"
        "Output example: `[{\"id\": \"123\", \"comment\": \"This is a comment\"}]`\n"
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)


@lc_tool
def tool(
    comments: List[Dict[str, Union[int, str]]],
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
