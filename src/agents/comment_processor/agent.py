"""Composite agent that orchestrates comment processing with planning support."""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool as lc_tool

from ...llms.volcano import create_model
from ..information_extract import tool as information_extract_tool
from ..sanitize_comment import tool as sanitize_comment_tool
from ..text_classification import tool as text_classification_tool
from ...tools.file_storage import read_text_file, write_text_file
from ...tools.statistics import invert_index, sort_by_len

logger = logging.getLogger(__name__)


def system_prompt() -> str:
    """Instruction set for coordinating comment processing."""
    return (
        "You are a comment-processing orchestrator agent.\n"
        "When encountering any task involving numerical calculation, array (list) manipulation, JSON parsing/serialization, or dictionary operation, "
        "always use the provided tools (such as invert_index, sort_by_len, etc.) rather than attempting to compute or process such logic yourself in the prompt. "
        "All such operations that can rely on tools should be delegated to the available tools—avoid performing direct calculations or structure traversals in the response.\n"
        "Use read_text_file to load comment content, sanitize_comment for cleaning, information_extract for extracting facts, text_classification for labeling, "
        "write_text_file for storing outputs, and statistics tools for any sorting/indexing/array/dictionary related tasks.\n"
        "Return only the file path of the final output file."
    )


def agent(model=None):
    """Create the comment processor agent with planning middleware enabled."""
    prompt_template = system_prompt()
    model = model or create_model()
    middleware = [TodoListMiddleware()]
    return create_agent(
        model,
        tools=[
            read_text_file,
            write_text_file,
            invert_index,
            sort_by_len,

            sanitize_comment_tool,
            information_extract_tool,
            text_classification_tool,
        ],
        system_prompt=prompt_template,
        middleware=middleware,
    )

