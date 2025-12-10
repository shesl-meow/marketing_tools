"""Composite agent that orchestrates comment processing with planning support."""

from __future__ import annotations

import logging
import sys

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from ...llms.volcano import create_model
from ..information_extract import demand_extract_tool
from ..binary_classification import sanitize_comment_tool
from ..text_classification import demand_classification_tool, sentiment_classification_tool
from ..report_formatter import tool as report_formatter_tool
from ...tools.file_storage import read_text_file, write_text_file
from ...tools.statistics import invert_index, sort_by_len, sort_by_val, count_elements
from ...tools.plot_draw import bar_chart, heap_map, pie_chart

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

def sop_preference_prompt() -> str:
    return (
        "You have a 'write_todos' tool, which can maintain a todo list for complex tasks.\n\n"
        "Generate a 'Product Iteration Proposal' by following these steps:\n"
        "1. Sanitize the input user comments using the 'sanitize_comment_tool';\n"
        "2. Extract all demands from the sanitized comments using the 'demand_extract_tool';\n"
        "3. Associate the comments with extracted demands using the 'demand_classification_tool';\n"
        "4. Analyze the sentiment of the sanitized comments using the 'sentiment_classification_tool';\n"
        "5. Process data for report using the statistic tools: 'invert_index', 'sort_by_len', 'count_elements' and 'sort_by_val';\n"
        "   At least we can conclude below metrics from the processed data: Top 3 most frequent demands, Percentage of all sentiment categories, Reason of negtive sentiment;"
        "6. Draw plot to make result more readable using the 'bar_chart', 'heap_map', 'pie_chart' tools;\n"
        "7. Provide the graph and sub_report path to the 'report_formatter' tool to generate the final 'Product Iteration Proposal' report;"
    )

def agent(model=None):
    """Create the comment processor agent with planning middleware enabled."""
    prompt_template = system_prompt()
    sop_template = sop_preference_prompt()
    model = model or create_model()
    middleware = [TodoListMiddleware(system_prompt=sop_template)]
    return create_agent(
        model,
        tools=[
            read_text_file,
            write_text_file,
            invert_index,
            sort_by_len,
            sort_by_val,
            count_elements,

            bar_chart,
            heap_map,
            pie_chart,

            sanitize_comment_tool,
            demand_extract_tool,
            demand_classification_tool,
            sentiment_classification_tool,
            report_formatter_tool,
        ],
        system_prompt=prompt_template,
        middleware=middleware,
    )

