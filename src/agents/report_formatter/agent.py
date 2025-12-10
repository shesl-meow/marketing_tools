"""Agent for formatting reports into structured Markdown outputs."""

import json
from typing import Dict

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool

from ...llms.volcano import create_model
from ...tools.file_storage import read_text_file, write_text_file


def system_prompt() -> str:
    return (
        "You are a report-formatter.\n"
        "Given the original raw input content and multiple analysis artifacts, craft a complete, structured Markdown report.\n"
        "Structure expectations:\n"
        "- Organize section titles based on each analysis source or role name provided.\n"
        "- When referencing visualization files, embed them using Markdown image syntax `![](<path>)`; do not describe images without linking.\n"
        "- In conclusions, cite or quote the original input content to substantiate claims—use concise evidence snippets.\n"
        "- Finish with a 'Next Step Action' section containing actionable recommendations.\n"
        "Use the available tools to persist the final Markdown file and return only the saved file path."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(
        model,
        tools=[read_text_file, write_text_file],
        system_prompt=prompt_template,
    )


@lc_tool
def tool(
    report_preference: str,
    raw_input: str,
    analysis_topic2report_path: Dict[str, str],
) -> str:
    """
    Format report inputs and analysis files into a structured Markdown report.

    Args:
        report_preference: report should be produced (e.g., style, audience, required sections).
        raw_input: Original input content being analyzed.
        analysis_topic2report_path: Mapping of analysis topic to its file path.

    Returns:
        Path to the generated Markdown (.md) file.
    """
    prompt_payload = {
        "raw_input": raw_input,
        "analysis_topic2report_path": analysis_topic2report_path,
    }

    rf_agent = agent()
    content = (
        "Format a structured Markdown report using the provided raw input and analysis artifacts. "
        "Use the available tools to read analysis files from the given paths. "
        "Follow the structure rules, save the Markdown to the suggested path using the available tool, and return only the saved file path.\n"
        f"Follow the report preference: {report_preference}. \n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    result = rf_agent.invoke({"messages": [{"role": "user", "content": content}]})
    output_path = result["messages"][-1].content
    return output_path
