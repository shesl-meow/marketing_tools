"""Agent for formatting reports into structured Markdown outputs."""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool

from ...llms.volcano import create_model
from ...tools.file_storage import FILES_DIR, write_text_file

logger = logging.getLogger(__name__)


def system_prompt() -> str:
    return (
        "You are a report-formatter.\n"
        "Given the original raw input content and multiple analysis artifacts, craft a complete, structured Markdown report.\n"
        "Structure expectations:\n"
        "- Organize section titles based on each analysis source or role name provided.\n"
        "- When referencing visualization files, embed them using Markdown image syntax `![](<path>)`; do not describe images without linking.\n"
        "- In conclusions, cite or quote the original input content to substantiate claims—use concise evidence snippets.\n"
        "- Finish with a 'Next Step Action' section containing actionable recommendations.\n"
        "Return only the Markdown content."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)


@lc_tool
def tool(raw_input: str, sub_analysis_result: Dict[str, str]) -> str:
    """
    Format report inputs and analysis files into a structured Markdown report.

    Args:
        raw_input: Original input content being analyzed.
        sub_analysis_result: Mapping of analysis role/name to its file path.

    Returns:
        Path to the generated Markdown (.md) file.
    """
    analysis_payload: List[Dict[str, Optional[str]]] = []
    for name, path_str in sub_analysis_result.items():
        path = Path(path_str).expanduser()
        item: Dict[str, Optional[str]] = {
            "source": name,
            "path": str(path),
        }
        try:
            item["content"] = path.read_text(encoding="utf-8")
        except Exception as exc:
            item["content"] = None
            logger.warning("Failed to read analysis file %s: %s", path, exc)
        analysis_payload.append(item)

    prompt_payload = {
        "raw_input": raw_input,
        "analysis_results": analysis_payload,
    }

    rf_agent = agent()
    content = (
        "Format a structured Markdown report using the provided raw input and analysis artifacts. "
        "Follow the structure rules and return only the Markdown string.\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )
    result = rf_agent.invoke({"messages": [{"role": "user", "content": content}]})
    markdown_content = result["messages"][-1].content

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FILES_DIR / f"{uuid.uuid4().hex}.md"
    write_text_file(markdown_content, str(output_path))
    return str(output_path)
