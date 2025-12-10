"""Unified StdIO entrypoint for registered agents."""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

# Allow running the script directly via `python chat.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI

from src.agents.information_extract.agent import agent as info_agent
from src.agents.sanitize_comment.agent import agent as sanitize_agent
from src.agents.text_classification.agent import agent as classify_agent
from src.agents.comment_processor.agent import agent as cp_agent


AGENT_REGISTRY: Dict[str, Callable[[], ChatOpenAI]] = {
    "information_extract": info_agent,
    "sanitize_comment": sanitize_agent,
    "text_classification": classify_agent,
    "comment_processor": cp_agent,
}


def _run_agent(agent: ChatOpenAI, content: str):
    # result = agent.invoke({"messages": [{"role": "user", "content": content}]})
    # full_content = result["messages"][-1].content  # Add newline after streaming
    # full_content = ""
    # for token, metadata in agent.stream({"messages": [{"role": "user", "content": content}]}, stream_mode="messages"):
    #     if token.content:
    #         print(token.content, end="", flush=True)
    #         full_content += token.content
    # return full_content

    for chunk in agent.stream({"messages": [{"role": "user", "content": content}]}, stream_mode="updates"):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a registered agent via StdIO.")
    parser.add_argument(
        "--agent",
        required=True,
        choices=AGENT_REGISTRY.keys(),
        help="Agent key to execute.",
    )
    args = parser.parse_args()

    agent_factory = AGENT_REGISTRY[args.agent]
    chat_agent = agent_factory()

    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        _run_agent(chat_agent, payload)
        return

    for line in sys.stdin:
        _run_agent(chat_agent, line)


if __name__ == "__main__":
    main()
