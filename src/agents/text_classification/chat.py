"""StdIO entrypoint for the text_classification agent."""

import sys
from pathlib import Path

# Allow running the script directly via `python chat.py`
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.text_classification.agent import agent as classify_agent


def _run_agent(in_agent, content: str) -> str:
    result = in_agent.invoke({"messages": [{"role": "user", "content": content}]})
    return result["messages"][-1].content


def main() -> None:
    in_agent = classify_agent()

    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        print(_run_agent(in_agent, payload))
        return

    for line in sys.stdin:
        print(_run_agent(in_agent, line), flush=True)


if __name__ == "__main__":
    main()

