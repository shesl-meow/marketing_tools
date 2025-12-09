"""StdIO entrypoint for the sanitize_comment agent."""

import sys
from pathlib import Path

# Allow running the script directly via `python chat.py`
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sanitize_comment.agent import agent as sanitize_agent

def _run_agent(sc_agent, content: str) -> str:
    result = sc_agent.invoke({"messages": [{"role": "user", "content": content}]})
    return result["messages"][-1].content

def main() -> None:
    sc_agent = sanitize_agent()

    if not sys.stdin.isatty():
        payload = sys.stdin.read()
        print(_run_agent(sc_agent, payload))
        return

    for line in sys.stdin:
        print(_run_agent(sc_agent, line), flush=True)


if __name__ == "__main__":
    main()
