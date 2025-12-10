from langchain_core.tools import tool as lc_tool
from .agent import agent

@lc_tool
def extract_top3_demands(file_path: str) -> str:
    """
    Run the comment-processor agent on a file and return the output file path.

    Args:
        file_path: Path to the input text file containing comments to process.

    Returns:
        Path to the output file produced by the agent.
    """
    cp_agent = agent()
    user_message = (
        "Extract the top 3 most frequent demands from the comments in the file: \n"
        f"{file_path}"
    )
    result = cp_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    return str(result["messages"][-1].content).strip()
