from langchain_core.tools import tool as lc_tool

from .agent import agent


@lc_tool
def analyze_user_sentiment(file_path: str) -> str:
    """
    Analyze user sentiment distribution and negative sentiment drivers from comments in a file.

    Args:
        file_path: Path to the input text file containing comments to process.

    Returns:
        Path to the output file produced by the agent.
    """
    cp_agent = agent()
    user_message = (
        "Perform user sentiment analysis on the comments in this file.\n"
        "Report the proportions of positive, negative, and neutral sentiments, "
        "and summarize the primary reasons causing negative sentiment.\n"
        f"Input file path: {file_path}"
    )
    result = cp_agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )
    return str(result["messages"][-1].content).strip()
