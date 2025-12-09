"""Agent for extracting arbitrary information from free-form user texts."""

from langchain.agents import create_agent
from ...llms.volcano import create_model


def system_prompt() -> str:
    return (
        "You are an information-extractor.\n"
        "You should identify specified information from a list of user texts and return the distinct items you find.\n"
        "Aggregate information as more as possible, avoid similar information and keep the most representative information.\n"
        "Return only a JSON array of unique strings.\n\n"
        "You will receive a list of texts in JSON format, each with 'id' and 'comment' (text content) keys.\n"
        "Input example: `Extra emotions from the below texts json:\\n[{\"id\": \"123\", \"comment\": \"I love this\"},{\"id\": \"456\", \"comment\": \"This makes me angry\"}]`\n"
        "Output example: `[\"joy\", \"anger\"]`\n"
        "Do not include duplicates or explanations—only the JSON array."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)
