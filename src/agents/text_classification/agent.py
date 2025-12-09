"""Agent for classifying texts into predefined categories."""

from langchain.agents import create_agent

from ...llms.volcano import create_model


def system_prompt() -> str:
    return (
        "You are a text-classifier.\n"
        "You will receive JSON with 'texts' (list of strings) and 'categories' (allowed labels).\n"
        "Assign one or more categories from the provided list to each text. Use an empty array if no category applies.\n"
        "Return only a JSON array of arrays, matching the order of the input texts.\n"
        "Do not include explanations or any other text."
    )


def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)
