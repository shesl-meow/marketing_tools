"""Agent for sanitizing free-form user comments using LangChain + OpenAI."""

import os
import re
from typing import Callable, Dict, List, Match, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ...llms.volcano import create_model

def system_prompt() -> str:
    return (
        "You are a comment-sanitizer.\n"
        "You should filter out and clean comments containing sarcasm, excessive or extreme compliments, or meaningless/spammy filler content.\n"
        "Return only the filtered comments.\n\n"
        "You will receive a list of comments in JSON format, you should return the filtered comments in JSON format.\n"
        "Input example: `[{\"id\": \"123\", \"comment\": \"This is a comment\"},{\"id\": \"456\", \"comment\": \"This is an another comment\"}]`\n"
        "Output example: `[{\"id\": \"123\", \"comment\": \"This is a comment\"}]`\n"
    )

def agent(model=None):
    prompt_template = system_prompt()
    model = model or create_model()
    return create_agent(model, tools=[], system_prompt=prompt_template)
