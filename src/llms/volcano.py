import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

def create_model(model_name: str = "deepseek-v3-2-251201") -> ChatOpenAI:
    if ChatOpenAI is None:
        raise ImportError("ChatOpenAI is not available; ensure langchain and openai extras are installed.")

    return ChatOpenAI(
        openai_api_key=os.environ.get("ARK_API_KEY"),
        openai_api_base="https://ark.cn-beijing.volces.com/api/v3",   
        model=model_name,
        streaming=True,
    )
