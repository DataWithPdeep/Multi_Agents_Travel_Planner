import os
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_secret(name: str, default=None):
    """Pehle Streamlit secrets dekho, na mile to .env / environment variable use karo."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


# Saari keys ek baar os.environ me daal do, taaki poora code (mcpClient, graph, ChatGroq)
# bina badlaav ke os.getenv se padh sake.
for _k in ["GROQ_API_KEY", "TAVILY_API_KEY", "AVIATIONSTACK_API_KEY",
           "OPENWEATHER_API_KEY", "DATABASE_URL"]:
    _v = get_secret(_k)
    if _v:
        os.environ[_k] = str(_v).strip()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def get_llm():
    return ChatGroq(
        model=get_secret("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
    )

from urllib.parse import urlparse
print("DB host:", urlparse(DATABASE_URL).hostname if DATABASE_URL else None)