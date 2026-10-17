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


TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = get_secret("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY")
DATABASE_URL = get_secret("DATABASE_URL")


def get_llm():
    return ChatGroq(
        model=get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    )