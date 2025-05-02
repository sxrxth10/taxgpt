import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY1 = os.getenv("OPENAI_API_KEY1")
settings = Settings()