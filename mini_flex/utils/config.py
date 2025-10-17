import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve config values
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA_DIR = os.getenv("DATA_DIR", "data")
MEMORY_PATH = os.getenv("MEMORY_PATH", "chat_memory.json")
