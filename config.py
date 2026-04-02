import os
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
env_candidates = [
    current_dir / ".env",
    current_dir.parent / ".env",
    current_dir.parent.parent / ".env",
    current_dir.parent.parent.parent / ".env",
]

for env_path in env_candidates:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
