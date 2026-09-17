"""Official public UTAS sources used by the prototype."""

import os

BASE_URL = "https://www.utas.edu.au"
PROJECTS_URL = f"{BASE_URL}/research/degrees/available-projects"
RESEARCH_DEGREES_URL = f"{BASE_URL}/research/degrees"
ENTRY_REQUIREMENTS_URL = f"{BASE_URL}/research/degrees/what-is-a-research-degree"
SCHOLARSHIPS_URL = f"{BASE_URL}/research/degrees/scholarships-and-fees"
FAQ_URL = f"{BASE_URL}/research/degrees/frequently-asked-questions"

# Optional local query planner. It never downloads a model automatically.
OLLAMA_HOST = os.environ.get("UTAS_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("UTAS_OLLAMA_MODEL", "qwen3:1.7b")
ANSWER_MODEL = os.environ.get("UTAS_ANSWER_MODEL", OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("UTAS_OLLAMA_TIMEOUT", "20"))
ANSWER_TIMEOUT_SECONDS = float(os.environ.get("UTAS_ANSWER_TIMEOUT", "90"))
