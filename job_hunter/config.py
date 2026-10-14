"""Runtime configuration, overridable through environment variables."""

import os

MODEL_NAME = os.getenv("JOB_HUNTER_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Extraction and scoring should be deterministic; writing benefits from a little variety.
EXTRACT_TEMPERATURE = 0.0
WRITE_TEMPERATURE = 0.4
RETRY_TEMPERATURE = 0.7

# Guards against runaway generations from small local models.
MAX_OUTPUT_TOKENS = int(os.getenv("JOB_HUNTER_MAX_TOKENS", "1200"))
REQUEST_TIMEOUT_S = float(os.getenv("JOB_HUNTER_TIMEOUT", "180"))

# How many times the critic may send the cover letter back for revision.
MAX_LETTER_REVISIONS = int(os.getenv("JOB_HUNTER_MAX_REVISIONS", "2"))
# Critic score (0-10) at or above which a letter is accepted.
LETTER_PASS_SCORE = 8

# Weights for the hybrid fit score (must sum to 1).
WEIGHT_REQUIRED = 0.55
WEIGHT_PREFERRED = 0.15
WEIGHT_LLM_JUDGEMENT = 0.30
