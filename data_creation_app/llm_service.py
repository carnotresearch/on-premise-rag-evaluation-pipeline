import requests
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------
# API CONFIG
# -------------------------
API_URL = os.getenv("API_URL")

HEADERS = {
    "Content-Type": "application/json"
}


# -------------------------
# API CALL (with retry + backoff)
# -------------------------
def call_llm(payload: dict, retries: int = 4) -> dict | None:
    """
    Makes LLM API call with retry logic.
    EXACT copy from your original code (no logic changed).
    """

    if not API_URL:
        raise ValueError("API_URL is not set in .env")

    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json=payload,
                timeout=90
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")

        except Exception as e:
            logger.warning(f"Request error: {e}")

        # exponential backoff
        wait = 2 ** attempt
        time.sleep(wait)

    return None