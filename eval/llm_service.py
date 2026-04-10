import requests
import logging
from time import sleep
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from eval.config import API_URL, MODEL_NAME

logger = logging.getLogger(__name__)

HEADERS = {
    "Content-Type": "application/json"
}


def call_llm(prompt, model=None):
    logger.debug("Calling LLM")

    # -------------------------
    # RETRY STRATEGY (UNCHANGED)
    # -------------------------
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)   # SAME AS YOUR CODE

    # -------------------------
    # MODEL SELECTION (IMPORTANT FIX)
    # -------------------------
    model_name = model if model else MODEL_NAME

    if not model_name:
        logger.error("MODEL_NAME is missing. Check .env file")
        return ""

    # -------------------------
    # PAYLOAD (EXACT FORMAT)
    # -------------------------
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0   # SAME AS YOUR CODE
        }
    }

    # -------------------------
    # API CALL
    # -------------------------
    try:
        response = session.post(
            API_URL,
            json=payload,
            headers=HEADERS,
            timeout=60
        )

    except requests.exceptions.ConnectionError:
        logger.error("Ollama not running. Start with: ollama serve")
        return ""

    # -------------------------
    # ERROR HANDLING (UNCHANGED)
    # -------------------------
    if response.status_code != 200:
        logger.error(f"API Error (status {response.status_code}): {response.text}")
        sleep(2)
        return ""

    # -------------------------
    # RESPONSE PARSING (EXACT)
    # -------------------------
    data = response.json()

    return data.get("message", {}).get("content", "")