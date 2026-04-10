import requests
import numpy as np
import logging

logger = logging.getLogger(__name__)

# -------------------------
# EMBEDDING FUNCTION
# -------------------------
def get_embedding(text):
    logger.debug("Generating embedding")

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-m3:latest",
            "prompt": text
        }
    )

    response.raise_for_status()
    return np.array(response.json()["embedding"])


# -------------------------
# EMBEDDING CLASS
# -------------------------
class BGEEmbedding:
    def embed_query(self, text):
        logger.debug("Embedding query")
        return get_embedding(text)

    def embed_documents(self, texts):
        logger.debug("Embedding documents")
        return [get_embedding(t) for t in texts]


# -------------------------
# GLOBAL OBJECT (IMPORTANT)
# -------------------------
embeddings = BGEEmbedding()