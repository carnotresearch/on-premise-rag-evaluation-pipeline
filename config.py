import os
from dotenv import load_dotenv

load_dotenv()
import os

class Config:
    API_URL = os.getenv("API_URL", "http://localhost:11434/api/chat")
    HEADERS = {
        "Content-Type": "application/json"
    }

# =========================
# Base Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "data", "raw_docs")

# =========================
# Elasticsearch Config
# =========================

ELASTIC_URL = "http://localhost:9200"
ELASTIC_INDEX = "rag_documents"

# embedding dimension for bge-m3
EMBEDDING_DIM = 1024


# =========================
# Ollama Models
# =========================

# embedding model
EMBEDDING_MODEL = "bge-m3:latest"

# local LLM
LLM_MODEL = "mistral:latest"
MODEL_NAME="mistral:latest"

# LLM_MODEL = "qwen2.5:7b-instruct"
ev_model="llama3.1:8b"

# =========================
# Chunking
# =========================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# =========================
# Retrieval Settings
# =========================

TOP_K = 5    # initial retrieval
FINAL_K = 3     # after reranking
ENABLE_RERANK = True
# chunk_size = 800
# chunk_overlap = 120
# top_k = 5
# final_k = 3
# rerank = True


# =========================
# Ollama Server
# =========================

OLLAMA_BASE_URL = "http://localhost:11434"


# =========================
# LangSmith (Optional)
# =========================

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "hanuman_god_rag")