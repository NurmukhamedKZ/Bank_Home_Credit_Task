"""
Центральная конфигурация приложения.
Все переменные окружения загружаются из .env и доступны через этот модуль.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== QDRANT ====================
QDRANT_API = os.getenv("QDRANT_API")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "CVs")

# ==================== SEARCH ====================
DEFAULT_SPARSE_METHOD = os.getenv("DEFAULT_SPARSE_METHOD", "bm25")

# ==================== LLM / EMBEDDINGS ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_PARSE_API = os.getenv("LLAMA_PARSE_API")
VOYAGE_API = os.getenv("VOYAGE_API")

# ==================== EMAIL ====================
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")

# ==================== GOOGLE ====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")