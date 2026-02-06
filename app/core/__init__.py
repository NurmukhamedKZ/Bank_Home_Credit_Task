"""
Core модуль - конфигурация и общие утилиты.
"""

from app.core.config import (
    QDRANT_API,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    DEFAULT_SPARSE_METHOD,
    OPENAI_API_KEY,
    LLAMA_PARSE_API,
    VOYAGE_API,
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    IMAP_SERVER,
)

__all__ = [
    "QDRANT_API",
    "QDRANT_URL",
    "QDRANT_COLLECTION_NAME",
    "DEFAULT_SPARSE_METHOD",
    "OPENAI_API_KEY",
    "LLAMA_PARSE_API",
    "VOYAGE_API",
    "EMAIL_ADDRESS",
    "EMAIL_PASSWORD",
    "IMAP_SERVER",
]
