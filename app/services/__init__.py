"""
Сервисы приложения.
"""

from app.services.cv_parser import CVParser
from app.services.email_fetcher import EmailFetcher
from app.services.search import search_candidates

__all__ = [
    "CVParser",
    "EmailFetcher",
    "search_candidates",
]
