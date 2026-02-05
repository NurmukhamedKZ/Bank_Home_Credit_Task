"""
FastAPI эндпоинты для поиска кандидатов.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from app.models.api import (
    SearchRequest,
    SearchResponse,
    HealthResponse,
    APIInfoResponse,
)
from app.services.cv_parser import CVParser
from app.services.search import search_candidates


router = APIRouter()

# Глобальная переменная для CVParser (инициализируется в lifespan)
_cv_parser: Optional[CVParser] = None


def get_cv_parser() -> CVParser:
    """Dependency для получения CVParser"""
    if _cv_parser is None:
        raise HTTPException(status_code=503, detail="CVParser не инициализирован")
    return _cv_parser


def set_cv_parser(parser: CVParser):
    """Установка глобального CVParser"""
    global _cv_parser
    _cv_parser = parser


def clear_cv_parser():
    """Очистка глобального CVParser"""
    global _cv_parser
    _cv_parser = None


@router.get("/", response_model=APIInfoResponse, tags=["Info"])
async def root():
    """Информация об API"""
    return APIInfoResponse(
        name="CV Search API",
        version="1.0.0",
        description="API для поиска релевантных кандидатов по тексту вакансии",
        endpoints={
            "POST /search": "Поиск кандидатов по тексту вакансии",
            "GET /health": "Проверка работоспособности сервиса",
            "GET /": "Информация об API"
        }
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(parser: CVParser = Depends(get_cv_parser)):
    """Проверка работоспособности сервиса"""
    try:
        collection_info = parser.qdrant_client.get_collection(parser.collection_name)
        
        return HealthResponse(
            status="healthy",
            collection=parser.collection_name,
            documents_count=collection_info.points_count,
            sparse_fitted=parser._sparse_fitted,
            sparse_method=parser.sparse_method
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ошибка подключения к Qdrant: {str(e)}")


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest, parser: CVParser = Depends(get_cv_parser)):
    """
    Поиск релевантных кандидатов по тексту вакансии
    
    - **vacancy_text**: Полный текст вакансии (требования, обязанности, навыки)
    - **search_mode**: Режим поиска
        - `dense` - семантический поиск через Voyage AI embeddings
        - `sparse` - keyword поиск через TF-IDF
        - `hybrid` - комбинация обоих методов (рекомендуется)
    - **top_k**: Количество кандидатов в результате (1-50)
    
    Возвращает список кандидатов, отсортированных по релевантности.
    """
    # Определяем фактический режим поиска
    actual_mode = request.search_mode
    if actual_mode in ["sparse", "hybrid"] and not parser._sparse_fitted:
        actual_mode = "dense" if request.search_mode == "hybrid" else request.search_mode
    
    candidates = search_candidates(
        parser=parser,
        query_text=request.vacancy_text,
        top_k=request.top_k,
        search_mode=request.search_mode
    )
    
    return SearchResponse(
        query_preview=request.vacancy_text[:100] + "..." if len(request.vacancy_text) > 100 else request.vacancy_text,
        search_mode=actual_mode,
        results_count=len(candidates),
        candidates=candidates
    )
