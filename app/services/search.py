"""
Сервис для поиска кандидатов в Qdrant.
"""

from typing import List, Optional

from fastapi import HTTPException
from qdrant_client import models
from qdrant_client.models import Prefetch

from app.models.api import CandidateResult, WorkExperienceResponse
from app.services.cv_parser import CVParser


def search_candidates(
    parser: CVParser,
    query_text: str,
    top_k: int = 10,
    search_mode: str = "hybrid"
) -> List[CandidateResult]:
    """
    Поиск кандидатов через Qdrant с поддержкой разных режимов
    
    Args:
        parser: Инициализированный CVParser
        query_text: Текст запроса (вакансия)
        top_k: Количество результатов
        search_mode: Режим поиска - "dense", "sparse", или "hybrid"
        
    Returns:
        Список кандидатов с оценками релевантности
    """
    # Валидация режима
    if search_mode not in ["dense", "sparse", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Неверный search_mode: {search_mode}. Используйте: 'dense', 'sparse', или 'hybrid'"
        )
    
    # Проверка доступности TF-IDF для sparse и hybrid
    if search_mode in ["sparse", "hybrid"] and not parser._sparse_fitted:
        if search_mode == "sparse":
            raise HTTPException(
                status_code=400,
                detail="TF-IDF не обучен. Sparse поиск недоступен. Используйте 'dense' режим."
            )
        # Для hybrid - fallback на dense
        search_mode = "dense"
    
    # ========== DENSE-ONLY SEARCH ==========
    if search_mode == "dense":
        dense_query = parser.dense_model.embed_documents([query_text])[0]
        
        results = parser.qdrant_client.query_points(
            collection_name=parser.collection_name,
            query=dense_query,
            using="default",
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    
    # ========== SPARSE-ONLY SEARCH (TF-IDF) ==========
    elif search_mode == "sparse":
        sparse_indices, sparse_values = parser.create_sparse_query(query_text)
        
        sparse_query_vector = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        results = parser.qdrant_client.query_points(
            collection_name=parser.collection_name,
            query=sparse_query_vector,
            using="sparse",
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    
    # ========== HYBRID SEARCH (Dense + Sparse) ==========
    elif search_mode == "hybrid":
        dense_query = parser.dense_model.embed_documents([query_text])[0]
        sparse_indices, sparse_values = parser.create_sparse_query(query_text)
        
        sparse_query_vector = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        # Hybrid search через prefetch и RRF fusion
        results = parser.qdrant_client.query_points(
            collection_name=parser.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using="default",
                    limit=top_k * 2
                ),
                Prefetch(
                    query=sparse_query_vector,
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    
    # Преобразуем результаты в CandidateResult
    candidates = []
    for rank, point in enumerate(results.points, 1):
        payload = point.payload
        
        # Преобразуем work_history
        work_history = []
        for work in payload.get('work_history', []):
            work_history.append(WorkExperienceResponse(
                role=work.get('role', ''),
                company=work.get('company', ''),
                start_date=work.get('start_date', ''),
                end_date=work.get('end_date', ''),
                description=work.get('description', ''),
                technologies=work.get('technologies', [])
            ))
        
        candidate = CandidateResult(
            rank=rank,
            score=round(point.score, 4),
            full_name=payload.get('full_name', 'Unknown'),
            email=payload.get('email'),
            phone=payload.get('phone'),
            location=payload.get('location', []),
            summary=payload.get('summary', ''),
            skills=payload.get('skills', []),
            total_experience_months=payload.get('total_experience_months', 0),
            work_history=work_history,
            languages=payload.get('languages', []),
            links=payload.get('links', []),
            source_file=payload.get('source_file')
        )
        candidates.append(candidate)
    
    return candidates
