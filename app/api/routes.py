"""
FastAPI эндпоинты для поиска кандидатов.
"""

from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends

from app.models.api import (
    SearchRequest,
    SearchResponse,
    HealthResponse,
    APIInfoResponse,
    SearchWithLLMResponse,
    CandidateWithLLMAnalysis,
    LLMAnalysisResult,
    MLClassifierRequest,
    CandidateMLResult,
    MLClassifierResponse,
    WorkExperienceResponse,
)
from app.models.cv import CVOutput, WorkExperience
from app.services.cv_parser import CVParser
from app.services.search import search_candidates
from app.services.llm_analyze import LLMAnalyzer
from app.services.ml_classifier import MLClassifier


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
            "POST /search": "Поиск кандидатов (Vector Search)",
            "POST /search/with-llm-analysis": "Поиск с LLM анализом топ-5",
            "POST /search/ml-classifier": "Поиск через ML классификатор (TF-IDF + Logistic)",
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


@router.post("/search/with-llm-analysis", response_model=SearchWithLLMResponse, tags=["Search"])
async def search_with_llm_analysis(
    request: SearchRequest,
    parser: CVParser = Depends(get_cv_parser)
):
    """
    Поиск релевантных кандидатов с LLM анализом топ-5
    
    Выполняет два этапа:
    1. **Векторный поиск** - находит топ-K кандидатов через Qdrant (dense/sparse/hybrid)
    2. **LLM анализ** - для топ-5 кандидатов получает детальную оценку через GPT-4
    
    LLM анализ включает:
    - Оценку релевантности (0-1) на основе 4 критериев
    - Сильные и слабые стороны кандидата
    - Ключевые совпадения с требованиями
    - Отсутствующие требования
    - Рекомендацию и детальное обоснование
    
    **Параметры:**
    - **vacancy_text**: Полный текст вакансии
    - **search_mode**: dense/sparse/hybrid (рекомендуется hybrid)
    - **top_k**: Количество кандидатов для векторного поиска (1-50)
    
    **⚠️ Примечание**: LLM анализ занимает ~3-5 секунд на кандидата
    """
    # Определяем фактический режим поиска
    actual_mode = request.search_mode
    if actual_mode in ["sparse", "hybrid"] and not parser._sparse_fitted:
        actual_mode = "dense" if request.search_mode == "hybrid" else request.search_mode
    
    # Шаг 1: Векторный поиск кандидатов
    print(f"🔍 Поиск кандидатов через {actual_mode}...")
    candidates = search_candidates(
        parser=parser,
        query_text=request.vacancy_text,
        top_k=request.top_k,
        search_mode=request.search_mode
    )
    
    if not candidates:
        return SearchWithLLMResponse(
            query_preview=request.vacancy_text[:100] + "..." if len(request.vacancy_text) > 100 else request.vacancy_text,
            search_mode=actual_mode,
            results_count=0,
            llm_analyzed_count=0,
            candidates=[]
        )
    
    # Шаг 2: LLM анализ топ-5 кандидатов
    top_n = min(5, len(candidates))
    print(f"🤖 LLM анализ топ-{top_n} кандидатов...")
    
    analyzer = LLMAnalyzer(model="gpt-4o", temperature=0.3)
    
    candidates_with_llm = []
    
    for i, candidate in enumerate(candidates, 1):
        # Для топ-5 добавляем LLM анализ
        if i <= top_n:
            try:
                print(f"   [{i}/{top_n}] Анализ: {candidate.full_name}...")
                
                # Преобразуем CandidateResult в CVOutput для анализа
                cv_data = CVOutput(
                    full_name=candidate.full_name,
                    email=candidate.email,
                    phone=candidate.phone,
                    location=candidate.location,
                    summary=candidate.summary,
                    total_experience_months=candidate.total_experience_months,
                    work_history=[
                        # Конвертируем WorkExperienceResponse обратно в WorkExperience
                        WorkExperience(
                            role=w.role,
                            company=w.company,
                            start_date=w.start_date,
                            end_date=w.end_date,
                            description=w.description,
                            technologies=w.technologies
                        )
                        for w in candidate.work_history
                    ],
                    education=[],  # Упрощаем для анализа
                    skills=candidate.skills,
                    languages=candidate.languages
                )
                
                # LLM анализ
                llm_result = analyzer.analyze_match(cv_data, request.vacancy_text)
                
                # Создаем CandidateWithLLMAnalysis
                candidate_with_llm = CandidateWithLLMAnalysis(
                    **candidate.dict(),
                    llm_analysis=LLMAnalysisResult(
                        relevance_score=llm_result.relevance_score,
                        overall_assessment=llm_result.overall_assessment,
                        summary=llm_result.summary,
                        strengths=llm_result.strengths,
                        weaknesses=llm_result.weaknesses,
                        key_matches=llm_result.key_matches,
                        missing_requirements=llm_result.missing_requirements,
                        recommendation=llm_result.recommendation,
                        reasoning=llm_result.reasoning
                    )
                )
                
                print(f"      ✅ LLM Score: {llm_result.relevance_score:.3f}")
                
            except Exception as e:
                print(f"      ⚠️ Ошибка LLM анализа: {e}")
                # Если ошибка - добавляем без LLM анализа
                candidate_with_llm = CandidateWithLLMAnalysis(**candidate.dict())
        else:
            # Остальные кандидаты без LLM анализа
            candidate_with_llm = CandidateWithLLMAnalysis(**candidate.dict())
        
        candidates_with_llm.append(candidate_with_llm)
    
    # Подсчитываем сколько кандидатов проанализировано
    llm_analyzed = sum(1 for c in candidates_with_llm if c.llm_analysis is not None)
    
    print(f"✅ Поиск завершен: {len(candidates_with_llm)} кандидатов, {llm_analyzed} с LLM анализом")
    
    return SearchWithLLMResponse(
        query_preview=request.vacancy_text[:100] + "..." if len(request.vacancy_text) > 100 else request.vacancy_text,
        search_mode=actual_mode,
        results_count=len(candidates_with_llm),
        llm_analyzed_count=llm_analyzed,
        candidates=candidates_with_llm
    )


@router.post("/search/ml-classifier", response_model=MLClassifierResponse, tags=["Search"])
async def search_ml_classifier(
    request: MLClassifierRequest,
    parser: CVParser = Depends(get_cv_parser)
):
    """
    Поиск кандидатов через ML классификатор (TF-IDF + Logistic Regression)
    
    **Supervised learning подход:**
    1. Использует обученный ML классификатор на TF-IDF фичах
    2. Для каждого CV в базе предсказывает вероятность релевантности
    3. Ранжирует по вероятности и возвращает топ-K
    
    **Преимущества:**
    - ⚡ Быстро (~1-2 секунды для всей базы)
    - 📊 Высокая точность (обучен на размеченных данных)
    - 🎯 Интерпретируемо (можно анализировать feature importance)
    - 💰 Бесплатно (не требует API)
    
    **Параметры:**
    - **vacancy_text**: Полный текст вакансии
    - **top_k**: Количество кандидатов в результате (1-50)
    - **threshold**: Порог вероятности (0.0-1.0, default=0.5)
    
    **Примечание:** Требует предварительно обученную модель в `data/models/ml_classifier_evaluation.pkl`
    """
    # Загружаем ML классификатор
    model_path = Path("data/models/ml_classifier_evaluation.pkl")
    
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail="ML классификатор не обучен. Запустите: python -m app.scripts.evaluate_ml_classifier"
        )
    
    try:
        print(f"📂 Загрузка ML классификатора...")
        classifier = MLClassifier.load(model_path)
        print(f"   ✅ Модель загружена")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка загрузки модели: {str(e)}"
        )
    
    print(f"🔍 ML классификатор: поиск кандидатов...")
    
    # Получаем все CV из Qdrant
    try:
        scroll_result = parser.qdrant_client.scroll(
            collection_name=parser.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        all_points = scroll_result[0]
        print(f"   📊 Найдено CV в базе: {len(all_points)}")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка получения CV из Qdrant: {str(e)}"
        )
    
    if not all_points:
        return MLClassifierResponse(
            query_preview=request.vacancy_text[:100] + "..." if len(request.vacancy_text) > 100 else request.vacancy_text,
            results_count=0,
            threshold=request.threshold,
            relevant_count=0,
            candidates=[]
        )
    
    # Предсказание для каждого CV
    print(f"   🤖 ML классификация {len(all_points)} кандидатов...")
    
    candidates_with_scores = []
    
    for point in all_points:
        payload = point.payload
        
        # Создаем текст CV для классификатора
        cv_text = payload.get('full_content', '')
        
        if not cv_text:
            # Fallback: создаем текст из структурированных данных
            cv_text = f"{payload.get('summary', '')} {' '.join(payload.get('skills', []))}"
        
        try:
            # ML предсказание
            ml_probability = classifier.predict_proba(request.vacancy_text, cv_text)
            ml_prediction = 1 if ml_probability >= request.threshold else 0
            
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
            
            candidate = CandidateMLResult(
                rank=0,  # Установим позже после сортировки
                score=ml_probability,  # Используем ML вероятность как score
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
                source_file=payload.get('source_file'),
                ml_probability=ml_probability,
                ml_prediction=ml_prediction
            )
            
            candidates_with_scores.append(candidate)
            
        except Exception as e:
            print(f"   ⚠️  Ошибка предсказания для {payload.get('full_name', 'Unknown')}: {e}")
            continue
    
    # Сортируем по ML вероятности
    candidates_with_scores.sort(key=lambda x: x.ml_probability, reverse=True)
    
    # Устанавливаем ранги
    for rank, candidate in enumerate(candidates_with_scores[:request.top_k], 1):
        candidate.rank = rank
    
    # Берем топ-K
    top_candidates = candidates_with_scores[:request.top_k]
    
    # Подсчитываем сколько выше порога
    relevant_count = sum(1 for c in top_candidates if c.ml_prediction == 1)
    
    print(f"   ✅ Найдено {len(top_candidates)} кандидатов, {relevant_count} выше порога {request.threshold}")
    
    return MLClassifierResponse(
        query_preview=request.vacancy_text[:100] + "..." if len(request.vacancy_text) > 100 else request.vacancy_text,
        results_count=len(top_candidates),
        threshold=request.threshold,
        relevant_count=relevant_count,
        candidates=top_candidates
    )
