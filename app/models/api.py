"""
Pydantic модели для API запросов и ответов.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class WorkExperienceResponse(BaseModel):
    """Опыт работы кандидата в ответе API"""
    role: str
    company: str
    start_date: str
    end_date: str
    description: str
    technologies: List[str]


class CandidateResult(BaseModel):
    """Результат поиска - информация о кандидате"""
    rank: int = Field(description="Позиция в рейтинге (1 = лучший)")
    score: float = Field(description="Оценка релевантности")
    full_name: str = Field(description="ФИО кандидата")
    email: Optional[str] = Field(default=None, description="Email")
    phone: Optional[str] = Field(default=None, description="Телефон")
    location: List[str] = Field(default_factory=list, description="Локация")
    summary: str = Field(default="", description="Краткое описание профиля")
    skills: List[str] = Field(default_factory=list, description="Навыки")
    total_experience_months: int = Field(default=0, description="Общий опыт в месяцах")
    work_history: List[WorkExperienceResponse] = Field(
        default_factory=list,
        description="История работы"
    )
    languages: List[str] = Field(default_factory=list, description="Языки")
    links: List[str] = Field(default_factory=list, description="Ссылки (LinkedIn, GitHub)")
    source_file: Optional[str] = Field(default=None, description="Идентификатор файла CV")


class SearchRequest(BaseModel):
    """Запрос на поиск кандидатов"""
    vacancy_text: str = Field(
        ...,
        min_length=10,
        description="Текст вакансии для поиска релевантных кандидатов"
    )
    search_mode: str = Field(
        default="dense",
        description="Режим поиска: 'dense' (Voyage AI), 'sparse' (TF-IDF), или 'hybrid' (оба)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Количество кандидатов в результате (1-50)"
    )


class SearchResponse(BaseModel):
    """Ответ на запрос поиска"""
    query_preview: str = Field(description="Превью текста запроса (первые 100 символов)")
    search_mode: str = Field(description="Использованный режим поиска")
    results_count: int = Field(description="Количество найденных кандидатов")
    candidates: List[CandidateResult] = Field(description="Список кандидатов")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str
    collection: str
    documents_count: int
    sparse_fitted: bool
    sparse_method: str = Field(default="tfidf")


class APIInfoResponse(BaseModel):
    """Информация об API"""
    name: str
    version: str
    description: str
    endpoints: dict


class LLMAnalysisResult(BaseModel):
    """Результат LLM анализа кандидата"""
    relevance_score: float = Field(description="LLM оценка релевантности (0-1)")
    overall_assessment: str = Field(description="Общая оценка: excellent/good/moderate/poor")
    summary: str = Field(description="Краткое резюме о соответствии")
    strengths: List[str] = Field(description="Сильные стороны кандидата")
    weaknesses: List[str] = Field(description="Слабые стороны кандидата")
    key_matches: List[str] = Field(description="Ключевые совпадения с требованиями")
    missing_requirements: List[str] = Field(description="Отсутствующие требования")
    recommendation: str = Field(description="Рекомендация: strongly_recommend/recommend/consider/not_recommend")
    reasoning: str = Field(description="Детальное обоснование")


class CandidateWithLLMAnalysis(CandidateResult):
    """Кандидат с LLM анализом"""
    llm_analysis: Optional[LLMAnalysisResult] = Field(
        default=None,
        description="Результат анализа через LLM (если доступен)"
    )


class SearchWithLLMResponse(BaseModel):
    """Ответ на запрос поиска с LLM анализом"""
    query_preview: str = Field(description="Превью текста запроса")
    search_mode: str = Field(description="Использованный режим поиска")
    results_count: int = Field(description="Количество найденных кандидатов")
    llm_analyzed_count: int = Field(description="Количество кандидатов проанализированных через LLM")
    candidates: List[CandidateWithLLMAnalysis] = Field(description="Список кандидатов с LLM анализом")
