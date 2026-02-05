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
        default="hybrid",
        description="Режим поиска: 'dense' (Voyage AI), 'sparse' (TF-IDF), или 'hybrid' (оба)"
    )
    top_k: int = Field(
        default=10,
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
