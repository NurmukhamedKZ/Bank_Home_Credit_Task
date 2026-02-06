"""
Pydantic модели для приложения.
"""

from app.models.cv import CVOutput, WorkExperience, Education
from app.models.api import (
    SearchRequest,
    SearchResponse,
    CandidateResult,
    WorkExperienceResponse,
    HealthResponse,
    APIInfoResponse,
    LLMAnalysisResult,
    CandidateWithLLMAnalysis,
    SearchWithLLMResponse,
    MLClassifierRequest,
    CandidateMLResult,
    MLClassifierResponse,
)

__all__ = [
    # CV модели
    "CVOutput",
    "WorkExperience",
    "Education",
    # API модели
    "SearchRequest",
    "SearchResponse",
    "CandidateResult",
    "WorkExperienceResponse",
    "HealthResponse",
    "APIInfoResponse",
    "LLMAnalysisResult",
    "CandidateWithLLMAnalysis",
    "SearchWithLLMResponse",
    "MLClassifierRequest",
    "CandidateMLResult",
    "MLClassifierResponse",
]
