"""
LLM Analyzer - анализ соответствия кандидата вакансии с объяснениями.

Использует LLM для:
- Оценки релевантности кандидата (0-1)
- Генерации текстового объяснения
- Выделения сильных и слабых сторон
"""

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.models.cv import CVOutput
from app.core.config import OPENAI_API_KEY


# ==================== PYDANTIC МОДЕЛИ ====================

class MatchAnalysis(BaseModel):
    """Результат анализа соответствия кандидата вакансии"""
    
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Оценка релевантности от 0.0 (не подходит) до 1.0 (идеально подходит)"
    )
    
    overall_assessment: str = Field(
        description="Общая оценка: 'excellent', 'good', 'moderate', 'poor'"
    )
    
    summary: str = Field(
        description="Краткое резюме (2-3 предложения) о соответствии кандидата вакансии"
    )
    
    strengths: List[str] = Field(
        description="Сильные стороны кандидата относительно вакансии (3-5 пунктов)"
    )
    
    weaknesses: List[str] = Field(
        description="Слабые стороны или пробелы относительно требований (2-4 пункта)"
    )
    
    key_matches: List[str] = Field(
        description="Ключевые совпадения навыков и опыта с требованиями (3-5 пунктов)"
    )
    
    missing_requirements: List[str] = Field(
        description="Отсутствующие или недостаточные требования (2-4 пункта)"
    )
    
    recommendation: str = Field(
        description="Рекомендация: 'strongly_recommend', 'recommend', 'consider', 'not_recommend'"
    )
    
    reasoning: str = Field(
        description="Детальное обоснование оценки и рекомендации (4-6 предложений)"
    )


# ==================== LLM ANALYZER ====================

class LLMAnalyzer:
    """
    Анализатор соответствия кандидата вакансии через LLM.
    Предоставляет детальную оценку с объяснениями.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Args:
            model: Модель OpenAI для анализа
            temperature: Температура генерации (0.0-1.0)
        """
        self.model_name = model
        self.temperature = temperature
        
        # Инициализация LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENAI_API_KEY,
            temperature=temperature
        )
        
        self.structured_llm = self.llm.with_structured_output(MatchAnalysis)
        
        # System prompt для анализа
        self.system_prompt = """You are an expert technical recruiter and talent analyst with deep knowledge of IT roles.

Your task is to analyze how well a candidate matches a job vacancy, providing:
1. A relevance score (0.0 to 1.0)
2. Detailed explanation of strengths and weaknesses
3. Key matching points and missing requirements
4. A clear recommendation

EVALUATION CRITERIA:
- Technical skills match (40%): Required vs actual skills, experience with specific technologies
- Experience level (25%): Years of experience, seniority, role complexity
- Domain fit (20%): Industry experience, project types, team size
- Soft skills & culture (15%): Communication, leadership, teamwork indicators

SCORING GUIDELINES:
- 0.9-1.0: Exceptional match, exceeds requirements
- 0.75-0.89: Strong match, meets most requirements with some extras
- 0.6-0.74: Good match, meets core requirements
- 0.4-0.59: Moderate match, missing some key requirements
- 0.2-0.39: Weak match, significant gaps
- 0.0-0.19: Poor match, fundamentally misaligned

Be objective, specific, and constructive. Focus on facts from the CV and job requirements."""
        
        # Создаем prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Analyze the match between this candidate and job vacancy.

JOB VACANCY:
{vacancy_text}

CANDIDATE CV:
Full Name: {full_name}
Total Experience: {experience_months} months ({experience_years} years)

Summary: {summary}

Skills: {skills}

Work History:
{work_history}

Education:
{education}

Languages: {languages}

Provide a comprehensive analysis with relevance score, strengths, weaknesses, and recommendation.""")
        ])
        
        # Создаем цепочку
        self.chain = self.prompt | self.structured_llm
    
    def analyze_match(
        self,
        cv_data: CVOutput,
        vacancy_text: str
    ) -> MatchAnalysis:
        """
        Анализирует соответствие кандидата вакансии
        
        Args:
            cv_data: Структурированные данные CV
            vacancy_text: Текст вакансии
            
        Returns:
            MatchAnalysis с оценкой и объяснениями
        """
        # Форматируем данные CV для промпта
        experience_years = cv_data.total_experience_months / 12
        
        skills_str = ", ".join(cv_data.skills) if cv_data.skills else "Not specified"
        
        work_history_str = self._format_work_history(cv_data.work_history)
        education_str = self._format_education(cv_data.education)
        languages_str = ", ".join(cv_data.languages) if cv_data.languages else "Not specified"
        
        # Вызываем LLM
        analysis = self.chain.invoke({
            "vacancy_text": vacancy_text,
            "full_name": cv_data.full_name,
            "experience_months": cv_data.total_experience_months,
            "experience_years": f"{experience_years:.1f}",
            "summary": cv_data.summary or "Not provided",
            "skills": skills_str,
            "work_history": work_history_str,
            "education": education_str,
            "languages": languages_str
        })
        
        return analysis
    
    def analyze_multiple(
        self,
        candidates: List[CVOutput],
        vacancy_text: str,
        show_progress: bool = True
    ) -> List[tuple[CVOutput, MatchAnalysis]]:
        """
        Анализирует несколько кандидатов для одной вакансии
        
        Args:
            candidates: Список CV кандидатов
            vacancy_text: Текст вакансии
            show_progress: Показывать прогресс
            
        Returns:
            Список пар (CV, MatchAnalysis)
        """
        results = []
        
        for i, cv in enumerate(candidates, 1):
            if show_progress:
                print(f"[{i}/{len(candidates)}] Анализ: {cv.full_name}...")
            
            try:
                analysis = self.analyze_match(cv, vacancy_text)
                results.append((cv, analysis))
                
                if show_progress:
                    print(f"   ✅ Score: {analysis.relevance_score:.3f} - {analysis.recommendation}")
            except Exception as e:
                if show_progress:
                    print(f"   ❌ Ошибка: {e}")
                continue
        
        return results
    
    def _format_work_history(self, work_history: List) -> str:
        """Форматирует историю работы для промпта"""
        if not work_history:
            return "No work history provided"
        
        formatted = []
        for i, work in enumerate(work_history[:5], 1):  # Топ-5 позиций
            formatted.append(
                f"{i}. {work.role} at {work.company} ({work.start_date} - {work.end_date})\n"
                f"   Technologies: {', '.join(work.technologies[:10]) if work.technologies else 'N/A'}\n"
                f"   {work.description[:200]}..."
            )
        
        return "\n".join(formatted)
    
    def _format_education(self, education: List) -> str:
        """Форматирует образование для промпта"""
        if not education:
            return "No education provided"
        
        formatted = [
            f"- {edu.degree} from {edu.institution} ({edu.year})"
            for edu in education
        ]
        
        return "\n".join(formatted)
    
    def get_score_interpretation(self, score: float) -> dict:
        """
        Интерпретация числового score
        
        Args:
            score: Оценка от 0.0 до 1.0
            
        Returns:
            Словарь с интерпретацией
        """
        if score >= 0.9:
            return {
                "level": "exceptional",
                "label": "🌟 Исключительное совпадение",
                "description": "Кандидат превосходит требования",
                "color": "green"
            }
        elif score >= 0.75:
            return {
                "level": "strong",
                "label": "✅ Сильное совпадение",
                "description": "Кандидат соответствует большинству требований",
                "color": "lightgreen"
            }
        elif score >= 0.6:
            return {
                "level": "good",
                "label": "👍 Хорошее совпадение",
                "description": "Кандидат соответствует основным требованиям",
                "color": "blue"
            }
        elif score >= 0.4:
            return {
                "level": "moderate",
                "label": "⚠️ Умеренное совпадение",
                "description": "Есть пробелы в ключевых требованиях",
                "color": "orange"
            }
        elif score >= 0.2:
            return {
                "level": "weak",
                "label": "❌ Слабое совпадение",
                "description": "Значительные расхождения с требованиями",
                "color": "red"
            }
        else:
            return {
                "level": "poor",
                "label": "🚫 Плохое совпадение",
                "description": "Кандидат не подходит для этой вакансии",
                "color": "darkred"
            }
