"""
Pydantic модели для структурированного представления CV.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    """Структура для опыта работы"""
    role: str = Field(description="Job title, e.g. 'Senior Python Developer'")
    company: str = Field(description="Company name")
    start_date: str = Field(description="Start date usually in YYYY-MM format")
    end_date: str = Field(description="End date in YYYY-MM format or 'Present'")
    description: str = Field(description="Short summary of responsibilities and achievements")
    technologies: List[str] = Field(description="Specific tools used in this role")


class Education(BaseModel):
    """Структура для образования"""
    institution: str
    degree: str = Field(description="Degree, e.g. 'Bachelor in Computer Science'")
    year: str = Field(description="Year of graduation")


class CVOutput(BaseModel):
    """Основная модель структурированного CV"""
    full_name: str = Field(description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    links: List[str] = Field(default_factory=list, description="URLs to LinkedIn, GitHub, Portfolio")
    location: List[str] = Field(default_factory=list, description="Location of the candidate")
    
    summary: str = Field(default="", description="A brief professional summary of the candidate")
    
    total_experience_months: int = Field(default=0, description="Total work experience in months")
    
    work_history: List[WorkExperience] = Field(default_factory=list, description="List of work experiences")
    education: List[Education] = Field(default_factory=list)
    
    skills: List[str] = Field(default_factory=list, description="List of technical/hard skills")
    languages: List[str] = Field(default_factory=list, description="Languages spoken and proficiency level")
