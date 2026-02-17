from pydantic import BaseModel
from typing import List, Optional, Dict

class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = []
    complexity: Optional[str] = "Basic" # Basic, Medium, Advanced
    duration: Optional[str] = None

class Experience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None

class ResumeSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience: List[Experience] = []
    projects: List[Project] = []
    education: List[Education] = []
    text_content: str  # Full raw text for scoring

class EvaluationResult(BaseModel):
    score: float
    skill_score: float = 0.0
    experience_score: float = 0.0
    project_score: float = 0.0
    structure_score: float = 0.0
    missing_skills: List[str]
    feedback: List[str]
    project_analysis: Optional[List[Dict]] = None # Detailed project feedback
