from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import json
from app.services.resume_parser import ResumeParser
from app.nlp.extractor import NlpExtractor
from app.scoring.engine import ScoringEngine
from app.explanation.generator import ExplanationGenerator
from app.schemas.resume import ResumeSchema, EvaluationResult
from app.schemas.job import JobDescription

router = APIRouter()
nlp_extractor = NlpExtractor()
scoring_engine = ScoringEngine()

@router.post("/analyze", response_model=EvaluationResult)
async def analyze_resume(
    resume_file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # 1. Read File
        content = await resume_file.read()
        filename = resume_file.filename
        
        # 2. Parse Text
        text = ResumeParser.extract_text(content, filename)
        
        # 3. Extract Info
        resume_data = nlp_extractor.extract_info(text)
        
        # 4. Parse Job Description
        try:
            job_dict = json.loads(job_description)
            job_desc = JobDescription(**job_dict)
            
            # If frontend sent no skills (or we want to override), extract them from JD text
            if not job_desc.mandatory_skills:
                # We reuse the NLP extractor for the JD text
                # We can treat the JD as a "resume" to extract skills
                jd_info = nlp_extractor.extract_info(job_desc.description)
                job_desc.mandatory_skills = jd_info.skills
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for job_description")
            
        # 5. Score (Returns Dict with breakdown)
        score_result = scoring_engine.calculate_score(resume_data, job_desc)
        total_score = score_result["total_score"]
        missing_skills = score_result["missing_skills"]
        
        # 6. Generate Explanation (Pass full score result including missing_skills)
        feedback = ExplanationGenerator.generate_feedback(score_result, resume_data, job_desc)
        
        # 7. Return Result
        project_analysis = []
        if resume_data.projects:
            for proj in resume_data.projects:
                project_analysis.append({
                    "name": proj.name,
                    "complexity": getattr(proj, 'complexity', 'Basic'),
                    "technologies": proj.technologies # Currently empty or needs population if we want to show it
                })

        return EvaluationResult(
            score=total_score,
            skill_score=score_result.get("skill_score", 0),
            experience_score=score_result.get("experience_score", 0),
            project_score=score_result.get("project_score", 0),
            structure_score=score_result.get("structure_score", 0),
            missing_skills=missing_skills, # Use the semantic-aware missing skills list
            feedback=feedback,
            project_analysis=project_analysis
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
