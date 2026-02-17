from sentence_transformers import SentenceTransformer, util
from app.schemas.resume import ResumeSchema
from app.schemas.job import JobDescription
from app.core.config import settings

class ScoringEngine:
    def __init__(self):
        try:
            self.model = SentenceTransformer(settings.TRANSFORMER_MODEL)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None


    
    # Wrapper to fix method name call in main pipeline if it was calculate_score
    def calculate_score(self, resume: ResumeSchema, job_desc: JobDescription) -> dict:
        if not self.model:
            return {"total_score": 50.0, "missing_skills": []}

        # 1. Skill Match (30%)
        skill_score, missing_skills = self._calculate_skill_score(resume, job_desc)

        # 2. Experience Match (30%)
        experience_score = self._calculate_experience_score(resume, job_desc)

        # 3. Project Match (30%)
        project_score = self._calculate_project_score(resume, job_desc)

        # 4. Structure Match (10%)
        structure_score = self._calculate_structure_score(resume)

        total_score = (skill_score * 0.3) + (experience_score * 0.3) + (project_score * 0.3) + (structure_score * 0.1)
        
        # 5. Fairness Boost for Freshers with Advanced Projects
        if (not resume.experience or len(resume.experience) == 0) and project_score > 80:
             total_score = min(total_score * 1.15, 100.0) # 15% boost
        
        return {
            "total_score": round(total_score, 2),
            "skill_score": skill_score,
            "experience_score": experience_score,
            "project_score": project_score,
            "structure_score": structure_score,
            "missing_skills": missing_skills
        }

    def _calculate_skill_score(self, resume: ResumeSchema, job_desc: JobDescription) -> (float, list):
        # 1. Exact & Semantic Matching
        if not job_desc.mandatory_skills:
            return 100.0, []
        
        matched_count = 0
        missing_skills = []
        resume_text_lower = resume.text_content.lower()
        
        # Pre-compute resume embeddings for efficiency (using chunks or full text?)
        # For simplicity and speed in this version, we'll embed the resume's extracted skills + a robust chunk of text
        # If resume.skills is empty, we fall back to text.
        
        resume_skills_text = " ".join(resume.skills) if resume.skills else ""
        # Combine skills and some context from experience to form a "Representation" of the candidate
        candidate_text = (resume_skills_text + " " + resume_text_lower[:2000]) 
        
        # We can also key off the extracted skills directly for precision
        resume_skill_embeddings = self.model.encode(resume.skills, convert_to_tensor=True) if resume.skills else None
        
        for skill in job_desc.mandatory_skills:
            skill_lower = skill.lower()
            is_matched = False
            
            # A. Exact Match (Fastest, Highest Confidence)
            # Check in extracted skills
            if any(skill_lower == s.lower() for s in resume.skills):
                matched_count += 1
                is_matched = True
                continue
                
            # Check in full text (Catch-all for simple keyword presence)
            if skill_lower in resume_text_lower:
                matched_count += 0.8 # Slightly less than 1.0 because it might be out of context
                is_matched = True
                continue
                
            # B. Semantic Match (XAI - "Meaning-Based")
            # If no exact match, check if we have a synonym or related concept
            # e.g., Job: "Machine Learning" vs Resume: "Deep Learning", "AI"
            if resume_skill_embeddings is not None and len(resume.skills) > 0:
                skill_emb = self.model.encode(skill, convert_to_tensor=True)
                # Compute cosine usage against all resume skills
                # cosmetic fix: util.cos_sim returns a tensor matrix
                cosine_scores = util.cos_sim(skill_emb, resume_skill_embeddings)[0]
                max_score = float(cosine_scores.max())
                
                if max_score > 0.75: # Threshold for "Strong Semantic Match"
                    matched_count += 0.9 # High credit for semantic match
                    is_matched = True
                elif max_score > 0.6: 
                    matched_count += 0.5 # Partial credit
                    # Note: We consider this "Partial Match" so we don't list it as strictly MISSING, 
                    # but we might flag it as "Weak Match" in a future update. For now, let's say it's not missing.
                    is_matched = True
            
            if not is_matched:
                missing_skills.append(skill)
                    
        return min((matched_count / len(job_desc.mandatory_skills)) * 100, 100.0), missing_skills


    def _calculate_experience_score(self, resume: ResumeSchema, job_desc: JobDescription) -> float:
        # Embedding similarity between job description and resume experience
        if not resume.experience:
            return 0.0
        
        job_embedding = self.model.encode(job_desc.description, convert_to_tensor=True)
        # Use extracted experience descriptions
        exp_text = " ".join([exp.description or "" for exp in resume.experience])
        if not exp_text.strip():
             return 0.0

        exp_embedding = self.model.encode(exp_text, convert_to_tensor=True)
        
        similarity = util.cos_sim(job_embedding, exp_embedding).item()
        # Scale similarity (usually 0.4-0.8) to 0-100 score
        # Normalizing: <0.2 -> 0, >0.7 -> 100
        score = max(0, min((similarity - 0.2) * 200, 100))
        return score

    def _calculate_project_score(self, resume: ResumeSchema, job_desc: JobDescription) -> float:
        """
        Evaluates projects based on:
        1. Relevance to Job (Do they use the job's skills?)
        2. Complexity (Basic/Medium/Advanced via Tech Stack) 
        3. Outcome (Metrics present?)
        """
        if not resume.projects:
            return 0.0
            
        total_proj_score = 0
        
        for proj in resume.projects:
            # A. Relevance Score
            relevance = 0
            desc_lower = (proj.description or "").lower()
            # Check if job skills are in project description
            for skill in job_desc.mandatory_skills:
                if skill.lower() in desc_lower:
                    relevance += 1
            
            rel_score = min((relevance / max(1, len(job_desc.mandatory_skills))) * 100, 100)
            
            # B. Complexity & Outcome Score
            comp_score, level = self._calculate_complexity_score(proj.description)
            proj.complexity = level # Save to schema for explanation later
            
            # Final score for this project
            # Weighted: 50% Relevance, 50% Quality (Complexity+Outcome)
            total_proj_score += (rel_score * 0.5 + comp_score * 0.5)
            
        # Average across projects
        avg_score = total_proj_score / len(resume.projects)
        return min(avg_score, 100.0)

    def _calculate_complexity_score(self, text: str) -> (float, str):
        if not text: return 0.0, "Basic"
        
        score = 0
        text_lower = text.lower()
        
        # 1. Tech Stack Count (Diversity)
        # Note: In a real run, we would pass the 'doc' from spaCy, but here we can just regex for now 
        # or use what we extracted. For Engine, we'll re-scan or assume 'technologies' field is populated.
        # Since 'technologies' field in Project is populated by Extractor (we need to ensure that),
        # Let's rely on heuristic scanning again if Schema isn't fully filled yet.
        
        # Heuristic scan for "Advanced" keywords
        advanced_keywords = ['docker', 'kubernetes', 'aws', 'ci/cd', 'microservices', 'redis', 'graphql', 'kafka', 'terraform', 'blochain', 'ai', 'machine learning', 'deep learning']
        medium_keywords = ['react', 'angular', 'node', 'django', 'flask', 'sql', 'database', 'api', 'rest']
        
        adv_count = sum(1 for k in advanced_keywords if k in text_lower)
        med_count = sum(1 for k in medium_keywords if k in text_lower)
        
        if adv_count >= 2:
            score += 50 # Base for Advanced
            level = "Advanced"
        elif med_count >= 2 or adv_count >= 1:
            score += 30 # Base for Medium
            level = "Medium"
        else:
            score += 10 # Base for Basic
            level = "Basic"
            
        # 2. Measurable Outcomes (Metrics)
        # Regex for "20%", "500 users", "10x speed"
        import re
        metrics = re.findall(r'\d+%|\d+x|\d+ users|\d+ customers|\d+k downloads', text_lower)
        
        if len(metrics) > 0:
            score += 20 # Bonus for outcomes
            if len(metrics) > 2:
                score += 10 # Extra bonus
                
        # 3. Length/Depth
        if len(text.split()) > 50:
            score += 10
            
        return min(score, 100.0), level

    def _calculate_structure_score(self, resume: ResumeSchema) -> float:
        score = 100.0
        if not resume.email: score -= 20
        if not resume.phone: score -= 20
        if not resume.skills: score -= 20
        return max(score, 0.0)
