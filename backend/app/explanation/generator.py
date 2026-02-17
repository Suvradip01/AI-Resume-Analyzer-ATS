from typing import List
from app.schemas.resume import ResumeSchema, EvaluationResult
from app.schemas.job import JobDescription

class ExplanationGenerator:
    @staticmethod
    def generate_feedback(score_data: dict, resume: ResumeSchema, job_desc: JobDescription) -> List[str]:
        feedback = []
        
        # 1. Mandatory Skills Gap
        # Use pre-calculated missing skills from Scoring Engine (which includes Semantic Matching)
        missing = score_data.get('missing_skills')
        
        # Fallback if not provided (shouldn't happen with updated engine)
        if missing is None:
            resume_text_lower = resume.text_content.lower()
            missing = []
            for skill in job_desc.mandatory_skills:
                 if not (any(skill.lower() in s.lower() for s in resume.skills) or skill.lower() in resume_text_lower):
                     missing.append(skill)
        
        if missing:
            feedback.append(f"❌ **Critical Missing Skills**: Your resume is missing: **{', '.join(missing)}**. These are mandatory for this role.")
        else:
            feedback.append("✅ **Skills**: You matched all mandatory skills. Good job!")

        # 2. Project Quality Analysis (Explainable AI)
        if not resume.projects:
            feedback.append("⚠️ **Projects**: No project section detected. Adding projects is the best way to show your skills.")
        else:
            for i, proj in enumerate(resume.projects):
                proj_name = proj.name or f"Project {i+1}"
                complexity = getattr(proj, 'complexity', 'Basic')
                
                # Complexity Feedback
                if complexity == "Basic":
                    feedback.append(f"⚠️ **{proj_name} (Basic Level)**: This project appears simple. To reach **Advanced** level, incorporate complex tech like **Docker, AWS, CI/CD, or Microservices**.")
                elif complexity == "Medium":
                    feedback.append(f"ℹ️ **{proj_name} (Medium Level)**: Good start. To make this stand out, try deploying it or adding **Scalability/Performance** optimizations.")
                else:
                    feedback.append(f"✅ **{proj_name} (Advanced Level)**: Strong project using advanced technology. Great work!")

                # Outcome Feedback (Metrics)
                desc = (proj.description or "").lower()
                import re
                has_metrics = bool(re.search(r'\d+%|\d+x|\d+ users|\d+ customers|\d+k downloads', desc))
                
                if not has_metrics:
                    feedback.append(f"💡 **Tip for {proj_name}**: You described *what* you did, but not the *result*. Add metrics like **'Reduced latency by 20%'** or **'Handled 500+ users'** to prove impact.")

        # 3. Experience Analysis
        if not resume.experience:
             feedback.append("⚠️ **Experience**: No experience section found. If you are a fresher, your Projects score is boosted, so make them count!")
        else:
             # Check if experience mentions job title keywords
             job_title_parts = job_desc.title.lower().split()
             exp_text = " ".join([e.description or "" for e in resume.experience]).lower()
             if not any(part in exp_text for part in job_title_parts if len(part) > 3):
                 feedback.append("💡 **Experience Relevance**: Your work history doesn't strongly reflect the target job title. Try to use similar terminology/keywords in your role descriptions.")

        # 4. Overall Score Interpretation
        score = score_data.get('total_score', 0)
        from app.scoring.engine import ScoringEngine
        
        # Check if score was boosted for fresher
        is_fresher_boost = (not resume.experience) and score > 80
        
        if is_fresher_boost:
             feedback.append(f"🚀 **Fresher Bonus**: Your score is high ({score}/100) because your Projects are Advanced, compensating for lack of experience. Keep it up!")
        elif score < 50:
            # Root Cause Analysis
            skill_score = score_data.get('skill_score', 0)
            exp_score = score_data.get('experience_score', 0)
            proj_score = score_data.get('project_score', 0)
            
            reasons = []
            if skill_score < 50: reasons.append("Missing Critical Skills")
            if exp_score < 50: reasons.append("Experience Mismatch")
            if proj_score < 50: reasons.append("Weak Project Descriptions")
            
            reason_str = f"Main issues: {', '.join(reasons)}." if reasons else ""
            feedback.append(f"📉 **Overall**: Your score is {score}/100. {reason_str} Focus on addressing these specific gaps.")
            
        elif score < 75:
            feedback.append(f"📈 **Overall**: Good score ({score}/100), but you can improve by adding metrics to your projects.")
        else:
            feedback.append(f"🏆 **Overall**: Excellent profile ({score}/100)! You are a strong candidate.")

        return feedback
