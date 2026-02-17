import spacy
import re
from typing import List, Dict
from app.schemas.resume import ResumeSchema, Experience, Project, Education
from app.core.config import settings

class NlpExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
        except OSError:
            print(f"Downloading {settings.SPACY_MODEL}...")
            from spacy.cli import download
            download(settings.SPACY_MODEL)
            self.nlp = spacy.load(settings.SPACY_MODEL)
        
        # Add EntityRuler for custom tech stack and metrics
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = self._get_patterns()
        ruler.add_patterns(patterns)

    def _get_patterns(self):
        # Define patterns for Tech Stacks and Metrics
        tech_stacks = [
            "Python", "Java", "C++", "JavaScript", "TypeScript", "React", "Angular", "Vue", "Next.js", "Node.js", 
            "Django", "FastAPI", "Flask", "Spring Boot", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "Redis", "Kafka", "RabbitMQ", "GraphQL", "REST API",
            "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "OpenCV",
            "Git", "CI/CD", "Jenkins", "GitHub Actions", "Linux", "Bash", "Shell"
        ]
        
        patterns = []
        for tech in tech_stacks:
            patterns.append({"label": "TECH", "pattern": [{"LOWER": tech.lower()}]})
            
        # Metric patterns (heuristic-based)
        # Metric patterns (heuristic-based)
        # e.g., "20%", "500 users", "10x"
        patterns.append({"label": "METRIC", "pattern": [{"TEXT": {"REGEX": r"\d+%"}}]})
        patterns.append({"label": "METRIC", "pattern": [{"TEXT": {"REGEX": r"\d+x"}}]})
        patterns.append({"label": "METRIC", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["users", "customers", "clients", "downloads", "requests"]}}]})
        
        return patterns

    def extract_info(self, text: str) -> ResumeSchema:
        doc = self.nlp(text)
        
        name = self._extract_name(doc)
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        skills = self._extract_skills(doc)
        
        # Simplified extraction for demo purposes; production needs complex regex/ML
        experience = self._extract_experience_sections(text)
        projects = self._extract_project_sections(text)
        
        return ResumeSchema(
            name=name,
            email=email,
            phone=phone,
            skills=skills,
            experience=experience,
            projects=projects,
            education=[], # Placeholder
            text_content=text
        )

    def _extract_name(self, doc):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def _extract_email(self, text):
        match = re.search(r'[\w\.-]+@[\w\.-]+', text)
        return match.group(0) if match else None

    def _extract_phone(self, text):
        match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        return match.group(0) if match else None

    def _extract_skills(self, doc) -> List[str]:
        skills = []
        # Use our custom TECH entity
        for ent in doc.ents:
            if ent.label_ == "TECH":
                skills.append(ent.text)
        
        # Fallback to noun chunks if no tech found (or mostly non-tech resume)
        if not skills:
             for token in doc:
                if token.pos_ == "NOUN" and token.is_alpha:
                    skills.append(token.text)
             skills = list(set(skills))[:10]
             
        return list(set(skills))

    def _extract_experience_sections(self, text: str) -> List[Experience]:
        """
        Extracts experience entries using regex to find 'Experience' sections
        and split by dates/titles.
        """
        experiences = []
        
        # 1. Isolate Experience Section
        # Look for "Experience", "Work History", "Employment"
        # Stop at "Education", "Projects", "Skills"
        pattern = r"(?i)(?:experience|work history|employment)([\s\S]*?)(?:education|projects|skills|certifications|$)"
        match = re.search(pattern, text)
        
        if not match:
            return []
            
        section_text = match.group(1).strip()
        
        # 2. Split into individual roles (Heuristic: Date patterns often stick out)
        # Pattern for dates: Jan 2020 - Present, 01/2020 - 02/2021
        date_pattern = r"(?i)(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{4}|\d{1,2}/\d{4})\s*(?:-|to|–)\s*(?:Present|Now|Current|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{4}|\d{1,2}/\d{4})"
        
        # We split the text by dates, assuming the text BEFORE a date is the Title/Company
        # This is tricky without a visual layout parser, but works for text dumps.
        chunks = re.split(date_pattern, section_text)
        
        # Determine if chunks contain meaningful info
        if len(chunks) > 1:
            # Simple heuristic: Create an entry for each found date-block
            # Refinement needed for production for exact Title vs Company split
            for i in range(0, len(chunks)-1, 2):
                # chunks[i] might be the title/company from previous split
                # chunks[i+1] is the captured date string
                # chunks[i+2] is the description
                
                # Handling the split artifacts
                desc = chunks[i+2] if i+2 < len(chunks) else ""
                
                experiences.append(Experience(
                    title="Role/Company Detected", # Placeholder for now as text dump loses structure
                    compay="Unknown",
                    description=desc[:200] + "...", # Truncate for summary
                    start_date=chunks[i+1] # The matched date
                ))
        else:
             # Fallback: Treat whole section as one block
             experiences.append(Experience(description=section_text[:500]))

        return experiences

    def _extract_project_sections(self, text: str) -> List[Project]:
        """
        Extracts projects looking for 'Projects' header.
        """
        projects = []
        
        # 1. Isolate Projects Section
        pattern = r"(?i)(?:projects|personal projects|academic projects)([\s\S]*?)(?:education|experience|skills|certifications|$)"
        match = re.search(pattern, text)
        
        if not match:
            return []
            
        section_text = match.group(1).strip()
        
        # 2. Split by bullet points or aggressive newlines if no dates
        # Heuristic: Each project often starts with a Name followed by technologies
        
        lines = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        current_project = None
        
        for line in lines:
            # If line looks like a title (short, no punctuation at end, maybe bold in original but we only have text)
            # This is hard on raw text.
            # Using simple block grouping: 
            
            # For now, let's treat every 3-4 lines as a project block for demo purposes
            # OR look for "Project Name:" pattern
            
            if not current_project:
                current_project = Project(name="Project Detected", description="")
            
            current_project.description += line + " "
            
            if len(current_project.description) > 200:
                projects.append(current_project)
                current_project = None
                
        if current_project:
            projects.append(current_project)
            
        return projects
