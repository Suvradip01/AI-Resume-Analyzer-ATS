import os

class Settings:
    PROJECT_NAME: str = "AI Resume Analyzer"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Paths - optional to make configurable
    SPACY_MODEL: str = "en_core_web_sm"
    TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

settings = Settings()
