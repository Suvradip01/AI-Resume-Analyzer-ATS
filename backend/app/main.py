from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import resume
from app.api.endpoints import recruiter

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Powered Resume Analyzer Backend"
)

# CORS — origins are driven by settings.ALLOWED_ORIGINS (env-configurable for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(resume.router, prefix="/api/v1/resume", tags=["resume"])
app.include_router(recruiter.router, prefix="/api/v1/recruiter", tags=["recruiter"])

@app.get("/")
def root():
    return {"message": "Welcome to AI Resume Analyzer API"}
