from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import resume

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI Powered Resume Analyzer Backend"
)

# CORS Configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(resume.router, prefix="/api/v1/resume", tags=["resume"])

@app.get("/")
def root():
    return {"message": "Welcome to AI Resume Analyzer API"}
