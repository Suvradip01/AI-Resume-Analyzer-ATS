# ── InSightATS Backend — Hugging Face Spaces Dockerfile ──────
FROM python:3.11-slim

# System deps needed by pdfplumber (poppler) and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies first (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and models
# In HF Spaces, models are pushed via Git LFS and are present in the repo
COPY --chown=user:user . /app/

# Switch to the non-root user
USER user

# Hugging Face Spaces use port 7860
ENV PORT=7860
ENV APP_ENV=production
# We store the DB in the app directory. Note: On HF Spaces Free tier, this resets if the space sleeps.
ENV DB_PATH=/app/db.sqlite3
ENV MODEL_DIR=/app/models

EXPOSE 7860

CMD ["python", "run.py"]
