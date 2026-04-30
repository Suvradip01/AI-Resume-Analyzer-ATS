# ── InSightATS Backend — Production Dockerfile (Oracle Cloud ARM / x86) ──────
FROM python:3.11-slim

# System deps needed by pdfplumber (poppler) and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY run.py .

# Models are NOT baked into the image — they are mounted from Oracle block storage
# at runtime via the MODEL_DIR env variable (saves 1.14 GB of image size)
# SQLite DB also lives outside the container on the persistent volume
ENV MODEL_DIR=/data/models
ENV DB_PATH=/data/db.sqlite3
ENV APP_ENV=production
ENV PORT=8000

# Expose the Uvicorn port (Nginx will proxy to this)
EXPOSE 8000

CMD ["python", "run.py"]
