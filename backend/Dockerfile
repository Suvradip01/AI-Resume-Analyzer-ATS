# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# System deps needed by pdfplumber (poppler) and python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install them as the user
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application source and models
# In HF Spaces, models are pushed via Git LFS and are present in the repo
COPY --chown=user . /app

# Hugging Face Spaces use port 7860
ENV PORT=7860
ENV APP_ENV=production
# We store the DB in the app directory. Note: On HF Spaces Free tier, this resets if the space sleeps.
ENV DB_PATH=/app/db.sqlite3
ENV MODEL_DIR=/app/models

CMD ["python", "run.py"]
