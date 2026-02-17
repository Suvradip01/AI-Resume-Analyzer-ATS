# AI Resume Analyzer Backend

## Prerequisites

- Python 3.9+
- pip

## Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
   *Note: `all-MiniLM-L6-v2` will download automatically on first run.*

## Running the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Docs are available at `http://127.0.0.1:8000/docs`.
