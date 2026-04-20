# InSightATS Frontend (React + Vite)

## Prerequisites

- Node.js 18+ (your machine: Node 22 works)
- npm

## Setup

```bash
cd frontend
npm install
```

## Run

```bash
npm run dev
```

Vite will print the local URL (typically `http://127.0.0.1:5173`).

## Backend integration

The dashboard submits:

- `resume_file` (PDF/DOCX/TXT)
- `job_description` (JSON string)

To the backend endpoint `POST /api/v1/resume/analyze`.
