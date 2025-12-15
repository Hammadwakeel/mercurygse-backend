PDF Extraction FastAPI
======================

This repository provides a FastAPI wrapper around a PDF visual/text extraction pipeline.

Structure
- `app/` — application package
  - `main.py` — FastAPI app and endpoints
  - `pipeline.py` — refactored pipeline logic (supports `dry_run=True` to avoid external APIs)
  - `qdrant_ingest.py` — markdown chunking + ingest placeholder
  - `utils.py` — helpers

Quickstart
1. Install system dependency for `pdf2image` (Debian/Ubuntu):

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

2. Create virtualenv and install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Use the `/process` endpoint to upload a PDF (Swagger UI at `/docs`). For quick local testing use `dry_run=true`.

Notes
- The code includes only a simulated/dry-run mode for model calls — enable real model usage by integrating your API keys and adding real calls into `app/pipeline.py` where marked.
- To ingest into Qdrant, provide your credentials and implement vectorization in `app/qdrant_ingest.py` (placeholder included).
