from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import os
import json
import uuid
import queue
import threading
from typing import Optional
from ..utils import save_upload_file_tmp
from ..services.pipeline_service import run_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
@router.post("/pdf/stream")
async def process_pdf_stream(file: UploadFile = File(...), max_pages: Optional[int] = None):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF uploads are supported')
    tmp_path, filename = save_upload_file_tmp(file)

    q = queue.Queue()
    job_id = str(uuid.uuid4())
    logger.info("Received upload %s -> %s; job=%s", file.filename, tmp_path, job_id)

    def progress_hook(ev: dict):
        ev_out = {"job_id": job_id, **ev}
        q.put(ev_out)

    def worker():
        try:
            run_pipeline(tmp_path, max_pages=max_pages, progress_hook=progress_hook, doc_id=job_id, original_filename=filename)
            q.put({"job_id": job_id, "event": "worker_done"})
        except Exception as e:
            q.put({"job_id": job_id, "event": "error", "error": str(e)})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_generator():
        try:
            while True:
                try:
                    ev = q.get(timeout=0.5)
                except Exception:
                    if not thread.is_alive():
                        break
                    continue
                # SSE format
                s = f"data: {json.dumps(ev)}\n\n"
                yield s.encode('utf-8')
            # drain any remaining events
            while not q.empty():
                ev = q.get()
                s = f"data: {json.dumps(ev)}\n\n"
                yield s.encode('utf-8')
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@router.get("/report")
async def download_report(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Report not found')
    return FileResponse(path, media_type='text/markdown', filename=os.path.basename(path))
