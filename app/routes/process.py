from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
import os
import json
import uuid
import queue
import threading
from typing import Optional
from ..utils import save_upload_file_tmp, read_metadata
from ..services.pipeline_service import run_pipeline
import logging
import base64

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/pdf/stream")
async def process_pdf_stream(file: UploadFile = File(...), max_pages: Optional[int] = None, download: Optional[bool] = False, background_tasks: BackgroundTasks = None):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF uploads are supported')
    tmp_path, filename = save_upload_file_tmp(file)

    q = queue.Queue()
    job_id = str(uuid.uuid4())
    logger.info("Received upload %s -> %s; job=%s", file.filename, tmp_path, job_id)
    def progress_hook(ev: dict):
        ev_out = {"job_id": job_id, **ev}
        q.put(ev_out)

    # If client requested a blocking download, run pipeline and return the file response
    if download:
        try:
            # keep_report so pipeline does not delete the report; run_pipeline may return a FileResponse when keep_report=True
            ret = run_pipeline(tmp_path, max_pages=max_pages, progress_hook=progress_hook, doc_id=job_id, original_filename=filename, keep_report=True)
            # If run_pipeline returned a FileResponse, return it directly to the client
            if hasattr(ret, 'background') or isinstance(ret, FileResponse):
                return ret
        except Exception as e:
            # ensure tmp is cleaned
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=str(e))

        # determine report path; if no file but report_text present, write a temp file
        report_path = None
        if isinstance(ret, dict):
            report_path = ret.get('report_path')
            report_text = ret.get('report_text')
        else:
            report_text = None

        if not report_path and report_text:
            # write report to DATA_DIR
            from ..utils import DATA_DIR
            fp = os.path.join(DATA_DIR, f"report_{job_id}.md")
            with open(fp, 'w', encoding='utf-8') as f:
                f.write(report_text)
            report_path = fp

        if not report_path or not os.path.exists(report_path):
            # nothing to return
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail='Report not available')

        # schedule cleanup after response
        def _cleanup(path_tmp: str, path_report: str):
            try:
                if os.path.exists(path_tmp):
                    os.remove(path_tmp)
            except Exception:
                pass
            try:
                if os.path.exists(path_report):
                    os.remove(path_report)
            except Exception:
                pass

        if background_tasks is None:
            background_tasks = BackgroundTasks()
        background_tasks.add_task(_cleanup, tmp_path, report_path)
        return FileResponse(report_path, media_type='text/markdown', filename=os.path.basename(report_path))

    

    def worker():
        try:
            ret = run_pipeline(tmp_path, max_pages=max_pages, progress_hook=progress_hook, doc_id=job_id, original_filename=filename)
            # If the pipeline returned report_text (e.g., ingestion skipped due to billing), attach a downloadable payload
            try:
                if isinstance(ret, dict) and ret.get("report_text"):
                    rpt_text = ret.get("report_text")
                    rpt_path = ret.get("report_path") or f"report_{job_id}.md"
                    b64 = base64.b64encode(rpt_text.encode('utf-8')).decode('utf-8')
                    q.put({"job_id": job_id, "event": "report_download", "report_filename": os.path.basename(rpt_path), "report_b64": b64})
            except Exception:
                logger.exception("Failed to attach report download for job %s", job_id)
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



@router.get("/report/download/{job_id}")
async def download_report_by_job(job_id: str):
    """Download the generated markdown report by `job_id` (uuid stored in metadata).

    Falls back to 404 if not found. This is the recommended, safe way to retrieve reports.
    """
    try:
        entries = read_metadata()
    except Exception:
        entries = []

    # prefer most recent matching entry
    for e in reversed(entries):
        if str(e.get('uuid')) == str(job_id):
            path = e.get('report')
            if path and os.path.exists(path):
                return FileResponse(path, media_type='text/markdown', filename=os.path.basename(path))
            else:
                raise HTTPException(status_code=404, detail='Report file not found')
    raise HTTPException(status_code=404, detail='Job not found')
