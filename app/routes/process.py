from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from starlette.concurrency import run_in_threadpool
import os
import json
import uuid
import queue
import threading
from typing import Optional
from ..utils import save_upload_file_tmp, read_metadata, DATA_DIR
from ..services.pipeline_service import run_pipeline
import logging
import base64

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/pdf/stream")
async def process_pdf_stream(
    file: UploadFile = File(...), 
    max_pages: Optional[int] = None, 
    download: Optional[bool] = False, 
    background_tasks: BackgroundTasks = None
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF uploads are supported')
    
    tmp_path, filename = await run_in_threadpool(save_upload_file_tmp, file)

    # FIX 1: Limit queue size to prevent memory leaks if client disconnects
    q = queue.Queue(maxsize=100)
    job_id = str(uuid.uuid4())
    logger.info("Received upload %s -> %s; job=%s", file.filename, tmp_path, job_id)

    def progress_hook(ev: dict):
        ev_out = {"job_id": job_id, **ev}
        try:
            q.put(ev_out, timeout=1)  # Don't block indefinitely if queue is full
        except queue.Full:
            pass # Drop event if consumer is too slow

    # --- BLOCKING PIPELINE EXECUTION ---
    # FIX 2: Run heavy blocking code in a threadpool so main loop stays free
    # We define a wrapper to handle exceptions cleanly within the thread
    def _safe_run_pipeline():
        return run_pipeline(
            tmp_path, 
            max_pages=max_pages, 
            progress_hook=progress_hook, 
            doc_id=job_id, 
            original_filename=filename, 
            keep_report=True # We always keep report initially to ensure we have a file to send
        )

    if download:
        try:
            # Await the threadpool execution
            ret = await run_in_threadpool(_safe_run_pipeline)
        except Exception as e:
            # Cleanup tmp on error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=str(e))

        # Determine the file path to return
        final_report_path = None
        
        # Check if pipeline returned a FileResponse or a dict
        if isinstance(ret, FileResponse):
            final_report_path = ret.path
        elif isinstance(ret, dict):
            final_report_path = ret.get('report_path')
            # If no path but we have text (e.g. ingestion billing error), write it now
            if not final_report_path and ret.get('report_text'):
                fp = os.path.join(DATA_DIR, f"report_{job_id}.md")
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(ret.get('report_text'))
                final_report_path = fp

        if not final_report_path or not os.path.exists(final_report_path):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise HTTPException(status_code=500, detail='Report generation failed or file not found')

        # FIX 3: Robust Cleanup using BackgroundTasks
        # We perform cleanup AFTER the response is sent
        def _cleanup(paths_to_remove):
            for p in paths_to_remove:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {p}: {e}")

        if background_tasks is None:
            background_tasks = BackgroundTasks()
        
        # Schedule removal of both the uploaded PDF and the generated Report
        background_tasks.add_task(_cleanup, [tmp_path, final_report_path])

        return FileResponse(
            final_report_path, 
            media_type='text/markdown', 
            filename=os.path.basename(final_report_path)
        )

    # --- STREAMING RESPONSE HANDLING ---
    
    def worker():
        try:
            # This runs in a separate thread, so blocking here is fine for THAT thread
            ret = run_pipeline(
                tmp_path, 
                max_pages=max_pages, 
                progress_hook=progress_hook, 
                doc_id=job_id, 
                original_filename=filename
            )
            
            # Handle download payload logic
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
        finally:
            # Worker is responsible for cleaning up the input PDF
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_generator():
        try:
            while True:
                # Non-blocking check to allow loop to release if needed
                try:
                    # Retrieve from queue without blocking the async loop
                    # We use run_in_threadpool for the blocking queue.get if we wanted 
                    # absolute purity, but q.get with timeout is acceptable in small bursts.
                    # Better: check empty first or use very short timeout
                    ev = q.get(timeout=0.1) 
                    
                    s = f"data: {json.dumps(ev)}\n\n"
                    yield s.encode('utf-8')
                    
                    if ev.get("event") in ("worker_done", "error"):
                        break
                        
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    # Yield control back to event loop
                    await run_in_threadpool(lambda: time.sleep(0.01)) 
                    continue
        except Exception:
            # Client disconnected
            pass

    return StreamingResponse(event_generator(), media_type='text/event-stream')

@router.get("/report/download/{job_id}")
async def download_report_by_job(job_id: str):
    """Download the generated markdown report by `job_id`."""
    try:
        # metadata read is fast enough, but ideally should be async or db-based
        entries = await run_in_threadpool(read_metadata)
    except Exception:
        entries = []

    for e in reversed(entries):
        if str(e.get('uuid')) == str(job_id):
            path = e.get('report')
            if path and os.path.exists(path):
                return FileResponse(path, media_type='text/markdown', filename=os.path.basename(path))
            else:
                raise HTTPException(status_code=404, detail='Report file not found')
    raise HTTPException(status_code=404, detail='Job not found')