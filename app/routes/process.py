from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import os
import json
import uuid
import queue
import threading
import time
from typing import Optional
import logging
from pdf2image import pdfinfo_from_path  # Added for security validation

from ..utils import (
    save_upload_file_tmp, 
    upload_file_to_bucket, 
    get_signed_url, 
    append_metadata_entry, 
    read_metadata,
    DATA_DIR
)
from ..services.pipeline_service import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/pdf/stream")
async def process_pdf_stream(
    file: UploadFile = File(...), 
    max_pages: Optional[int] = None, 
    download: Optional[bool] = False, 
    background_tasks: BackgroundTasks = None
):
    # Basic extension check (first line of defense)
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Only PDF uploads are supported')
    
    # 1. Save locally first (needed for processing) - Non-blocking I/O
    tmp_path, filename = await run_in_threadpool(save_upload_file_tmp, file)
    
    # --- SECURITY CHECK: Validate actual file content ---
    # Attempt to read PDF info. If this fails, it's not a valid PDF.
    try:
        await run_in_threadpool(pdfinfo_from_path, tmp_path)
    except Exception:
        # Cleanup the invalid file immediately
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        logger.warning(f"Uploaded file {filename} claimed to be PDF but failed validation.")
        raise HTTPException(status_code=400, detail="Invalid PDF file. The file is corrupted or not a valid PDF document.")
    # ----------------------------------------------------

    job_id = str(uuid.uuid4())
    logger.info("Received upload %s -> %s; job=%s", file.filename, tmp_path, job_id)

    # 2. Upload Original PDF to Supabase Bucket
    pdf_bucket_path = f"jobs/{job_id}/source.pdf"
    await run_in_threadpool(upload_file_to_bucket, tmp_path, pdf_bucket_path)

    # Setup Queue for Streaming
    q = queue.Queue(maxsize=100)

    def progress_hook(ev: dict):
        ev_out = {"job_id": job_id, **ev}
        try:
            q.put(ev_out, timeout=1)
        except queue.Full:
            pass

    # Wrapper for the heavy blocking pipeline
    def _safe_run_pipeline():
        return run_pipeline(
            tmp_path, 
            max_pages=max_pages, 
            progress_hook=progress_hook, 
            doc_id=job_id, 
            original_filename=filename
        )

    # --- BLOCKING / DOWNLOAD MODE ---
    if download:
        try:
            # Run pipeline in threadpool
            ret = await run_in_threadpool(_safe_run_pipeline)
            
            # Identify the generated local report path
            local_report_path = None
            if isinstance(ret, dict):
                local_report_path = ret.get('report_path')
                # Fallback for error texts (e.g. billing limits)
                if not local_report_path and ret.get('report_text'):
                    fp = os.path.join(DATA_DIR, f"report_{job_id}.md")
                    with open(fp, 'w', encoding='utf-8') as f:
                        f.write(ret.get('report_text'))
                    local_report_path = fp

            if not local_report_path or not os.path.exists(local_report_path):
                raise HTTPException(status_code=500, detail='Report generation failed')

            # 3. Upload Result to Supabase
            md_bucket_path = f"jobs/{job_id}/report.md"
            await run_in_threadpool(upload_file_to_bucket, local_report_path, md_bucket_path)

            # 4. Save Metadata (pointing to Bucket Path)
            entry = {
                "uuid": job_id,
                "original_filename": filename,
                "report": md_bucket_path, 
                "created_at": time.time(),
                "expires_at": time.time() + 86400
            }
            await run_in_threadpool(append_metadata_entry, entry)

            # 5. Return JSON with Signed URL (Stateless & Fast)
            download_url = await run_in_threadpool(get_signed_url, md_bucket_path)
            
            # Cleanup local files immediately
            def _cleanup_local(paths):
                for p in paths:
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except: pass

            if background_tasks is None:
                background_tasks = BackgroundTasks()
            background_tasks.add_task(_cleanup_local, [tmp_path, local_report_path])

            return {
                "job_id": job_id,
                "status": "completed",
                "download_url": download_url,
                "message": "Report generated and stored securely."
            }

        except Exception as e:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            raise HTTPException(status_code=500, detail=str(e))

    # --- STREAMING MODE ---
    
    def worker():
        try:
            # Worker runs in its own thread, calls blocking pipeline
            ret = run_pipeline(
                tmp_path, 
                max_pages=max_pages, 
                progress_hook=progress_hook, 
                doc_id=job_id, 
                original_filename=filename
            )
            
            # Handle Upload & Download Link Generation in Worker
            try:
                local_report_path = None
                if isinstance(ret, dict):
                    local_report_path = ret.get("report_path")
                
                if local_report_path and os.path.exists(local_report_path):
                    # Upload
                    md_bucket_path = f"jobs/{job_id}/report.md"
                    upload_file_to_bucket(local_report_path, md_bucket_path)
                    
                    # Metadata
                    entry = {
                        "uuid": job_id,
                        "original_filename": filename,
                        "report": md_bucket_path,
                        "created_at": time.time(),
                        "expires_at": time.time() + 86400
                    }
                    append_metadata_entry(entry)

                    # Send Download URL event
                    url = get_signed_url(md_bucket_path)
                    q.put({"job_id": job_id, "event": "report_ready", "download_url": url})
                    
                    # Cleanup report
                    os.remove(local_report_path)

            except Exception as e:
                logger.exception("Worker post-processing failed")

            q.put({"job_id": job_id, "event": "worker_done"})
            
        except Exception as e:
            q.put({"job_id": job_id, "event": "error", "error": str(e)})
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except: pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_generator():
        try:
            while True:
                try:
                    ev = q.get(timeout=0.1)
                    s = f"data: {json.dumps(ev)}\n\n"
                    yield s.encode('utf-8')
                    if ev.get("event") in ("worker_done", "error"):
                        break
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    await run_in_threadpool(lambda: time.sleep(0.01))
                    continue
        except Exception:
            pass

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@router.get("/report/download/{job_id}")
async def download_report_by_job(job_id: str):
    """Generates a secure, temporary download link for the report."""
    try:
        entries = await run_in_threadpool(read_metadata)
    except Exception:
        entries = []

    # Find entry
    target = next((e for e in entries if str(e.get('uuid')) == job_id), None)
    
    if not target:
        raise HTTPException(status_code=404, detail='Job not found')
        
    bucket_path = target.get('report')
    if not bucket_path:
        raise HTTPException(status_code=404, detail='Report path missing')

    url = await run_in_threadpool(get_signed_url, bucket_path)
    
    if not url:
        raise HTTPException(status_code=404, detail='File not found in storage')

    return {"job_id": job_id, "download_url": url}