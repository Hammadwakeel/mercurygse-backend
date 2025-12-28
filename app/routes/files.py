from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool
from typing import Optional

from ..utils import (
    list_all_jobs, 
    get_job_by_filename, 
    get_signed_url
)

router = APIRouter()

@router.get("/list")
async def list_files():
    """
    Returns a list of all uploaded PDF filenames and their corresponding 
    Markdown report paths currently stored in the database.
    """
    result = await run_in_threadpool(list_all_jobs)
    return {
        "status": "success",
        "count": len(result["pdf_files"]),
        "uploaded_pdfs": result["pdf_files"],
        "generated_reports": result["md_files"]
    }

@router.get("/download")
async def download_file_by_name(filename: str = Query(..., description="The exact name of the uploaded PDF file")):
    """
    Takes a file name (e.g., 'document.pdf') as input and returns the 
    download URL for its generated Markdown report.
    """
    # 1. Find the job associated with this filename
    job = await run_in_threadpool(get_job_by_filename, filename)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in records.")

    # 2. Get the path to the report
    report_path = job.get("report")
    if not report_path:
        raise HTTPException(status_code=404, detail="Report path is missing for this file.")

    # 3. Generate a secure download link
    download_url = await run_in_threadpool(get_signed_url, report_path)
    
    if not download_url:
        raise HTTPException(status_code=500, detail="Could not generate download link.")

    return {
        "filename": filename,
        "job_id": job.get("uuid"),
        "report_download_url": download_url
    }