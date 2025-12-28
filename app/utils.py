import os
import tempfile
import time
from typing import Tuple, List, Dict, Optional
from supabase import create_client, Client
from .core import config as core_config

# Initialize Supabase Client
_cfg = core_config.load_config()
supabase: Client = create_client(_cfg["SUPABASE_URL"], _cfg["SUPABASE_KEY"])
BUCKET_NAME = _cfg["STORAGE_BUCKET"]

# Local data dir for temp processing only
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

def save_upload_file_tmp(upload_file) -> Tuple[str, str]:
    """Save a FastAPI UploadFile to a temporary file and return (tmp_path, filename)."""
    suffix = os.path.splitext(upload_file.filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        content = upload_file.file.read()
        out.write(content)
    return tmp_path, upload_file.filename

# --- Supabase Storage Helpers ---

def upload_file_to_bucket(file_path: str, destination_path: str) -> str:
    """Uploads a local file to Supabase Storage and returns the bucket path."""
    try:
        with open(file_path, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=destination_path,
                file=f,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
        return destination_path
    except Exception as e:
        print(f"Supabase Upload failed: {e}")
        # Proceeding allows the pipeline to continue even if backup fails, 
        # but in strict mode you might want to raise.
        return ""

def get_signed_url(bucket_path: str, expiry_seconds=3600) -> Optional[str]:
    """Generates a secure, temporary download link."""
    try:
        res = supabase.storage.from_(BUCKET_NAME).create_signed_url(bucket_path, expiry_seconds)
        return res.get("signedURL")
    except Exception as e:
        print(f"Failed to generate signed URL: {e}")
        return None

# --- Supabase Database Helpers ---

def append_metadata_entry(entry: dict):
    """Insert a new job entry into Supabase DB."""
    try:
        data = {
            "job_id": str(entry.get("uuid")),
            "original_filename": entry.get("original_filename"),
            "report_path": entry.get("report"), # Stores the BUCKET PATH (e.g., jobs/123/report.md)
            "created_at": int(entry.get("created_at", time.time())),
            "expires_at": int(entry.get("expires_at", time.time() + 86400))
        }
        supabase.table("job_metadata").insert(data).execute()
    except Exception as e:
        print(f"Error writing metadata to Supabase: {e}")

def read_metadata(limit: int = 100) -> List[Dict]:
    """Fetch recent metadata entries from Supabase."""
    try:
        response = supabase.table("job_metadata")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        results = []
        for row in response.data:
            results.append({
                "uuid": row["job_id"],
                "original_filename": row["original_filename"],
                "report": row["report_path"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"]
            })
        return results
    except Exception as e:
        print(f"Error reading metadata from Supabase: {e}")
        return []

# Cleanup is now less critical for local disk, but good for DB hygiene
# Only runs if explicitly started by main.py
def cleanup_expired_reports(retention_seconds: int = 24 * 3600, interval_seconds: int = 60 * 60):
    """Background loop that cleans up old DB entries. Storage lifecycle rules should handle files."""
    while True:
        try:
            now = int(time.time())
            # Simple cleanup: Delete DB rows where expires_at < now
            supabase.table("job_metadata").delete().lt("expires_at", now).execute()
        except Exception:
            pass
        time.sleep(interval_seconds)

_cleanup_thread_started = False

def start_cleanup_thread(retention_seconds: int = 24 * 3600, interval_seconds: int = 60 * 60):
    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    # Use local import to keep global namespace clean
    import threading as _th
    t = _th.Thread(target=cleanup_expired_reports, args=(retention_seconds, interval_seconds), daemon=True)
    t.start()
    _cleanup_thread_started = True


def get_job_by_filename(filename: str) -> Optional[Dict]:
    """Fetch a job entry by its original filename."""
    try:
        # Query Supabase for the filename
        response = supabase.table("job_metadata")\
            .select("*")\
            .eq("original_filename", filename)\
            .limit(1)\
            .execute()
        
        if response.data and len(response.data) > 0:
            row = response.data[0]
            return {
                "uuid": row["job_id"],
                "original_filename": row["original_filename"],
                "report": row["report_path"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"]
            }
    except Exception as e:
        print(f"Error checking duplicate: {e}")
    return None

def list_all_jobs(limit: int = 100) -> Dict[str, List[str]]:
    """Return a separated list of PDF filenames and MD report paths."""
    try:
        data = read_metadata(limit)
        pdf_files = [item["original_filename"] for item in data]
        md_files = [item["report"] for item in data if item.get("report")]
        return {
            "pdf_files": pdf_files,
            "md_files": md_files,
            "full_data": data # Useful if frontend needs ID mapping
        }
    except Exception as e:
        print(f"Error listing files: {e}")
        return {"pdf_files": [], "md_files": []}