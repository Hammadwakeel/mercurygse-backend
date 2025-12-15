import os
import tempfile
from typing import Tuple
import json
import time
import threading

# metadata storage for lightweight JSON DB
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
_metadata_lock = threading.Lock()


def save_upload_file_tmp(upload_file) -> Tuple[str, str]:
    """Save a FastAPI UploadFile to a temporary file and return (tmp_path, filename)."""
    suffix = os.path.splitext(upload_file.filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        content = upload_file.file.read()
        out.write(content)
    return tmp_path, upload_file.filename


def append_metadata_entry(entry: dict):
    """Append an entry to the metadata JSON file (list of entries). Thread-safe."""
    with _metadata_lock:
        data = []
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(entry)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def read_metadata() -> list:
    with _metadata_lock:
        if not os.path.exists(METADATA_PATH):
            return []
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


def write_metadata(entries: list):
    with _metadata_lock:
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)


def cleanup_expired_reports(retention_seconds: int = 24 * 3600, interval_seconds: int = 60 * 60):
    """Background loop that removes expired report files and prunes metadata entries.

    - `retention_seconds` controls how long reports are kept after `created_at` or `expires_at`.
    - `interval_seconds` controls how often the background loop runs.
    """
    while True:
        try:
            entries = read_metadata()
            now = time.time()
            changed = False
            remaining = []
            for e in entries:
                expires = e.get('expires_at') or (e.get('created_at', 0) + retention_seconds)
                if expires and now >= float(expires):
                    # delete file if present
                    p = e.get('report')
                    try:
                        if p and os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                    changed = True
                else:
                    remaining.append(e)
            if changed:
                write_metadata(remaining)
        except Exception:
            pass
        time.sleep(interval_seconds)


_cleanup_thread_started = False


def start_cleanup_thread(retention_seconds: int = 24 * 3600, interval_seconds: int = 60 * 60):
    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    import threading as _th
    t = _th.Thread(target=cleanup_expired_reports, args=(retention_seconds, interval_seconds), daemon=True)
    t.start()
    _cleanup_thread_started = True
