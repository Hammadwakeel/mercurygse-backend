import os
import tempfile
from typing import Tuple
import json
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
