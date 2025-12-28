import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

def load_config() -> Dict[str, str]:
    return {
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "VOYAGE_API_KEY": os.environ.get("VOYAGE_API_KEY"),
        "QDRANT_URL": os.environ.get("QDRANT_URL"),
        "QDRANT_API_KEY": os.environ.get("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.environ.get("QDRANT_COLLECTION"),
        "QDRANT_BATCH_SIZE": os.environ.get("QDRANT_BATCH_SIZE"),
        # Supabase Config
        "SUPABASE_URL": os.environ.get("SUPABASE_URL"),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY"),
        "STORAGE_BUCKET": os.environ.get("STORAGE_BUCKET"),
    }