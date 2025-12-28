from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import your core configurations and services
from .core import config as core_config
from .services import model_client

# Import routers
from .routes import process, health, files

logger = logging.getLogger("pdf_extraction")
if not logger.handlers:
    # simple default handler
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

app = FastAPI(title="PDF Extraction Service")

# Optional: Add CORS if accessing from a frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # initialize clients from environment if available
    cfg = core_config.load_config()
    
    # Log presence (do not print secrets)
    logger.info("GOOGLE_API_KEY set: %s", bool(cfg.get("GOOGLE_API_KEY")))
    logger.info("VOYAGE_API_KEY set: %s", bool(cfg.get("VOYAGE_API_KEY")))
    logger.info("QDRANT_URL set: %s", bool(cfg.get("QDRANT_URL")))
    logger.info("QDRANT_API_KEY set: %s", bool(cfg.get("QDRANT_API_KEY")))
    logger.info("SUPABASE_URL set: %s", bool(cfg.get("SUPABASE_URL")))
    logger.info("SUPABASE_KEY set: %s", bool(cfg.get("SUPABASE_KEY")))

    # Initialize GenAI
    genai = model_client.init_genai_client(cfg.get("GOOGLE_API_KEY"))
    if genai:
        logger.info("GenAI client initialized successfully")
    else:
        logger.warning("GenAI client not initialized - missing key or import failure")

    # Initialize Embeddings
    emb = model_client.init_embeddings(cfg.get("VOYAGE_API_KEY"))
    if emb:
        logger.info("Embeddings client initialized successfully")
    else:
        logger.warning("Embeddings client not initialized - missing key or import failure")

    # Initialize Qdrant
    qc = model_client.init_qdrant_client(cfg.get("QDRANT_URL"), cfg.get("QDRANT_API_KEY"))
    if qc:
        logger.info("Qdrant client initialized successfully")
    else:
        logger.warning("Qdrant client not initialized - missing URL/API key or import failure")

    # Start metadata cleanup background thread (deletes reports after retention)
    try:
        from .utils import start_cleanup_thread
        start_cleanup_thread()
        logger.info("Started metadata cleanup thread (24h retention)")
    except Exception:
        logger.exception("Failed to start cleanup thread")

# --- Router Registration ---

# Mount the Processing Router (e.g., /process/pdf/stream)
app.include_router(process.router, prefix="/process", tags=["Process"])

# Mount the Health Router (e.g., /health/live)
app.include_router(health.router, prefix="/health", tags=["Health"])

# --- FIX: REGISTER THE FILES ROUTER ---
app.include_router(files.router, prefix="/files", tags=["File Management"])

@app.get("/", tags=["root"])
def read_root():
    return {"message": "Welcome to the PDF Extraction Service", "status": "running"}