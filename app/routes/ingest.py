from fastapi import APIRouter, HTTPException
import os
from typing import Optional
from ..services.qdrant_service import chunk_markdown_by_page, ingest_chunks_into_qdrant

router = APIRouter()


@router.post("/")
async def ingest(report_path: str, collection: Optional[str] = "mercurygse"):
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail='Report not found')
    chunks = chunk_markdown_by_page(report_path)
    res = ingest_chunks_into_qdrant(chunks, collection_name=collection)
    return {"chunks": len(chunks), "ingest_result": res}
