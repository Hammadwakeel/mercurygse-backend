import re
import time
import random
from typing import List, Dict, Optional, Callable
import os
from .model_client import init_qdrant_client, init_embeddings

# chunking regex
PAGE_SPLIT_RE = re.compile(r'(?m)^(##\s+Page\s+\d+.*)$')


def chunk_markdown_by_page(md_path: str) -> List[Dict]:
    with open(md_path, 'r', encoding='utf-8') as f:
        md = f.read()
    parts = PAGE_SPLIT_RE.split(md)
    chunks = []
    preamble = parts[0].strip()
    if preamble:
        chunks.append({
            'id': 'page_0', 'page': 0, 'page_type': None, 'text': preamble, 'char_length': len(preamble)
        })
    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ''
        m = re.search(r'Page\s+(\d+)', header)
        page_num = int(m.group(1)) if m else None
        page_type = None
        if 'SIMPLE' in header.upper():
            page_type = 'SIMPLE'
        elif 'COMPLEX' in header.upper():
            page_type = 'COMPLEX'
        full_text = f"{header}\n\n{body}".strip()
        chunks.append({'id': f'page_{page_num}', 'page': page_num, 'page_type': page_type, 'text': full_text, 'char_length': len(full_text)})
        i += 2
    return chunks


def ingest_chunks_into_qdrant(
    chunks: List[Dict],
    collection_name: str = 'manual_pages',
    batch_size: int = 256,
    progress_hook: Optional[Callable[[dict], None]] = None,
    retry_attempts: int = 3,
) -> Dict:
    """Ingest chunks into Qdrant using Voyage embeddings and langchain vector store.
    Implements chunked/batched upserts. Calls `progress_hook` after each batch when provided.
    This function performs real ingestion and does not support dry-run.
    """

    qc = init_qdrant_client()
    emb = init_embeddings()
    if qc is None or emb is None:
        return {'error': 'qdrant-or-embeddings-missing'}

    try:
        # lazy import heavy libs
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.documents import Document

        # compute vector size by embedding a small sample
        try:
            sample_vec = emb.embed_query('sample size')
            vector_size = len(sample_vec)
        except Exception:
            vector_size = None

        # create collection if not exists
        try:
            existing = [c.name for c in qc.get_collections().collections]
        except Exception:
            existing = []
        if collection_name not in existing and vector_size is not None:
            from qdrant_client.models import VectorParams, Distance
            qc.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

        # build documents (skip page 0 preamble)
        docs_all = []
        for c in chunks:
            if c.get('page') in (None, 0):
                continue
            docs_all.append(Document(page_content=c['text'], metadata={'chunk_id': c['id'], 'page': c['page'], 'page_type': c.get('page_type'), 'char_length': c.get('char_length')}))

        total_docs = len(docs_all)
        if total_docs == 0:
            return {'ingested': 0, 'collection': collection_name}

        store = QdrantVectorStore(client=qc, collection_name=collection_name, embedding=emb)

        # helper to get embeddings for a list of texts with fallback
        def embed_texts(texts: List[str]) -> List[List[float]]:
            # try common batch method names used by embedding wrappers
            if hasattr(emb, 'embed_documents'):
                return emb.embed_documents(texts)
            if hasattr(emb, 'embed_texts'):
                return emb.embed_texts(texts)
            # fallback to per-item embedding
            return [emb.embed_query(t) for t in texts]

        ingested = 0
        # process in batches
        for i in range(0, total_docs, batch_size):
            batch_docs = docs_all[i:i+batch_size]
            texts = [d.page_content for d in batch_docs]

            # get embeddings with retries
            last_err = None
            for attempt in range(1, retry_attempts + 1):
                try:
                    vectors = embed_texts(texts)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < retry_attempts:
                        time.sleep(0.5 * attempt + random.uniform(0, 0.2))
                        continue
                    raise

            # attach vectors to documents via metadata (QdrantVectorStore will compute embeddings again if not provided),
            # but many vector stores accept raw embeddings; we can use store.client upsert directly if needed.
            # We'll attempt store.add_documents(batch) and fall back to per-doc add if necessary.
            success = False
            for attempt in range(1, retry_attempts + 1):
                try:
                    # The high-level API will call embedding again unless we upsert directly; it's acceptable for now.
                    store.add_documents(batch_docs)
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt < retry_attempts:
                        time.sleep(0.4 * attempt + random.uniform(0, 0.2))
                        continue
                    raise

            if not success:
                raise RuntimeError(f"Failed to ingest batch starting at {i}: {last_err}")

            ingested += len(batch_docs)
            # emit progress
            if progress_hook:
                progress_hook({
                    'event': 'ingest_batch',
                    'batch_index': i // batch_size,
                    'batch_size': len(batch_docs),
                    'total_docs': total_docs,
                    'ingested_so_far': ingested,
                })

        return {'ingested': ingested, 'collection': collection_name}
    except Exception as e:
        return {'error': str(e)}
