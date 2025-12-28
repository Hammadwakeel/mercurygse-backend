import re
import time
import random
import logging
from typing import List, Dict, Optional, Callable
import os
from .model_client import init_qdrant_client, init_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# chunking regex
PAGE_SPLIT_RE = re.compile(r'(?m)^(##\s+Page\s+\d+.*)$')

# Cache for vector size to avoid repeated API calls
# Voyage-3-large is 1024 dimensions.
_VECTOR_SIZE_CACHE = None

def chunk_markdown_by_page(md_path: str) -> List[Dict]:
    """
    Splits markdown into page-based chunks. 
    Falls back to recursive character splitting if page headers are missing.
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        md = f.read()
    
    parts = PAGE_SPLIT_RE.split(md)
    
    # FALLBACK LOGIC:
    # If regex split results in too few parts (likely failed to match headers)
    # and the document is large enough to warrant splitting.
    if len(parts) < 2 and len(md) > 500:
        logger.warning(f"Regex chunking failed for {os.path.basename(md_path)} (no '## Page X' found). Using recursive fallback.")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = splitter.split_text(md)
        
        # Return fallback chunks
        return [{
            'id': f'chunk_{i}', 
            'page': 1, # Dummy page number since we lost structure
            'page_type': 'fallback', 
            'text': t, 
            'char_length': len(t)
        } for i, t in enumerate(texts)]

    # STANDARD LOGIC (Regex successful)
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
    """

    qc = init_qdrant_client()
    emb = init_embeddings()
    if qc is None or emb is None:
        return {'error': 'qdrant-or-embeddings-missing'}

    # Determine slowdown between embedding calls to respect provider rate limits.
    # Env overrides: VOYAGE_EMBED_SLEEP_SECONDS or QDRANT_EMBED_SLEEP_SECONDS
    slow_seconds = None
    try:
        env_s = os.environ.get('VOYAGE_EMBED_SLEEP_SECONDS') or os.environ.get('QDRANT_EMBED_SLEEP_SECONDS')
        if env_s:
            slow_seconds = float(env_s)
        else:
            # If embeddings backend looks like Voyage, default to 21 seconds (<= 3 RPM)
            emb_mod = getattr(emb, '__module__', '') or repr(emb)
            if 'voyage' in emb_mod.lower() or 'voyage' in repr(emb).lower():
                slow_seconds = 21.0
    except Exception:
        slow_seconds = None
    if slow_seconds:
        logger.info('Embedding slowdown enabled: sleeping %.2fs between embedding batches', slow_seconds)

    try:
        # lazy import heavy libs
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.documents import Document
        from qdrant_client.models import VectorParams, Distance

        # PERFORMANCE FIX: Use Cached Vector Size
        global _VECTOR_SIZE_CACHE
        if _VECTOR_SIZE_CACHE is None:
            # Check if we can determine it from the embedding object or default to 1024 (Voyage)
            try:
                # Attempt a single call to set it dynamically if needed, but only once per app lifecycle
                logger.info("Determining embedding vector size...")
                sample_vec = emb.embed_query('sample size')
                _VECTOR_SIZE_CACHE = len(sample_vec)
                logger.info(f"Vector size set to {_VECTOR_SIZE_CACHE}")
            except Exception:
                # Default fallback if call fails
                logger.warning("Failed to determine vector size dynamically, defaulting to 1024")
                _VECTOR_SIZE_CACHE = 1024
        
        vector_size = _VECTOR_SIZE_CACHE

        # create collection if not exists
        try:
            existing = [c.name for c in qc.get_collections().collections]
        except Exception:
            existing = []
        
        if collection_name not in existing and vector_size is not None:
            qc.create_collection(
                collection_name=collection_name, 
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

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

        ingested = 0
        # process in batches
        for i in range(0, total_docs, batch_size):
            batch_docs = docs_all[i:i+batch_size]

            # -------------------------------------------------------------
            # FIX: Sleep BEFORE the embedding call to respect Voyage limits
            # -------------------------------------------------------------
            if slow_seconds:
                try:
                    logger.debug('Sleeping %.2fs before embedding batch %s', slow_seconds, i // batch_size)
                    time.sleep(slow_seconds)
                except Exception:
                    pass
            
            success = False
            last_err = None
            
            for attempt in range(1, retry_attempts + 1):
                try:
                    store.add_documents(batch_docs)
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    # Special handling for Rate Limits (429) -> wait longer
                    if 'rate limit' in str(e).lower() or '429' in str(e):
                        time.sleep(30)
                    elif attempt < retry_attempts:
                        time.sleep(0.4 * attempt + random.uniform(0, 0.2))
                    continue

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
        # Detect common billing/permission messages from Voyage / embedding providers
        s = str(e).lower()
        if 'payment method' in s or 'add your payment method' in s or 'billing' in s or 'reduced rate limits' in s or 'rate limits' in s:
            return {'error': 'voyage_billing', 'message': str(e)}
        return {'error': str(e)}