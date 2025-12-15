"""
Model client and embedding factories.
Read API keys from environment variables.
"""
import os
from typing import Optional
from ..core import config as core_config

genai_client = None
embeddings = None
qdrant_client = None


def init_genai_client(api_key: Optional[str] = None):
    global genai_client
    try:
        from google import genai
        if api_key is None:
            cfg = core_config.load_config()
            api_key = cfg.get("GOOGLE_API_KEY")
        genai_client = genai.Client(api_key=api_key) if api_key else None
    except Exception:
        genai_client = None
    return genai_client


def init_embeddings(voyage_api_key: Optional[str] = None):
    global embeddings
    try:
        from langchain_voyageai import VoyageAIEmbeddings
        if voyage_api_key is None:
            cfg = core_config.load_config()
            voyage_api_key = cfg.get("VOYAGE_API_KEY")
        if voyage_api_key:
            os.environ.setdefault("VOYAGE_API_KEY", voyage_api_key)
            embeddings = VoyageAIEmbeddings(model="voyage-3-large")
            return embeddings
    except Exception:
        pass
    return None


def init_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    global qdrant_client
    try:
        from qdrant_client import QdrantClient
        if url is None or api_key is None:
            cfg = core_config.load_config()
            if url is None:
                url = cfg.get("QDRANT_URL")
            if api_key is None:
                api_key = cfg.get("QDRANT_API_KEY")
        if url:
            qdrant_client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
            return qdrant_client
    except Exception:
        qdrant_client = None
    return None


class ModelClient:
    """Simple wrapper that exposes current clients as properties.

    The module keeps module-level references (genai_client, embeddings, qdrant_client)
    and this wrapper exposes them dynamically so other modules can import
    `model_client` and access attributes like `model_client.genai_client`.
    """

    @property
    def genai_client(self):
        return genai_client

    @property
    def embeddings(self):
        return embeddings

    @property
    def qdrant_client(self):
        return qdrant_client

    def init_all(self):
        init_genai_client()
        init_embeddings()
        init_qdrant_client()


model_client = ModelClient()
