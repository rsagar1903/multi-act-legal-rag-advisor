import re
from functools import lru_cache
from typing import Dict, List

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _build_index(client, collection_name: str) -> Dict:
    collection = client.get_collection(collection_name)
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    corpus = [tokenize(doc) for doc in documents]
    index = BM25Okapi(corpus) if BM25Okapi and corpus else None
    return {
        "index": index,
        "documents": documents,
        "metadatas": metadatas,
    }


def _build_all_indexes_uncached(db_path: str) -> Dict[str, Dict]:
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    indexes = {}
    for collection in client.list_collections():
        collection_name = getattr(collection, "name", collection)
        try:
            indexes[collection_name] = _build_index(client, collection_name)
        except Exception as exc:
            print(f"BM25 index build failed for {collection_name}: {str(exc)}")
    return indexes


@lru_cache(maxsize=4)
def _build_all_indexes_cached(db_path: str) -> Dict[str, Dict]:
    return _build_all_indexes_uncached(db_path)


_streamlit_cached_builder = None


def build_bm25_indexes(db_path: str) -> Dict[str, Dict]:
    """Build BM25 indexes, using Streamlit resource caching when available."""
    global _streamlit_cached_builder
    try:
        import streamlit as st

        if _streamlit_cached_builder is None:
            _streamlit_cached_builder = st.cache_resource(show_spinner=False)(_build_all_indexes_uncached)
        return _streamlit_cached_builder(db_path)
    except Exception:
        return _build_all_indexes_cached(db_path)


def clear_bm25_cache():
    _build_all_indexes_cached.cache_clear()
    if _streamlit_cached_builder is not None and hasattr(_streamlit_cached_builder, "clear"):
        _streamlit_cached_builder.clear()
