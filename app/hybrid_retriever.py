import os
from typing import Dict, List

try:
    from .bm25_index import build_bm25_indexes, tokenize
except ImportError:
    from bm25_index import build_bm25_indexes, tokenize


PERSIST_DIR = os.path.abspath("./multi_act_db")


def _normalize_semantic_results(results: Dict) -> Dict[str, List]:
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    if documents and isinstance(documents[0], list):
        documents = documents[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]

    output = {"documents": [], "metadatas": []}
    for index, doc in enumerate(documents):
        meta = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        if index < len(distances):
            try:
                meta["confidence"] = round(max(0.0, min(1.0, 1.0 - float(distances[index]))), 4)
            except (TypeError, ValueError):
                pass
        meta["retrieval_mode"] = "semantic"
        output["documents"].append(doc)
        output["metadatas"].append(meta)
    return output


def semantic_search(query_text: str, collection, k: int = 10) -> Dict[str, List]:
    results = collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return _normalize_semantic_results(results)


def bm25_search(query_text: str, collection_name: str, k: int = 10) -> Dict[str, List]:
    indexes = build_bm25_indexes(PERSIST_DIR)
    payload = indexes.get(collection_name, {})
    index = payload.get("index")
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])

    if not index or not documents:
        return {"documents": [], "metadatas": []}

    scores = index.get_scores(tokenize(query_text))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
    max_score = max((float(score) for _, score in ranked), default=0.0)

    output = {"documents": [], "metadatas": []}
    for position, (doc_index, score) in enumerate(ranked, start=1):
        if score <= 0:
            continue
        meta = dict(metadatas[doc_index] or {})
        meta["bm25_score"] = round(float(score), 4)
        meta["confidence"] = round(float(score) / max_score, 4) if max_score else 0.0
        meta["retrieval_mode"] = "keyword"
        meta["keyword_rank"] = position
        output["documents"].append(documents[doc_index])
        output["metadatas"].append(meta)
    return output


def _key(doc: str, meta: Dict) -> str:
    return "|".join(
        [
            str(meta.get("act", "")),
            str(meta.get("section", meta.get("chunk_index", ""))),
            str(doc)[:180],
        ]
    )


def reciprocal_rank_fusion(bm25_results: Dict[str, List], semantic_results: Dict[str, List], k: int = 60, top_n: int = 10) -> Dict[str, List]:
    fused: Dict[str, Dict] = {}

    for result_set, label in ((bm25_results, "bm25"), (semantic_results, "semantic")):
        for rank, (doc, meta) in enumerate(zip(result_set.get("documents", []), result_set.get("metadatas", [])), start=1):
            result_key = _key(doc, meta)
            if result_key not in fused:
                fused[result_key] = {"document": doc, "metadata": dict(meta), "score": 0.0, "modes": set()}
            fused[result_key]["score"] += 1.0 / (rank + k)
            fused[result_key]["modes"].add(label)
            if "confidence" in meta:
                fused[result_key]["metadata"]["confidence"] = max(
                    float(fused[result_key]["metadata"].get("confidence", 0)),
                    float(meta.get("confidence", 0)),
                )

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_n]
    return {
        "documents": [item["document"] for item in ranked],
        "metadatas": [
            {
                **item["metadata"],
                "rrf_score": round(item["score"], 6),
                "retrieval_mode": "+".join(sorted(item["modes"])),
            }
            for item in ranked
        ],
    }


def retrieve_collection(query_text: str, collection_name: str, collection, n_results: int = 5, mode: str = "Hybrid (BM25 + Semantic)") -> Dict[str, List]:
    semantic = {"documents": [], "metadatas": []}
    keyword = {"documents": [], "metadatas": []}

    if mode in ("Semantic only", "Hybrid (BM25 + Semantic)"):
        semantic = semantic_search(query_text, collection, k=max(n_results, 10))
    if mode in ("Keyword only", "Hybrid (BM25 + Semantic)"):
        keyword = bm25_search(query_text, collection_name, k=max(n_results, 10))

    if mode == "Semantic only":
        return {"documents": semantic["documents"][:n_results], "metadatas": semantic["metadatas"][:n_results]}
    if mode == "Keyword only":
        return {"documents": keyword["documents"][:n_results], "metadatas": keyword["metadatas"][:n_results]}
    return reciprocal_rank_fusion(keyword, semantic, top_n=n_results)
