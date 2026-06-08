import json
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import chromadb
import openai
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from .agent_router import classify_query
    from .bm25_index import tokenize
    from .scenario_processor import analyze_scenario
    from .hybrid_retriever import retrieve_collection
except ImportError:
    from agent_router import classify_query
    from bm25_index import tokenize
    from scenario_processor import analyze_scenario
    from hybrid_retriever import retrieve_collection

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


load_dotenv()

PERSIST_DIR = os.path.abspath("./multi_act_db")
USER_UPLOAD_COLLECTION = "user_uploads"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_resources():
    """Load reusable Chroma, embedding, and OpenAI resources without Streamlit side effects."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    model = SentenceTransformer(DEFAULT_MODEL)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    default_collection = None
    for name in ("bns_sections", "BNS_sections"):
        try:
            default_collection = client.get_collection(name)
            break
        except Exception:
            continue

    return default_collection, model, client


def normalize_section_query(query: str) -> str | None:
    """Extract a section number from a user query."""
    match = re.search(r"(?:section|sec\.?)\s*(\d+[a-zA-Z]?)|^(\d+[a-zA-Z]?)$", query.lower().strip())
    return (match.group(1) or match.group(2)) if match else None


def safe_display_metadata(meta):
    """Handle Chroma metadata values safely."""
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except Exception:
            return {"raw": meta}
    return meta or {}


def extract_section_numbers(text: str) -> set[str]:
    matches = re.findall(r"(?:BNS|IPC|CrPC|CPC|BSA)?\s*Section\s+(\d+[a-zA-Z]?)", text, flags=re.I)
    return set(matches)


def _collection_names(client, include_uploads: bool = True) -> List[str]:
    names = [collection.name for collection in client.list_collections()]
    if not include_uploads:
        names = [name for name in names if name != USER_UPLOAD_COLLECTION]
    return names


def _normalize_results(results: Dict) -> Dict[str, List]:
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    if documents and isinstance(documents[0], list):
        documents = documents[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]

    normalized = {"documents": [], "metadatas": []}
    for index, doc in enumerate(documents):
        meta = safe_display_metadata(metadatas[index] if index < len(metadatas) else {})
        if not isinstance(meta, dict):
            meta = {"raw": str(meta)}
        if index < len(distances):
            try:
                confidence = max(0.0, min(1.0, 1.0 - float(distances[index])))
                meta["confidence"] = round(confidence, 4)
            except (TypeError, ValueError):
                pass
        normalized["documents"].append(doc)
        normalized["metadatas"].append(meta)
    return normalized


def _query_collection_semantic(collection, query_text: str, n_results: int) -> Dict[str, List]:
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return _normalize_results(results)


def _result_key(doc: str, meta: Dict) -> Tuple[str, str, str]:
    return (
        str(meta.get("act", "")),
        str(meta.get("section", meta.get("chunk_index", ""))),
        str(doc)[:160],
    )


def _dedupe_results(documents: Iterable[str], metadatas: Iterable[Dict]) -> Dict[str, List]:
    seen = set()
    output = {"documents": [], "metadatas": []}
    for doc, meta in zip(documents, metadatas):
        meta = safe_display_metadata(meta)
        key = _result_key(doc, meta)
        if key in seen:
            continue
        seen.add(key)
        output["documents"].append(doc)
        output["metadatas"].append(meta)
    return output


def _filter_by_confidence(results: Dict[str, List], threshold: float) -> Dict[str, List]:
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    def passes(meta, active_threshold: float) -> bool:
        value = safe_display_metadata(meta).get("confidence", 1.0)
        if value is None:
            value = 1.0
        try:
            return float(value) >= active_threshold
        except (TypeError, ValueError):
            return True

    pairs = [
        (doc, meta)
        for doc, meta in zip(documents, metadatas)
        if passes(meta, threshold)
    ]
    if len(pairs) < 2 and threshold > 0.3:
        pairs = [
            (doc, meta)
            for doc, meta in zip(documents, metadatas)
            if passes(meta, 0.3)
        ]
    return {
        "documents": [doc for doc, _ in pairs],
        "metadatas": [meta for _, meta in pairs],
    }


def _debug_match(doc: str, meta: Dict, index: int) -> Dict:
    meta = safe_display_metadata(meta)
    return {
        "rank": index,
        "document_name": meta.get("document_name"),
        "section": meta.get("section"),
        "heading": meta.get("heading"),
        "source": meta.get("source"),
        "retrieval_mode": meta.get("retrieval_mode"),
        "confidence": meta.get("confidence"),
        "rrf_score": meta.get("rrf_score"),
        "snippet": str(doc)[:500],
    }


def _print_upload_debug(debug_info: Dict):
    upload_debug = debug_info.get("user_uploads", {})
    print("\n[UPLOAD_DEBUG] include_uploads:", debug_info.get("include_uploads"))
    print("[UPLOAD_DEBUG] retrieval_query:", debug_info.get("retrieval_query"))
    print("[UPLOAD_DEBUG] collection_present:", upload_debug.get("collection_present"))
    print("[UPLOAD_DEBUG] collection_count:", upload_debug.get("collection_count"))
    print("[UPLOAD_DEBUG] raw_match_count:", len(upload_debug.get("raw_matches", [])))
    for match in upload_debug.get("raw_matches", []):
        print(
            "[UPLOAD_DEBUG] RAW",
            f"rank={match.get('rank')}",
            f"doc={match.get('document_name')}",
            f"confidence={match.get('confidence')}",
            f"mode={match.get('retrieval_mode')}",
            f"snippet={match.get('snippet')!r}",
        )
    print("[UPLOAD_DEBUG] kept_after_filter_count:", len(upload_debug.get("kept_matches", [])))
    for match in upload_debug.get("kept_matches", []):
        print(
            "[UPLOAD_DEBUG] KEPT",
            f"rank={match.get('rank')}",
            f"doc={match.get('document_name')}",
            f"confidence={match.get('confidence')}",
            f"mode={match.get('retrieval_mode')}",
            f"snippet={match.get('snippet')!r}",
        )
    print("[UPLOAD_DEBUG] end\n")


def query_all_acts(
    query_text: str,
    n_results: int = 3,
    include_uploads: bool = True,
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
    confidence_threshold: float = 0.45,
    debug_uploads: bool = False,
) -> Dict[str, List]:
    """Query all legal collections, optionally including user uploads."""
    _, _, client = load_resources()
    all_docs: List[str] = []
    all_meta: List[Dict] = []
    debug_info = {
        "include_uploads": include_uploads,
        "retrieval_query": query_text,
        "retrieval_mode": retrieval_mode,
        "confidence_threshold": confidence_threshold,
        "user_uploads": {
            "collection_present": False,
            "collection_count": 0,
            "raw_matches": [],
            "kept_matches": [],
        },
    }

    for collection_name in _collection_names(client, include_uploads=include_uploads):
        try:
            collection = client.get_collection(collection_name)
            collection_count = collection.count()
            if collection_name == USER_UPLOAD_COLLECTION:
                debug_info["user_uploads"]["collection_present"] = True
                debug_info["user_uploads"]["collection_count"] = collection_count
            if collection_count == 0:
                continue
            results = retrieve_collection(
                query_text=query_text,
                collection_name=collection_name,
                collection=collection,
                n_results=n_results,
                mode=retrieval_mode,
            )
            for meta in results.get("metadatas", []):
                if collection_name == USER_UPLOAD_COLLECTION:
                    meta["source"] = "user_upload"
                    meta.setdefault("act", "Uploaded Document")
                else:
                    meta.setdefault("source", "preloaded_act")
            all_docs.extend(results.get("documents", []))
            all_meta.extend(results.get("metadatas", []))
            if collection_name == USER_UPLOAD_COLLECTION:
                debug_info["user_uploads"]["raw_matches"] = [
                    _debug_match(doc, meta, index)
                    for index, (doc, meta) in enumerate(
                        zip(results.get("documents", []), results.get("metadatas", [])),
                        start=1,
                    )
                ]
        except Exception as exc:
            print(f"Error querying {collection_name}: {str(exc)}")
            if collection_name == USER_UPLOAD_COLLECTION:
                debug_info["user_uploads"]["error"] = str(exc)

    deduped = _dedupe_results(all_docs, all_meta)
    filtered = _filter_by_confidence(deduped, confidence_threshold)
    debug_info["user_uploads"]["kept_matches"] = [
        _debug_match(doc, meta, index)
        for index, (doc, meta) in enumerate(zip(filtered["documents"], filtered["metadatas"]), start=1)
        if safe_display_metadata(meta).get("source") == "user_upload"
    ]
    ranked = sorted(
        zip(filtered["documents"], filtered["metadatas"]),
        key=lambda pair: float(pair[1].get("rrf_score") or pair[1].get("confidence") or 0),
        reverse=True,
    )
    if debug_uploads:
        _print_upload_debug(debug_info)
    return {
        "documents": [doc for doc, _ in ranked],
        "metadatas": [meta for _, meta in ranked],
        "debug": debug_info,
    }


def search_section_all_acts(section_number: str, include_uploads: bool = True) -> Dict[str, List]:
    """Search for an exact section number across all legal acts."""
    _, _, client = load_resources()
    documents: List[str] = []
    metadatas: List[Dict] = []

    for collection_name in _collection_names(client, include_uploads=include_uploads):
        try:
            collection = client.get_collection(collection_name)
            results = collection.get(
                where={"section": section_number},
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
                meta = safe_display_metadata(meta)
                meta.setdefault("confidence", 1.0)
                meta.setdefault("source", "user_upload" if collection_name == USER_UPLOAD_COLLECTION else "preloaded_act")
                if collection_name == USER_UPLOAD_COLLECTION:
                    meta.setdefault("act", "Uploaded Document")
                documents.append(doc)
                metadatas.append(meta)
        except Exception as exc:
            print(f"Error searching {collection_name}: {str(exc)}")

    return _dedupe_results(documents, metadatas)


def generate_context(results: Dict[str, List], numbered: bool = False):
    """Convert retrieved documents into model context and an optional source map."""
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return ("No matching sections found", {}) if numbered else "No matching sections found"

    lines = []
    source_map = {}
    for index, (doc, raw_meta) in enumerate(zip(documents, metadatas), start=1):
        meta = safe_display_metadata(raw_meta)
        act = meta.get("act", "UNKNOWN")
        section = meta.get("section_display") or meta.get("section") or meta.get("document_name", "Document")
        heading = meta.get("heading", "")
        label = f"{act} {section}".strip()
        if heading:
            label = f"{label}: {heading}"
        if numbered:
            lines.append(f"[{index}] {label} - {doc}")
            source_map[index] = {
                "act": act,
                "section": str(meta.get("section", section)),
                "heading": heading,
                "snippet": str(doc)[:200],
                "confidence": meta.get("confidence"),
                "source": meta.get("source", "preloaded_act"),
                "document_name": meta.get("document_name"),
            }
        else:
            lines.append(str(doc))

    context = "\n\n".join(lines)
    return (context, source_map) if numbered else context


def generate_response(prompt: str, context: str | Dict[str, List]):
    """Generate a legal answer and a numbered source map."""
    if isinstance(context, dict):
        context_text, source_map = generate_context(context, numbered=True)
    else:
        context_text = context
        source_map = {}

    try:
        messages = [
            {
                "role": "system",
                "content": f"""
You are a multi-act legal expert. Analyze using provisions from BNS, IPC, CrPC, CPC, BSA, and uploaded user documents when present.
You MUST cite every factual claim using [N] notation where N is the source number provided.
Always cite before the period.
If the retrieved context is weak or missing, say so clearly and avoid inventing law.

Structure responses as:
1. Applicable Sections
2. Key Elements
3. Potential Defenses or Caveats

Context:
{context_text}
""",
            },
            {"role": "user", "content": prompt},
        ]
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125"),
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content, source_map
    except Exception as exc:
        return f"Error generating response: {str(exc)}", source_map


def build_sections(results: Dict[str, List], analysis_text: str = "") -> List[Dict]:
    """Create UI-ready section summaries from retrieval metadata."""
    sections = []
    seen = set()
    mentioned_sections = extract_section_numbers(analysis_text)

    for meta in results.get("metadatas", []):
        meta = safe_display_metadata(meta)
        if not isinstance(meta, dict):
            continue
        section_num = str(meta.get("section", ""))
        if mentioned_sections and section_num and section_num not in mentioned_sections:
            continue

        item = {
            "section": section_num or str(meta.get("document_name", "Document")),
            "display": meta.get("section_display", f"Section {section_num}" if section_num else meta.get("document_name", "Document")),
            "heading": meta.get("heading", ""),
            "act": meta.get("act", "UNKNOWN"),
            "confidence": meta.get("confidence"),
            "source": meta.get("source", "preloaded_act"),
        }
        key = (item["act"], item["section"], item["heading"])
        if key in seen:
            continue
        seen.add(key)
        sections.append(item)
    return sections


def _normalize_doc_results(documents: List[str], metadatas: List[Dict], scores: List[float] | None = None, mode: str = "semantic") -> Dict[str, List]:
    output = {"documents": [], "metadatas": []}
    max_score = max(scores or [0]) if scores else 0
    for index, doc in enumerate(documents):
        meta = safe_display_metadata(metadatas[index] if index < len(metadatas) else {})
        meta = dict(meta or {})
        if scores:
            score = scores[index]
            meta["confidence"] = round(float(score) / max_score, 4) if max_score else 0.0
        meta["source"] = "user_upload"
        meta.setdefault("act", "Uploaded Document")
        meta["retrieval_mode"] = mode
        output["documents"].append(doc)
        output["metadatas"].append(meta)
    return output


def _semantic_active_document_search(collection, query: str, document_name: str | None, n_results: int) -> Dict[str, List]:
    _, model, _ = load_resources()
    where = {"document_name": document_name} if document_name else None
    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(n_results, 1), max(collection.count(), 1)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    normalized = _normalize_results(results)
    for meta in normalized["metadatas"]:
        meta["source"] = "user_upload"
        meta.setdefault("act", "Uploaded Document")
        meta["retrieval_mode"] = "semantic"
    return normalized


def _keyword_active_document_search(collection, query: str, document_name: str | None, n_results: int) -> Dict[str, List]:
    if BM25Okapi is None:
        return {"documents": [], "metadatas": []}

    where = {"document_name": document_name} if document_name else None
    data = collection.get(where=where, include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    if not documents:
        return {"documents": [], "metadatas": []}

    index = BM25Okapi([tokenize(doc) for doc in documents])
    scores = index.get_scores(tokenize(query))
    ranked = [(doc_index, float(score)) for doc_index, score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True) if score > 0]
    ranked = ranked[:n_results]
    return _normalize_doc_results(
        [documents[index] for index, _ in ranked],
        [metadatas[index] for index, _ in ranked],
        [score for _, score in ranked],
        mode="keyword",
    )


def _rrf_document_results(keyword_results: Dict[str, List], semantic_results: Dict[str, List], top_n: int = 5) -> Dict[str, List]:
    fused = {}
    for result_set, mode in ((keyword_results, "keyword"), (semantic_results, "semantic")):
        for rank, (doc, meta) in enumerate(zip(result_set.get("documents", []), result_set.get("metadatas", [])), start=1):
            key = _result_key(doc, meta)
            if key not in fused:
                fused[key] = {"document": doc, "metadata": dict(meta), "score": 0.0, "modes": set()}
            fused[key]["score"] += 1.0 / (rank + 60)
            fused[key]["modes"].add(mode)
            if meta.get("confidence") is not None:
                fused[key]["metadata"]["confidence"] = max(
                    float(fused[key]["metadata"].get("confidence") or 0),
                    float(meta.get("confidence") or 0),
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


def _query_active_document_chunks(
    query: str,
    document_name: str | None = None,
    n_results: int = 5,
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
) -> Dict[str, List]:
    """Retrieve chunks from the active uploaded document only."""
    _, _, client = load_resources()
    try:
        collection = client.get_collection(USER_UPLOAD_COLLECTION)
    except Exception:
        return {"documents": [], "metadatas": []}

    try:
        if retrieval_mode == "Semantic only":
            return _semantic_active_document_search(collection, query, document_name, n_results)
        if retrieval_mode == "Keyword only":
            return _keyword_active_document_search(collection, query, document_name, n_results)
        semantic = _semantic_active_document_search(collection, query, document_name, max(n_results, 10))
        keyword = _keyword_active_document_search(collection, query, document_name, max(n_results, 10))
        return _rrf_document_results(keyword, semantic, top_n=n_results)
    except Exception as exc:
        print(f"Document-focused upload retrieval failed: {str(exc)}")
        return {"documents": [], "metadatas": []}


def _format_document_focused_context(document_results: Dict[str, List], law_results: Dict[str, List]) -> tuple[str, Dict]:
    """Build [DOC-N] and [LAW-N] context blocks for document-focused Q&A."""
    lines = []
    source_map = {}

    for index, (doc, meta) in enumerate(zip(document_results.get("documents", []), document_results.get("metadatas", [])), start=1):
        meta = safe_display_metadata(meta)
        key = f"DOC-{index}"
        label = meta.get("document_name") or "Uploaded document"
        lines.append(f"[{key}] {label} chunk {meta.get('chunk_index', index)} - {doc}")
        source_map[key] = {
            "act": "Uploaded Document",
            "section": str(meta.get("section", key)),
            "heading": meta.get("heading", label),
            "snippet": str(doc)[:200],
            "confidence": meta.get("confidence"),
            "source": "user_upload",
            "document_name": meta.get("document_name"),
        }

    for index, (doc, meta) in enumerate(zip(law_results.get("documents", []), law_results.get("metadatas", [])), start=1):
        meta = safe_display_metadata(meta)
        key = f"LAW-{index}"
        act = meta.get("act", "UNKNOWN")
        section = meta.get("section_display") or meta.get("section") or "Provision"
        heading = meta.get("heading", "")
        label = f"{act} {section}".strip()
        if heading:
            label = f"{label}: {heading}"
        lines.append(f"[{key}] {label} - {doc}")
        source_map[key] = {
            "act": act,
            "section": str(meta.get("section", section)),
            "heading": heading,
            "snippet": str(doc)[:200],
            "confidence": meta.get("confidence"),
            "source": "legal_act",
            "document_name": None,
        }

    return "\n\n".join(lines), source_map


def _safe_json_object(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                pass
    return {}


def _extract_legal_points_from_document_chunks(query: str, document_results: Dict[str, List]) -> Dict:
    """Extract issues, acts, and law-search queries from retrieved document chunks."""
    doc_context_lines = []
    for index, doc in enumerate(document_results.get("documents", []), start=1):
        doc_context_lines.append(f"[DOC-{index}] {doc}")
    doc_context = "\n\n".join(doc_context_lines)

    fallback_query = " ".join([query, " ".join(str(doc)[:300] for doc in document_results.get("documents", [])[:3])]).strip()
    fallback = {
        "issues": [],
        "acts": ["bns", "ipc", "crpc", "cpc", "bsa"],
        "search_queries": [fallback_query or query],
    }
    if not doc_context:
        return fallback

    prompt = f"""
The user is asking about an uploaded legal document. Use only the retrieved document chunks below to extract structured legal points.
Return only JSON with this schema:
{{
  "issues": [
    {{
      "issue": "short legal issue name",
      "facts": "facts from the document chunks",
      "acts": ["bns", "ipc", "crpc", "cpc", "bsa"],
      "potential_sections": ["..."],
      "search_query": "query to retrieve relevant Indian legal provisions"
    }}
  ],
  "acts": ["bns", "ipc", "crpc", "cpc", "bsa"],
  "search_queries": ["..."]
}}

User query:
{query}

Retrieved document chunks:
{doc_context}
"""
    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125"),
            messages=[
                {"role": "system", "content": "You extract legal retrieval points from uploaded Indian legal documents. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        payload = _safe_json_object(response.choices[0].message.content)
    except Exception as exc:
        print(f"Document legal point extraction failed: {str(exc)}")
        return fallback

    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    acts = [str(act).lower() for act in payload.get("acts", []) if str(act).lower() in {"bns", "ipc", "crpc", "cpc", "bsa"}]
    search_queries = [str(item).strip() for item in payload.get("search_queries", []) if str(item).strip()]

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        acts.extend(str(act).lower() for act in issue.get("acts", []) if str(act).lower() in {"bns", "ipc", "crpc", "cpc", "bsa"})
        if issue.get("search_query"):
            search_queries.append(str(issue["search_query"]).strip())
        else:
            search_query = " ".join(
                str(part)
                for part in [
                    issue.get("issue", ""),
                    issue.get("facts", ""),
                    " ".join(str(section) for section in issue.get("potential_sections", [])),
                ]
                if part
            ).strip()
            if search_query:
                search_queries.append(search_query)

    deduped_queries = []
    seen_queries = set()
    for search_query in search_queries or fallback["search_queries"]:
        key = search_query.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        deduped_queries.append(search_query)

    return {
        "issues": [issue for issue in issues if isinstance(issue, dict)],
        "acts": sorted(set(acts)) or fallback["acts"],
        "search_queries": deduped_queries[:8],
    }


def _collection_names_for_acts(client, acts: Iterable[str]) -> List[str]:
    wanted = {str(act).lower() for act in acts if str(act).lower() in {"bns", "ipc", "crpc", "cpc", "bsa"}}
    names = []
    for collection in client.list_collections():
        collection_name = collection.name
        lowered = collection_name.lower()
        if collection_name == USER_UPLOAD_COLLECTION:
            continue
        if not wanted or any(lowered.startswith(act) for act in wanted):
            names.append(collection_name)
    return names


def _query_law_for_document_points(
    legal_points: Dict,
    retrieval_mode: str,
    confidence_threshold: float,
    n_results: int = 4,
) -> Dict[str, List]:
    """Query legal act collections using extracted document legal points."""
    _, _, client = load_resources()
    all_docs: List[str] = []
    all_meta: List[Dict] = []
    collection_names = _collection_names_for_acts(client, legal_points.get("acts", []))

    for search_query in legal_points.get("search_queries", []):
        for collection_name in collection_names:
            try:
                collection = client.get_collection(collection_name)
                if collection.count() == 0:
                    continue
                results = retrieve_collection(
                    query_text=search_query,
                    collection_name=collection_name,
                    collection=collection,
                    n_results=n_results,
                    mode=retrieval_mode,
                )
                for meta in results.get("metadatas", []):
                    meta.setdefault("source", "legal_act")
                    meta["document_search_query"] = search_query
                all_docs.extend(results.get("documents", []))
                all_meta.extend(results.get("metadatas", []))
            except Exception as exc:
                print(f"Document law query failed for {collection_name}: {str(exc)}")

    deduped = _dedupe_results(all_docs, all_meta)
    filtered = _filter_by_confidence(deduped, confidence_threshold)
    ranked = sorted(
        zip(filtered["documents"], filtered["metadatas"]),
        key=lambda pair: float(pair[1].get("rrf_score") or pair[1].get("confidence") or 0),
        reverse=True,
    )
    return {
        "documents": [doc for doc, _ in ranked[:12]],
        "metadatas": [meta for _, meta in ranked[:12]],
    }


def run_document_focused_analysis(
    query: str,
    document_name: str | None = None,
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
    confidence_threshold: float = 0.45,
) -> Dict:
    """Answer follow-up questions where the uploaded document is the subject of analysis."""
    document_results = _query_active_document_chunks(
        query,
        document_name=document_name,
        n_results=5,
        retrieval_mode=retrieval_mode,
    )
    legal_points = _extract_legal_points_from_document_chunks(query, document_results)
    law_results = _query_law_for_document_points(
        legal_points=legal_points,
        retrieval_mode=retrieval_mode,
        confidence_threshold=confidence_threshold,
        n_results=4,
    )
    context_text, source_map = _format_document_focused_context(document_results, law_results)

    messages = [
        {
            "role": "system",
            "content": f"""
You are analyzing an uploaded legal document through Indian legal provisions.
The uploaded document is the subject of analysis.
The legal acts are reference knowledge only.

Use [DOC-N] citations for factual claims about the uploaded document.
Use [LAW-N] citations for legal/statutory claims.
Always cite before the period.
If document context is missing, say that the active uploaded document was not retrieved.
If legal provisions are weak or missing, say that the legal act retrieval was inconclusive.

Structured legal points extracted from the document:
{json.dumps(legal_points, indent=2)}

Context:
{context_text or "No document or law context retrieved."}
""",
        },
        {"role": "user", "content": query},
    ]

    try:
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125"),
            messages=messages,
            temperature=0.1,
        )
        analysis = response.choices[0].message.content
    except Exception as exc:
        analysis = f"Error generating document-focused response: {str(exc)}"

    combined_results = {
        "documents": document_results.get("documents", []) + law_results.get("documents", []),
        "metadatas": document_results.get("metadatas", []) + law_results.get("metadatas", []),
    }
    return {
        "analysis": analysis,
        "source_map": source_map,
        "sections": build_sections(law_results, analysis),
        "context_preview": [str(doc)[:200] + "..." for doc in combined_results["documents"][:4]],
        "results": combined_results,
        "debug": {
            "document_focused": True,
            "active_document": document_name,
            "document_chunks": len(document_results.get("documents", [])),
            "law_chunks": len(law_results.get("documents", [])),
            "legal_points": legal_points,
        },
    }


def run_legal_analysis(
    query: str,
    include_uploads: bool = True,
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
    confidence_threshold: float = 0.45,
    debug_uploads: bool = False,
) -> Dict:
    """End-to-end RAG pipeline used by Streamlit UIs."""
    section_num = normalize_section_query(query)
    if section_num:
        results = search_section_all_acts(section_num, include_uploads=include_uploads)
        if not results["documents"]:
            results = query_all_acts(
                query,
                n_results=3,
                include_uploads=include_uploads,
                retrieval_mode=retrieval_mode,
                confidence_threshold=confidence_threshold,
                debug_uploads=debug_uploads,
            )
    elif classify_query(query) == "direct":
        results = query_all_acts(
            query,
            n_results=5,
            include_uploads=include_uploads,
            retrieval_mode=retrieval_mode,
            confidence_threshold=confidence_threshold,
            debug_uploads=debug_uploads,
        )
    else:
        scenario_data = analyze_scenario(query)
        if scenario_data:
            offenses = [scenario_data.get("primary_offense", query)] + scenario_data.get("related_offenses", [])
            retrieval_query = f"{query} {' '.join(offenses)}"
        else:
            retrieval_query = query
        results = query_all_acts(
            retrieval_query,
            n_results=4,
            include_uploads=include_uploads,
            retrieval_mode=retrieval_mode,
            confidence_threshold=confidence_threshold,
            debug_uploads=debug_uploads,
        )

    analysis, source_map = generate_response(query, results)
    return {
        "analysis": analysis,
        "source_map": source_map,
        "sections": build_sections(results, analysis),
        "context_preview": [str(doc)[:200] + "..." for doc in results.get("documents", [])[:3]],
        "results": results,
        "debug": results.get("debug", {}),
    }
