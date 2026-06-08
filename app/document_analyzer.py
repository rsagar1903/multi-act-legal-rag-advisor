import json
import os
from typing import Dict, Iterable, List

import chromadb
import openai
from chromadb.config import Settings
from dotenv import load_dotenv

try:
    from .hybrid_retriever import retrieve_collection
except ImportError:
    from hybrid_retriever import retrieve_collection


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PERSIST_DIR = os.path.abspath("./multi_act_db")
USER_UPLOAD_COLLECTION = "user_uploads"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125")
LEGAL_ACTS = {"bns", "ipc", "crpc", "cpc", "bsa"}
SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def _truncate(text: str, max_chars: int = 18000) -> str:
    text = text or ""
    return text[:max_chars]


def _safe_json_loads(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _normalize_analysis(payload: Dict) -> Dict:
    issues = payload.get("legal_issues") or payload.get("issues") or []
    if not isinstance(issues, list):
        issues = []

    normalized_issues = []
    for issue in issues:
        if isinstance(issue, str):
            issue = {"issue": issue}
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity", "Medium")).title()
        if severity not in SEVERITY_ORDER:
            severity = "Medium"
        normalized_issues.append(
            {
                "issue": issue.get("issue") or issue.get("name") or "Unspecified legal issue",
                "facts": issue.get("facts") or issue.get("supporting_facts") or "",
                "acts": [str(act).lower() for act in issue.get("acts", []) if str(act).lower() in LEGAL_ACTS],
                "potential_sections": [str(section) for section in issue.get("potential_sections", [])],
                "severity": severity,
            }
        )

    acts = [str(act).lower() for act in payload.get("applicable_acts", []) if str(act).lower() in LEGAL_ACTS]
    for issue in normalized_issues:
        acts.extend(issue.get("acts", []))
    acts = sorted(set(acts)) or ["bns", "ipc", "crpc", "bsa"]

    highest_severity = "Low"
    for issue in normalized_issues:
        if SEVERITY_ORDER[issue["severity"]] > SEVERITY_ORDER[highest_severity]:
            highest_severity = issue["severity"]

    return {
        "document_type": payload.get("document_type", "Unknown legal document"),
        "parties": payload.get("parties", []),
        "brief_summary": payload.get("brief_summary", payload.get("summary", "")),
        "key_facts": payload.get("key_facts", []),
        "legal_issues": normalized_issues,
        "applicable_acts": acts,
        "potential_sections": [str(section) for section in payload.get("potential_sections", [])],
        "severity": payload.get("severity") or highest_severity,
    }


def extract_legal_issues_from_document(text: str) -> Dict:
    """Stage 1: extract legal issues, acts, facts, and severity from the uploaded document."""
    prompt = f"""
Analyze the uploaded legal document as the subject of legal analysis.
Return only JSON with this schema:
{{
  "document_type": "FIR | contract | bail application | court order | notice | other",
  "parties": ["..."],
  "brief_summary": "...",
  "key_facts": ["..."],
  "legal_issues": [
    {{
      "issue": "grievous hurt | fraudulent misrepresentation | breach of contract | ...",
      "facts": "specific facts from the document supporting this issue",
      "acts": ["bns", "ipc", "crpc", "cpc", "bsa"],
      "potential_sections": ["..."],
      "severity": "Low | Medium | High | Critical"
    }}
  ],
  "applicable_acts": ["bns", "ipc", "crpc", "cpc", "bsa"],
  "potential_sections": ["..."],
  "severity": "Low | Medium | High | Critical"
}}

Document:
{_truncate(text)}
"""
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful Indian legal document analyst. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return _normalize_analysis(_safe_json_loads(response.choices[0].message.content))


def _client():
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_names_for_acts(client, acts: Iterable[str]) -> List[str]:
    requested = {str(act).lower() for act in acts if str(act).lower() in LEGAL_ACTS}
    names = []
    for collection in client.list_collections():
        name = collection.name
        lowered = name.lower()
        if name == USER_UPLOAD_COLLECTION:
            continue
        if not requested or any(lowered.startswith(act) for act in requested):
            names.append(name)
    return names


def _provision_key(doc: str, meta: Dict) -> str:
    return "|".join([str(meta.get("act", "")), str(meta.get("section", "")), str(doc)[:180]])


def retrieve_legal_provisions_for_document(
    issues: List[Dict],
    acts: List[str],
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
    n_results_per_issue: int = 3,
) -> List[Dict]:
    """Stage 2: retrieve legal act provisions for extracted document issues."""
    client = _client()
    collections = _collection_names_for_acts(client, acts)
    seen = set()
    provisions = []

    issue_queries = []
    for issue in issues:
        query_parts = [
            issue.get("issue", ""),
            issue.get("facts", ""),
            " ".join(issue.get("potential_sections", [])),
        ]
        query = " ".join(part for part in query_parts if part).strip()
        if query:
            issue_queries.append((issue.get("issue", "Legal issue"), query))

    if not issue_queries:
        issue_queries = [("Document legal issues", "legal issues applicable provisions")]

    for issue_label, query in issue_queries:
        for collection_name in collections:
            try:
                collection = client.get_collection(collection_name)
                if collection.count() == 0:
                    continue
                results = retrieve_collection(
                    query_text=query,
                    collection_name=collection_name,
                    collection=collection,
                    n_results=n_results_per_issue,
                    mode=retrieval_mode,
                )
                for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
                    meta = dict(meta or {})
                    meta.setdefault("source", "legal_act")
                    key = _provision_key(doc, meta)
                    if key in seen:
                        continue
                    seen.add(key)
                    provisions.append(
                        {
                            "issue": issue_label,
                            "document": doc,
                            "metadata": meta,
                        }
                    )
            except Exception as exc:
                print(f"Document provision retrieval failed for {collection_name}: {str(exc)}")

    provisions.sort(
        key=lambda item: float(item["metadata"].get("rrf_score") or item["metadata"].get("confidence") or 0),
        reverse=True,
    )
    return provisions[:18]


def _format_provisions(provisions: List[Dict]) -> tuple[str, Dict]:
    lines = []
    source_map = {}
    for index, provision in enumerate(provisions, start=1):
        meta = provision.get("metadata", {})
        doc = provision.get("document", "")
        act = meta.get("act", "UNKNOWN")
        section = meta.get("section_display") or meta.get("section") or "Provision"
        heading = meta.get("heading", "")
        label = f"{act} {section}"
        if heading:
            label = f"{label}: {heading}"
        lines.append(f"[{index}] {label} - {doc}")
        source_map[index] = {
            "act": act,
            "section": str(meta.get("section", section)),
            "heading": heading,
            "snippet": str(doc)[:200],
            "confidence": meta.get("confidence"),
            "source": "legal_act",
            "document_name": None,
        }
    return "\n\n".join(lines), source_map


def generate_document_legal_report(doc_text: str, analysis: Dict, provisions: List[Dict]) -> Dict:
    """Stage 3: generate a structured legal report with cited legal provisions."""
    provision_context, source_map = _format_provisions(provisions)
    prompt = f"""
The uploaded document is the subject of analysis. The legal provisions are reference knowledge.
Use the provisions below and cite every legal claim using [N] before the period.
Do not cite document facts with [N]; use citations only for legal provisions.

Extracted document analysis:
{json.dumps(analysis, indent=2)}

Document excerpt:
{_truncate(doc_text, 10000)}

Retrieved legal provisions:
{provision_context or "No legal provisions retrieved."}

Write a structured report with exactly these sections:
1. Document Overview
2. Applicable Legal Framework
3. Section-by-Section Mapping
4. Identified Legal Issues
5. Strengths and Weaknesses
6. Recommended Actions
7. Risk Assessment
"""
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You prepare grounded Indian legal document analysis reports."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return {
        "report": response.choices[0].message.content,
        "source_map": source_map,
    }


def run_full_document_analysis(
    doc_text: str,
    document_name: str = "uploaded_document",
    retrieval_mode: str = "Hybrid (BM25 + Semantic)",
) -> Dict:
    """Run the full document-first legal analysis pipeline."""
    analysis = extract_legal_issues_from_document(doc_text)
    provisions = retrieve_legal_provisions_for_document(
        issues=analysis.get("legal_issues", []),
        acts=analysis.get("applicable_acts", []),
        retrieval_mode=retrieval_mode,
    )
    report = generate_document_legal_report(doc_text, analysis, provisions)
    return {
        "document_name": document_name,
        "analysis": analysis,
        "provisions": provisions,
        "report": report["report"],
        "source_map": report["source_map"],
    }
