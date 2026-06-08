import html
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import streamlit as st

try:
    from .legal_engine import (
        USER_UPLOAD_COLLECTION,
        load_resources,
        run_document_focused_analysis as engine_run_document_focused_analysis,
        run_legal_analysis as engine_run_legal_analysis,
    )
except ImportError:
    from legal_engine import (
        USER_UPLOAD_COLLECTION,
        load_resources,
        run_document_focused_analysis as engine_run_document_focused_analysis,
        run_legal_analysis as engine_run_legal_analysis,
    )


APP_ROOT = Path(__file__).resolve().parents[1]


st.set_page_config(
    page_title="Multi-Act Legal Advisor Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "recent_queries": [],
        "show_context": False,
        "include_uploads": True,
        "retrieval_mode": "Hybrid (BM25 + Semantic)",
        "confidence_threshold": 0.45,
        "debug_uploads": False,
        "doc_analysis": None,
        "active_document_name": None,
        "document_mode": False,
        "last_survivor_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "resources" not in st.session_state:
        st.session_state.resources = load_resources()


def _database_has_collections() -> bool:
    try:
        _, _, client = load_resources()
        return bool(client.list_collections())
    except Exception:
        return False


def _run_ingestion():
    script = APP_ROOT / "scripts" / "embed_all_acts.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(APP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        load_resources.cache_clear()
        st.success("Sample legal database ingested.")
    else:
        st.error("Ingestion failed.")
        st.code(completed.stderr or completed.stdout)


def _render_upload_panel():
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )
    if not uploaded_file:
        return

    document_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("last_uploaded_document") == document_key:
        st.caption("Already indexed in this session.")
        return

    if st.button("Index & Analyze", use_container_width=True):
        progress = st.progress(0, text="Reading document...")
        try:
            try:
                from .document_uploader import chunk_text, embed_and_store, load_document
                from .document_analyzer import run_full_document_analysis
            except ImportError:
                from document_uploader import chunk_text, embed_and_store, load_document
                from document_analyzer import run_full_document_analysis

            text = load_document(uploaded_file)
            progress.progress(35, text="Chunking document...")
            chunks = chunk_text(text)
            progress.progress(60, text="Embedding and storing chunks...")
            indexed = embed_and_store(
                chunks,
                USER_UPLOAD_COLLECTION,
                {"document_name": uploaded_file.name, "heading": uploaded_file.name},
            )
            progress.progress(80, text="Analyzing document through legal framework...")
            doc_analysis = run_full_document_analysis(
                text,
                document_name=uploaded_file.name,
                retrieval_mode=st.session_state.retrieval_mode,
            )
            progress.progress(100, text="Indexed and analyzed")
            st.session_state.last_uploaded_document = document_key
            st.session_state.active_document_name = uploaded_file.name
            st.session_state.doc_analysis = doc_analysis
            st.session_state.document_mode = True
            try:
                _, _, client = load_resources()
                upload_count = client.get_collection(USER_UPLOAD_COLLECTION).count()
            except Exception:
                upload_count = indexed
            st.success(f"Indexed {indexed} chunks into {USER_UPLOAD_COLLECTION}. Current upload chunks: {upload_count}.")
        except Exception as exc:
            st.error(f"Upload failed: {str(exc)}")


def _severity_badge(severity: str) -> str:
    severity = str(severity or "Unknown").title()
    styles = {
        "Critical": "background:#fee2e2;color:#991b1b;",
        "High": "background:#ffedd5;color:#9a3412;",
        "Medium": "background:#fef9c3;color:#854d0e;",
        "Low": "background:#dcfce7;color:#166534;",
    }
    return f"<span style='border-radius:4px;font-weight:700;padding:0.2rem 0.4rem;{styles.get(severity, 'background:#e5e7eb;color:#374151;')}'>{html.escape(severity)}</span>"


def _render_document_analysis_panel():
    doc_analysis = st.session_state.get("doc_analysis")
    if not doc_analysis:
        return

    analysis = doc_analysis.get("analysis", {})
    st.header("Active Document")
    st.caption(st.session_state.get("active_document_name") or doc_analysis.get("document_name"))
    st.markdown(_severity_badge(analysis.get("severity")), unsafe_allow_html=True)

    summary = analysis.get("brief_summary")
    if summary:
        st.write(summary)

    acts = analysis.get("applicable_acts", [])
    if acts:
        st.caption(f"Applicable acts: {', '.join(str(act).upper() for act in acts)}")

    issues = analysis.get("legal_issues", [])
    with st.expander("Key Legal Issues", expanded=True):
        if not issues:
            st.caption("No legal issues extracted.")
        for issue in issues:
            st.markdown(
                f"**{issue.get('issue', 'Issue')}** · {issue.get('severity', 'Medium')}"
            )
            if issue.get("facts"):
                st.caption(issue["facts"])
            if issue.get("potential_sections"):
                st.caption(f"Potential sections: {', '.join(issue['potential_sections'])}")

    with st.expander("Full Legal Report", expanded=False):
        st.markdown(doc_analysis.get("report") or "No report generated.")

    if st.button("Clear Document", use_container_width=True):
        st.session_state.doc_analysis = None
        st.session_state.active_document_name = None
        st.session_state.last_uploaded_document = None
        st.session_state.document_mode = False
        st.rerun()


def render_sidebar():
    """All sidebar controls and tools."""
    with st.sidebar:
        st.title("Control Panel")

        st.header("Chat")
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.header("Retrieval")
        document_ready = bool(st.session_state.get("doc_analysis"))
        st.session_state.document_mode = st.toggle(
            "Document mode",
            value=bool(st.session_state.document_mode and document_ready),
            disabled=not document_ready,
            help="ON: analyze the active uploaded document through legal acts. OFF: use the generic legal RAG pipeline.",
        )
        if st.session_state.document_mode:
            st.caption("Document pipeline: doc chunks -> legal points -> law collections -> final synthesis.")
        elif document_ready:
            st.caption("Generic pipeline is active. Turn on Document mode to analyze the upload as the subject.")
        st.session_state.include_uploads = st.toggle(
            "Include my documents",
            value=st.session_state.include_uploads,
            disabled=st.session_state.document_mode,
            help="Generic-pipeline option. Document mode always retrieves the active document separately.",
        )
        st.session_state.retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["Hybrid (BM25 + Semantic)", "Semantic only", "Keyword only"],
            index=["Hybrid (BM25 + Semantic)", "Semantic only", "Keyword only"].index(st.session_state.retrieval_mode),
        )
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.3,
            max_value=0.9,
            value=float(st.session_state.confidence_threshold),
            step=0.05,
            help="Retrieved chunks below this confidence are filtered. If fewer than two survive, the engine retries at 0.30.",
        )
        st.caption(f"Current answer sources above threshold: {st.session_state.last_survivor_count}")
        st.session_state.debug_uploads = st.toggle(
            "Debug uploaded document retrieval",
            value=st.session_state.debug_uploads,
            help="Print and display raw user_uploads chunks retrieved for each query.",
        )

        st.header("Display")
        st.session_state.show_context = st.toggle(
            "Show retrieval context",
            value=st.session_state.show_context,
            help="Reveal the exact legal text passages used for generating answers.",
        )

        _render_upload_panel()
        _render_document_analysis_panel()

        if not _database_has_collections():
            st.header("Setup")
            st.warning("No preloaded legal database found.")
            if st.button("Ingest sample data", use_container_width=True):
                _run_ingestion()

        if st.session_state.recent_queries:
            st.header("Recent Queries")
            for i, query in enumerate(st.session_state.recent_queries[-6:]):
                if st.button(query[:36], key=f"recent_{i}", use_container_width=True):
                    process_query(query)

        st.header("Export")
        export_format = st.radio("Format", ["Markdown", "JSON"], horizontal=True)
        export_conversation(export_format)


def export_conversation(format: str):
    """Generate downloadable conversation exports."""
    if not st.session_state.messages:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_act_conversation_{timestamp}"

    if format == "Markdown":
        content = ["# Multi-Act Legal Advisor Conversation\n"]
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Legal Advisor"
            content.append(f"## {role}\n{msg['content']}\n")
        data = "\n".join(content)
        st.download_button("Download Markdown", data=data, file_name=f"{filename}.md", use_container_width=True)
    else:
        data = json.dumps(st.session_state.messages, indent=2)
        st.download_button("Download JSON", data=data, file_name=f"{filename}.json", use_container_width=True)


def run_legal_analysis(query: str) -> Dict:
    """Run the engine and add UI timing metadata."""
    start_time = time.time()
    response = engine_run_legal_analysis(
        query,
        include_uploads=st.session_state.include_uploads,
        retrieval_mode=st.session_state.retrieval_mode,
        confidence_threshold=st.session_state.confidence_threshold,
        debug_uploads=st.session_state.debug_uploads,
    )
    total_time = time.time() - start_time
    st.session_state.last_survivor_count = len(response.get("source_map", {}))
    response["timing"] = {"total": round(total_time * 1000)}
    return response


def run_document_qa(query: str) -> Dict:
    """Run document-focused Q&A where the upload is the subject."""
    start_time = time.time()
    response = engine_run_document_focused_analysis(
        query,
        document_name=st.session_state.get("active_document_name"),
        retrieval_mode=st.session_state.retrieval_mode,
        confidence_threshold=st.session_state.confidence_threshold,
    )
    total_time = time.time() - start_time
    st.session_state.last_survivor_count = len(response.get("source_map", {}))
    response["timing"] = {"total": round(total_time * 1000)}
    return response


def _citation_markup(content: str) -> str:
    escaped = html.escape(content)
    return re.sub(
        r"\[((?:DOC|LAW)-\d+|\d+)\]",
        r'<span class="citation">[\1]</span>',
        escaped,
    ).replace("\n", "<br>")


def _confidence_badge(score):
    if score is None:
        return '<span class="badge badge-low">Unknown</span>'
    try:
        score = float(score)
    except (TypeError, ValueError):
        return '<span class="badge badge-low">Unknown</span>'
    if score > 0.75:
        return f'<span class="badge badge-high">High {score:.2f}</span>'
    if score >= 0.5:
        return f'<span class="badge badge-med">Medium {score:.2f}</span>'
    return f'<span class="badge badge-low">Low {score:.2f}</span>'


def _render_sources(source_map: Dict):
    if not source_map:
        return

    with st.expander("Sources", expanded=False):
        for number, source in source_map.items():
            act = html.escape(str(source.get("act", "UNKNOWN")))
            section = html.escape(str(source.get("section", "")))
            heading = html.escape(str(source.get("heading", "")))
            snippet = html.escape(str(source.get("snippet", "")))
            document_name = source.get("document_name")
            source_label = f"{act} · {section}"
            if document_name:
                source_label = f"{source_label} · {html.escape(str(document_name))}"
            st.markdown(
                f"""
<div class="source-row">
  <div><span class="source-number">[{number}]</span> <span class="act-badge">{source_label}</span> {_confidence_badge(source.get("confidence"))}</div>
  <div class="source-heading">{heading}</div>
  <blockquote>{snippet}</blockquote>
</div>
""",
                unsafe_allow_html=True,
            )


def _render_sections(sections):
    if not sections:
        return

    st.divider()
    st.markdown("**Referenced Sections**")
    sections_by_act = {}
    for sec in sections:
        sections_by_act.setdefault(sec.get("act", "UNKNOWN"), []).append(sec)

    for act, act_sections in sections_by_act.items():
        with st.expander(act, expanded=False):
            for section in act_sections:
                heading = f": {section['heading']}" if section.get("heading") else ""
                st.markdown(
                    f"{section['display']}{heading} {_confidence_badge(section.get('confidence'))}",
                    unsafe_allow_html=True,
                )


def _render_upload_debug(debug_info: Dict):
    if not debug_info or not st.session_state.get("debug_uploads"):
        return

    upload_debug = debug_info.get("user_uploads", {})
    with st.expander("Uploaded Document Retrieval Debug", expanded=True):
        st.write(
            {
                "include_uploads": debug_info.get("include_uploads"),
                "retrieval_query": debug_info.get("retrieval_query"),
                "retrieval_mode": debug_info.get("retrieval_mode"),
                "confidence_threshold": debug_info.get("confidence_threshold"),
                "collection_present": upload_debug.get("collection_present"),
                "collection_count": upload_debug.get("collection_count"),
                "raw_match_count": len(upload_debug.get("raw_matches", [])),
                "kept_after_filter_count": len(upload_debug.get("kept_matches", [])),
                "error": upload_debug.get("error"),
            }
        )

        st.markdown("**Raw matches from `user_uploads` before confidence filtering**")
        raw_matches = upload_debug.get("raw_matches", [])
        if not raw_matches:
            st.caption("No uploaded-document chunks were retrieved before filtering.")
        for match in raw_matches:
            st.markdown(
                f"**#{match.get('rank')} · {match.get('document_name') or 'uploaded document'}** "
                f"confidence={match.get('confidence')} mode={match.get('retrieval_mode')}"
            )
            st.code(match.get("snippet") or "", language="text")


def _render_document_pipeline_debug(debug_info: Dict):
    if not debug_info or not st.session_state.get("debug_uploads"):
        return
    if not debug_info.get("document_focused"):
        return

    with st.expander("Document Pipeline Debug", expanded=True):
        st.write(
            {
                "active_document": debug_info.get("active_document"),
                "document_chunks": debug_info.get("document_chunks"),
                "law_chunks": debug_info.get("law_chunks"),
            }
        )
        st.markdown("**Extracted legal points**")
        st.json(debug_info.get("legal_points") or {})

        st.markdown("**Uploaded chunks kept after confidence filtering**")
        kept_matches = upload_debug.get("kept_matches", [])
        if not kept_matches:
            st.caption("No uploaded-document chunks survived the current confidence threshold.")
        for match in kept_matches:
            st.markdown(
                f"**#{match.get('rank')} · {match.get('document_name') or 'uploaded document'}** "
                f"confidence={match.get('confidence')} mode={match.get('retrieval_mode')}"
            )
            st.code(match.get("snippet") or "", language="text")


def render_message(role: str, content: str, **kwargs):
    """Render a chat message with optional metadata."""
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(_citation_markup(content), unsafe_allow_html=True)
            _render_sources(kwargs.get("source_map") or {})
            _render_sections(kwargs.get("sections") or [])
            _render_upload_debug(kwargs.get("debug") or {})
            _render_document_pipeline_debug(kwargs.get("debug") or {})

            if st.session_state.show_context and kwargs.get("context_preview"):
                with st.expander("View Retrieved Context"):
                    for doc in kwargs["context_preview"]:
                        st.caption(doc)
                        st.divider()

            with st.expander("Performance Metrics", expanded=False):
                if kwargs.get("timing"):
                    st.metric("Total Analysis Time", f"{kwargs['timing']['total']}ms")
        else:
            st.markdown(content)


def process_query(query: str):
    """Handle a new user query end-to-end."""
    if not query.strip():
        st.warning("Please enter a valid question")
        return

    if not st.session_state.recent_queries or st.session_state.recent_queries[-1] != query:
        if len(st.session_state.recent_queries) >= 10:
            st.session_state.recent_queries.pop(0)
        st.session_state.recent_queries.append(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Analyzing across legal sources..."):
        try:
            if st.session_state.get("document_mode") and st.session_state.get("doc_analysis"):
                response = run_document_qa(query)
            else:
                response = run_legal_analysis(query)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["analysis"],
                    "sections": response["sections"],
                    "context_preview": response["context_preview"],
                    "source_map": response["source_map"],
                    "timing": response["timing"],
                    "debug": response.get("debug", {}),
                }
            )
        except Exception as exc:
            st.error(f"Analysis failed: {str(exc)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I couldn't process that request. Please try rephrasing or ask about a different legal topic.",
                }
            )


def _render_styles():
    st.markdown(
        """
<style>
.citation {
  background: #e8f1ff;
  border: 1px solid #9bbcff;
  border-radius: 4px;
  color: #174ea6;
  font-weight: 700;
  padding: 0 0.25rem;
}
.badge, .act-badge, .source-number {
  border-radius: 4px;
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 700;
  line-height: 1;
  margin-left: 0.25rem;
  padding: 0.22rem 0.35rem;
}
.badge-high { background: #dff4e7; color: #176b3a; }
.badge-med { background: #fff4cc; color: #7a5a00; }
.badge-low { background: #eceff3; color: #4d5965; }
.act-badge { background: #f1f5f9; color: #233143; margin-left: 0; }
.source-number { background: #111827; color: #ffffff; margin-left: 0; }
.source-row {
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 0.9rem;
  padding-bottom: 0.8rem;
}
.source-heading {
  color: #4b5563;
  font-size: 0.9rem;
  margin-top: 0.35rem;
}
blockquote {
  border-left: 3px solid #cbd5e1;
  color: #374151;
  margin: 0.55rem 0 0;
  padding-left: 0.7rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def main():
    init_session_state()
    _render_styles()

    st.title("Multi-Act Legal Advisor Pro")
    st.caption(
        "Analyze BNS, IPC, CrPC, CPC, BSA, and your uploaded legal documents. "
        "Disclaimer: not a substitute for professional legal advice."
    )

    render_sidebar()

    for msg in st.session_state.messages:
        render_message(
            msg["role"],
            msg["content"],
            sections=msg.get("sections"),
            context_preview=msg.get("context_preview"),
            source_map=msg.get("source_map"),
            timing=msg.get("timing"),
            debug=msg.get("debug"),
        )

    if prompt := st.chat_input("Ask about any legal section, document, or scenario..."):
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
