import streamlit as st

try:
    from .legal_engine import (
        build_sections,
        run_legal_analysis,
    )
except ImportError:
    from legal_engine import (
        build_sections,
        run_legal_analysis,
    )


def _confidence_label(score):
    if score is None:
        return "Unknown"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    if score > 0.75:
        return f"High ({score:.2f})"
    if score >= 0.5:
        return f"Medium ({score:.2f})"
    return f"Low ({score:.2f})"


def display_referenced_sections(results, analysis_text=""):
    """Display retrieved section metadata grouped by act."""
    sections = build_sections(results, analysis_text)
    if not sections:
        st.warning("No matching section details available")
        return False

    st.markdown("**Referenced Sections**")
    sections_by_act = {}
    for section in sections:
        sections_by_act.setdefault(section.get("act", "UNKNOWN"), []).append(section)

    for act, act_sections in sections_by_act.items():
        with st.expander(f"{act}", expanded=True):
            for section in act_sections:
                confidence = _confidence_label(section.get("confidence"))
                heading = f": {section['heading']}" if section.get("heading") else ""
                st.write(f"- {section['display']}{heading} · {confidence}")
    return True


def main():
    st.set_page_config(page_title="Multi-Act Legal Advisor", layout="wide")
    st.title("Multi-Act Legal Advisor")
    st.caption("Supporting BNS, IPC, CrPC, CPC, BSA, and uploaded documents")

    query = st.text_area("Describe your legal scenario or question:")
    include_uploads = st.toggle("Include my documents", value=True)
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["Hybrid (BM25 + Semantic)", "Semantic only", "Keyword only"],
    )
    confidence_threshold = st.slider("Confidence threshold", 0.3, 0.9, 0.45, 0.05)

    if st.button("Analyze"):
        if not query.strip():
            st.warning("Please enter a query")
            return

        with st.spinner("Analyzing across legal sources..."):
            try:
                response = run_legal_analysis(
                    query,
                    include_uploads=include_uploads,
                    retrieval_mode=retrieval_mode,
                    confidence_threshold=confidence_threshold,
                )
                st.markdown("## Legal Analysis")
                st.markdown(response["analysis"])
                display_referenced_sections(response["results"], response["analysis"])
            except Exception as exc:
                st.error(f"Analysis failed: {str(exc)}")
                st.info("Please try rephrasing your query")


if __name__ == "__main__":
    main()
