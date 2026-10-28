"""Streamlit user interface for uploading PDFs and asking grounded questions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAIError
from pypdf import PdfReader, PdfWriter

from ai_document_assistant.application import build_rag_assistant, index_uploaded_pdf
from ai_document_assistant.assistant import ConfigurationError
from ai_document_assistant.documents import DocumentError
from ai_document_assistant.rag import RAGAnswer, RAGAssistant

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
CHAT_PANEL_HEIGHT = 500
PDF_VIEWER_HEIGHT = 620
PDF_READING_HEIGHT = 900
LAYOUT_RATIOS = {
    "Reader focus (65/35)": [1.7, 0.9],
    "Balanced (50/50)": [1.0, 1.0],
    "Chat focus (35/65)": [0.9, 1.7],
}
logger = logging.getLogger(__name__)


def _get_assistant() -> RAGAssistant:
    """Return the assistant for the current browser session."""

    assistant = st.session_state.get("assistant")
    if assistant is None:
        assistant = build_rag_assistant()
        st.session_state["assistant"] = assistant
    return assistant


def _index_upload(
    uploaded_file: Any,
    *,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Index a new uploaded PDF and return its original display name."""

    filename = Path(uploaded_file.name or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise DocumentError("only PDF files are supported")

    content = uploaded_file.getvalue()
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentError("PDF exceeds the 20 MB limit")

    file_hash = hashlib.sha256(content).hexdigest()
    if file_hash == st.session_state.get("indexed_file_hash"):
        return str(st.session_state["indexed_source"])

    assistant = build_rag_assistant()
    summary = index_uploaded_pdf(
        assistant,
        content,
        source_name=filename,
        progress=progress,
    )

    st.session_state["assistant"] = assistant
    st.session_state["indexed_file_hash"] = file_hash
    st.session_state["indexed_source"] = filename
    st.session_state["indexed_page_count"] = summary.page_count
    st.session_state["indexed_chunk_count"] = summary.chunk_count
    st.session_state["pdf_bytes"] = content
    st.session_state["selected_page"] = 1
    st.session_state.pop("selected_citation", None)
    st.session_state.pop("document_search", None)
    st.session_state["messages"] = []
    return filename


def _clear_document() -> None:
    for key in (
        "assistant",
        "indexed_file_hash",
        "indexed_source",
        "indexed_page_count",
        "indexed_chunk_count",
        "pdf_bytes",
        "selected_page",
        "selected_citation",
        "document_search",
        "messages",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def _render_diagnostics(slot: Any, diagnostics: dict[str, object] | None) -> None:
    slot.empty()
    if not diagnostics:
        return

    with slot.container(), st.expander("Diagnostics", expanded=False):
        st.caption(f"Retrieved chunks: {diagnostics['retrieved_count']}")
        st.caption(f"Cited sources: {diagnostics['citation_count']}")
        st.caption(f"Retrieval time: {float(diagnostics['retrieval_ms']):.1f} ms")
        st.caption(f"Generation time: {float(diagnostics['generation_ms']):.1f} ms")
        st.caption(f"Total time: {float(diagnostics['total_ms']):.1f} ms")
        st.caption(f"Model: {diagnostics['model'] or 'configured provider'}")


def _display_sidebar() -> tuple[Any, Any]:
    latest_diagnostics: dict[str, object] | None = None
    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        source = st.session_state.get("indexed_source")
        if source is None:
            st.caption("Upload a PDF to begin.")
        else:
            st.write(source)
            st.caption(f"Pages indexed: {st.session_state.get('indexed_page_count', 0)}")
            st.caption(f"Chunks indexed: {st.session_state.get('indexed_chunk_count', 0)}")
            if st.button("Clear document", use_container_width=True):
                _clear_document()

        feedback_records = [
            {
                "question": message.get("question", ""),
                "answer": message["content"],
                "feedback": message["feedback"],
                "timestamp": message.get("feedback_timestamp"),
            }
            for message in st.session_state.get("messages", [])
            if message.get("role") == "assistant" and "feedback" in message
        ]
        if feedback_records:
            st.download_button(
                "Download feedback data",
                data=json.dumps(feedback_records, indent=2),
                file_name="assistant-feedback.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Workspace controls", expanded=False):
            st.selectbox(
                "Layout",
                list(LAYOUT_RATIOS),
                index=0,
                key="workspace_layout",
                help="Choose how much space the book and assistant receive.",
            )
            st.checkbox(
                "Full-screen reading mode",
                key="reading_mode",
                help="Show the book at full width and move the assistant below it.",
            )

        for message in reversed(st.session_state.get("messages", [])):
            if message.get("role") == "assistant" and message.get("diagnostics"):
                latest_diagnostics = message["diagnostics"]
                break

    diagnostics_slot = st.sidebar.empty()
    _render_diagnostics(diagnostics_slot, latest_diagnostics)
    return diagnostics_slot, uploaded_file


@st.cache_data(show_spinner=False)
def _page_pdf(pdf_bytes: bytes, page_number: int) -> bytes:
    """Return one page as a standalone PDF for the viewer."""

    reader = PdfReader(BytesIO(pdf_bytes))
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError("selected page is outside the document")

    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _select_page(page_number: int) -> None:
    st.session_state["selected_page"] = page_number


def _select_citation(citation: dict[str, object]) -> None:
    st.session_state["selected_page"] = int(citation["page_number"])
    st.session_state["selected_citation"] = citation


@st.cache_data(show_spinner=False)
def _search_pdf(pdf_bytes: bytes, query: str) -> tuple[tuple[int, str], ...]:
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return ()

    reader = PdfReader(BytesIO(pdf_bytes))
    matches: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        position = text.casefold().find(normalized_query)
        if position < 0:
            continue
        start = max(0, position - 90)
        end = min(len(text), position + len(query) + 150)
        matches.append((page_number, text[start:end]))
    return tuple(matches)


@st.cache_data(show_spinner=False)
def _pdf_outline(pdf_bytes: bytes) -> tuple[tuple[str, int, int], ...]:
    reader = PdfReader(BytesIO(pdf_bytes))
    try:
        outline = reader.outline
    except (AttributeError, ValueError):
        return ()

    entries: list[tuple[str, int, int]] = []

    def visit(items: list[object], level: int = 0) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item, level + 1)
                continue
            try:
                page_number = reader.get_destination_page_number(item) + 1
                title = str(item.title).strip()
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if title:
                entries.append((title, page_number, level))

    if isinstance(outline, list):
        visit(outline)
    return tuple(entries)


def _display_selected_excerpt() -> None:
    citation = st.session_state.get("selected_citation")
    if not citation or not str(citation.get("text", "")).strip():
        return

    source = str(citation.get("source", "document"))
    page_number = int(citation["page_number"])
    excerpt = escape(str(citation["text"]).strip())
    st.caption(f"Supporting excerpt from page {page_number}")
    with st.container(height=180, border=True):
        st.markdown(
            f"<div style='padding:0.75rem;border-left:4px solid #4f8bf9;"
            f"background-color:rgba(79,139,249,0.10);'><mark>{excerpt}</mark></div>",
            unsafe_allow_html=True,
        )
    if st.button("Clear highlighted excerpt", key="clear-selected-citation"):
        st.session_state.pop("selected_citation", None)
        st.rerun()
    st.caption(f"{source} - page {page_number}")


def _display_document_tools(pdf_bytes: bytes, source: str) -> None:
    with st.expander("Book tools", expanded=False):
        search_query = st.text_input(
            "Search within book",
            key="document_search",
            placeholder="Find a word or phrase...",
        )
        if search_query.strip():
            results = _search_pdf(pdf_bytes, search_query)
            if not results:
                st.caption("No matching pages found.")
            else:
                st.caption(f"{len(results)} matching page(s)")
                for index, (page_number, snippet) in enumerate(results[:8]):
                    if st.button(
                        f"Page {page_number}: {snippet[:100]}",
                        key=f"search-result-{index}",
                        use_container_width=True,
                    ):
                        _select_citation(
                            {
                                "source": source,
                                "page_number": page_number,
                                "text": snippet,
                                "score": None,
                            }
                        )

        outline = _pdf_outline(pdf_bytes)
        if outline:
            st.caption("Table of contents")
            for index, (title, page_number, level) in enumerate(outline):
                label = f"{'  ' * level}{title} (page {page_number})"
                if st.button(
                    label,
                    key=f"outline-{index}",
                    use_container_width=True,
                ):
                    _select_page(page_number)


def _display_document_viewer(*, reading_mode: bool = False) -> None:
    """Display the selected source page beside the conversation."""

    pdf_bytes = st.session_state.get("pdf_bytes")
    page_count = int(st.session_state.get("indexed_page_count", 0))
    source = st.session_state.get("indexed_source", "document.pdf")
    if not pdf_bytes or page_count <= 0:
        return

    st.subheader("Book")
    if int(st.session_state.get("selected_page", 1)) > page_count:
        st.session_state["selected_page"] = 1
    selected_page = st.number_input(
        "Source page",
        min_value=1,
        max_value=page_count,
        step=1,
        key="selected_page",
    )
    st.caption(f"{source} - page {selected_page} of {page_count}")
    _display_document_tools(pdf_bytes, source)
    _display_selected_excerpt()

    try:
        page_bytes = _page_pdf(pdf_bytes, selected_page)
    except (ValueError, OSError):
        st.error("That PDF page could not be displayed.")
        return

    pdf_viewer = getattr(st, "pdf", None)
    if pdf_viewer is None:
        st.info("Install the PDF viewer extra to display pages inside the app.")
        st.download_button(
            "Download selected page",
            data=page_bytes,
            file_name=f"{Path(source).stem}-page-{selected_page}.pdf",
            mime="application/pdf",
            key="download-selected-page",
        )
        return

    pdf_viewer(
        page_bytes,
        height=PDF_READING_HEIGHT if reading_mode else PDF_VIEWER_HEIGHT,
        key="selected-page-viewer",
        alt=f"Page {selected_page} of {source}",
    )
    st.caption("Use the embedded viewer controls for zoom and fit-to-width.")


def _display_citations(citations: list[dict[str, object]], key_prefix: str) -> None:
    st.caption("Sources - select a page to view it")
    for index, citation in enumerate(citations):
        source = str(citation["source"])
        page_number = int(citation["page_number"])
        score = citation.get("score")
        label = f"Page {page_number} - {source}"
        with st.expander(label):
            if score is not None:
                st.caption(f"Relevance score: {float(score):.3f}")
            text = str(citation.get("text", "")).strip()
            if text:
                st.write(text)
            if st.button(
                "View this page",
                key=f"{key_prefix}-citation-{index}",
                use_container_width=True,
            ):
                _select_citation(citation)


def _display_answer(result: RAGAnswer, key_prefix: str) -> None:
    """Display an answer and its page citations."""

    st.markdown(result.answer)
    if result.citations:
        _display_citations(
            [
                {
                    "source": citation.source,
                    "page_number": citation.page_number,
                    "text": citation.text,
                    "score": citation.score,
                }
                for citation in result.citations
            ],
            key_prefix,
        )
def _display_feedback(message: dict[str, object], key_prefix: str) -> None:
    feedback_widget = getattr(st, "feedback", None)
    if feedback_widget is None:
        return
    feedback = feedback_widget("thumbs", key=f"{key_prefix}-feedback")
    if feedback is not None:
        message["feedback"] = "up" if feedback == 1 else "down"
        message["feedback_timestamp"] = datetime.now(UTC).isoformat()


def _display_chat_history() -> None:
    for index, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                _display_citations(message["citations"], f"history-{index}")
            if message["role"] == "assistant":
                _display_feedback(message, f"history-{index}")


def _diagnostics_payload(result: RAGAnswer) -> dict[str, object] | None:
    if result.diagnostics is None:
        return None
    return {
        "retrieved_count": len(result.retrieved_chunk_ids),
        "citation_count": len(result.citations),
        "retrieval_ms": result.diagnostics.retrieval_ms,
        "generation_ms": result.diagnostics.generation_ms,
        "total_ms": result.diagnostics.total_ms,
        "model": result.diagnostics.model,
    }


def _display_chat_workspace(diagnostics_slot: Any) -> None:
    st.subheader("Ask questions")
    chat_panel = st.container(height=CHAT_PANEL_HEIGHT, border=True)
    with chat_panel:
        _display_chat_history()

    st.markdown(
        "<div class='ask-book-prompt'><strong>Ask the book</strong>"
        "<span>Ask about an idea, argument, character, or page.</span></div>",
        unsafe_allow_html=True,
    )
    question = st.chat_input("Ask a question about the book...")
    if question is None:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with chat_panel:
        with st.chat_message("user"):
            st.markdown(question)

        assistant_message: dict[str, object] = {
            "role": "assistant",
            "question": question,
        }
        with st.chat_message("assistant"), st.status(
            "Preparing answer...", expanded=False
        ) as answer_status:
            try:
                answer_status.write("Retrieving supporting passages...")
                answer_status.write("Generating a grounded answer...")
                result = _get_assistant().answer(question)
            except (OpenAIError, RuntimeError, ValueError):
                answer_status.update(label="Answer failed", state="error")
                logger.exception("Failed to answer a UI question")
                st.error("The AI provider failed while answering the question.")
                return
            answer_status.update(label="Answer ready", state="complete")
            _display_answer(result, f"current-{len(st.session_state['messages'])}")
            assistant_message["content"] = result.answer
            assistant_message["citations"] = [
                {
                    "source": citation.source,
                    "page_number": citation.page_number,
                    "text": citation.text,
                    "score": citation.score,
                }
                for citation in result.citations
            ]
            diagnostics = _diagnostics_payload(result)
            if diagnostics:
                assistant_message["diagnostics"] = diagnostics
                _render_diagnostics(diagnostics_slot, diagnostics)
            _display_feedback(assistant_message, "current-answer")

    st.session_state["messages"].append(assistant_message)


def main() -> None:
    """Render the Streamlit document chat application."""

    st.set_page_config(page_title="AI Document Assistant", page_icon="PDF", layout="wide")
    st.markdown(
        """
        <style>
        .ask-book-prompt {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            margin: 0.75rem 0 0.35rem;
            padding: 0.65rem 0.85rem;
            border: 1px solid rgba(79, 139, 249, 0.45);
            border-radius: 0.75rem;
            background: linear-gradient(135deg, rgba(79, 139, 249, 0.16), rgba(79, 139, 249, 0.04));
        }
        .ask-book-prompt strong {
            color: #2563eb;
            font-size: 1.05rem;
        }
        .ask-book-prompt span {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.85rem;
        }
        [data-testid="stChatInput"] {
            border: 2px solid rgba(79, 139, 249, 0.75);
            border-radius: 0.85rem;
            box-shadow: 0 0 0 3px rgba(79, 139, 249, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI Document Assistant")
    st.write("Upload a PDF, then ask questions that are answered using its contents.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    diagnostics_slot, uploaded_file = _display_sidebar()
    if uploaded_file is not None:
        with st.status("Indexing document...", expanded=True) as indexing_status:
            try:
                source = _index_upload(uploaded_file, progress=indexing_status.write)
            except ConfigurationError as error:
                indexing_status.update(label="Configuration error", state="error")
                st.error(str(error))
            except DocumentError as error:
                indexing_status.update(label="Document error", state="error")
                st.error(str(error))
            except OpenAIError as error:
                indexing_status.update(label="AI provider error", state="error")
                logger.exception("Failed to index uploaded document")
                status = getattr(error, "status_code", None)
                detail = f" (HTTP {status})" if status else ""
                st.error(
                    "The AI provider failed while processing the document"
                    f"{detail}. Check your OpenRouter key, credits, and embedding model."
                )
            except RuntimeError:
                indexing_status.update(label="Indexing failed", state="error")
                logger.exception("Failed to index uploaded document")
                st.error("Document processing produced an invalid embedding response.")
            else:
                indexing_status.update(label="Document ready", state="complete")
                st.success(
                    f"Ready: {source} ({st.session_state['indexed_page_count']} pages indexed)."
                )

    if st.session_state.get("indexed_source") is None:
        st.info("Upload a PDF using the sidebar to start reading and asking questions.")
        return

    if st.session_state.get("reading_mode", False):
        _display_document_viewer(reading_mode=True)
        with st.expander("Ask the assistant", expanded=False):
            _display_chat_workspace(diagnostics_slot)
        return

    layout = st.session_state.get("workspace_layout", "Reader focus (65/35)")
    document_column, chat_column = st.columns(LAYOUT_RATIOS[layout], gap="large")
    with document_column:
        _display_document_viewer()
    with chat_column:
        _display_chat_workspace(diagnostics_slot)


if __name__ == "__main__":
    main()
