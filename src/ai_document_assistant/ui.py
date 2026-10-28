"""Streamlit user interface for uploading PDFs and asking grounded questions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
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
SUGGESTED_QUESTIONS = (
    "What is the main idea of this document?",
    "List every major concept covered in the document.",
    "What are the most important takeaways?",
)
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
        "messages",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def _display_sidebar() -> None:
    with st.sidebar:
        st.header("Document")
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


def _display_document_viewer() -> None:
    """Display the selected source page beside the conversation."""

    pdf_bytes = st.session_state.get("pdf_bytes")
    page_count = int(st.session_state.get("indexed_page_count", 0))
    source = st.session_state.get("indexed_source", "document.pdf")
    if not pdf_bytes or page_count <= 0:
        return

    st.subheader("Document")
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
        height=650,
        key="selected-page-viewer",
        alt=f"Page {selected_page} of {source}",
    )


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
                _select_page(page_number)


def _display_answer(result: RAGAnswer, key_prefix: str) -> None:
    """Display an answer, its page citations, and retrieval diagnostics."""

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
    with st.expander("Retrieval details"):
        st.caption(f"Retrieved chunks: {len(result.retrieved_chunk_ids)}")
        st.caption(f"Cited sources: {len(result.citations)}")
        if result.diagnostics:
            st.caption(f"Retrieval time: {result.diagnostics.retrieval_ms:.1f} ms")
            st.caption(f"Generation time: {result.diagnostics.generation_ms:.1f} ms")
            st.caption(f"Total time: {result.diagnostics.total_ms:.1f} ms")
            st.caption(f"Model: {result.diagnostics.model or 'configured provider'}")


def _display_feedback(message: dict[str, object], key_prefix: str) -> None:
    feedback_widget = getattr(st, "feedback", None)
    if feedback_widget is None:
        return
    feedback = feedback_widget("thumbs", key=f"{key_prefix}-feedback")
    if feedback is not None:
        message["feedback"] = "up" if feedback == 1 else "down"
        message["feedback_timestamp"] = datetime.now(UTC).isoformat()


def _display_suggested_questions() -> str | None:
    if st.session_state["messages"]:
        return None

    st.caption("Try a suggested question")
    for index, suggested_question in enumerate(SUGGESTED_QUESTIONS):
        if st.button(
            suggested_question,
            key=f"suggested-question-{index}",
            use_container_width=True,
        ):
            return suggested_question
    return None


def _display_chat_history() -> None:
    for index, message in enumerate(st.session_state["messages"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                _display_citations(message["citations"], f"history-{index}")
            if message["role"] == "assistant":
                _display_feedback(message, f"history-{index}")


def main() -> None:
    """Render the Streamlit document chat application."""

    st.set_page_config(page_title="AI Document Assistant", page_icon="PDF")
    st.title("AI Document Assistant")
    st.write("Upload a PDF, then ask questions that are answered using its contents.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
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

    _display_sidebar()

    if st.session_state.get("indexed_source") is None:
        st.info("Upload a PDF to start asking questions.")
        return

    document_column, chat_column = st.columns([1.1, 0.9], gap="large")
    with document_column:
        _display_document_viewer()

    with chat_column:
        st.subheader("Ask questions")
        _display_chat_history()
        suggested_question = _display_suggested_questions()
        question = suggested_question or st.chat_input("Ask a question about the PDF")
        if question is None:
            return

        st.session_state["messages"].append({"role": "user", "content": question})
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
                answer_status.write("Searching the document...")
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
            _display_feedback(assistant_message, "current-answer")

        st.session_state["messages"].append(assistant_message)


if __name__ == "__main__":
    main()
